import json
import os
import re
import unicodedata
import uuid
from collections import Counter

import pdfplumber
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.config import (
    LAST_DOCUMENT_PATH,
    OCR_PREPEND_SECTION_TITLE,
    OCR_TABLE_MODE,
    SCAN_TEXT_THRESHOLD,
    TABLE_EXTRACTOR,
)
from app.ingestion.ocr import ocr_page_to_markdown, split_markdown_blocks
from app.rag.vector_store import get_vector_store


def save_last_document(meta: dict) -> None:
    """Persist a pointer to the most recently ingested document.

    Lets a later query session discover which document to search without the
    user (or the LLM) needing to know its document_id.
    """
    os.makedirs(os.path.dirname(LAST_DOCUMENT_PATH), exist_ok=True)
    with open(LAST_DOCUMENT_PATH, "w") as f:
        json.dump(meta, f, indent=2)


def get_latest_document_id():
    """Return the document_id of the most recently ingested document, or None."""
    if not os.path.exists(LAST_DOCUMENT_PATH):
        return None
    with open(LAST_DOCUMENT_PATH) as f:
        return json.load(f).get("document_id")


def _clean_cell(cell) -> str:
    """Normalize a table cell for Markdown (None -> "", collapse newlines)."""
    if cell is None:
        return ""
    return " ".join(str(cell).split())


def table_to_markdown(rows) -> str:
    """Convert a pdfplumber table (list[list[cell]]) to a Markdown table.

    First row is treated as the header. Returns "" if there is nothing usable.
    """
    cleaned = [
        [_clean_cell(cell) for cell in row]
        for row in rows
        if row is not None
    ]

    cleaned = [row for row in cleaned if any(cell for cell in row)]

    if not cleaned:
        return ""

    num_cols = max(len(row) for row in cleaned)

    def pad(row):
        return row + [""] * (num_cols - len(row))

    header = pad(cleaned[0])
    separator = ["---"] * num_cols
    body = [pad(row) for row in cleaned[1:]]

    lines = [
        "| " + " | ".join(header) + " |",
        "| " + " | ".join(separator) + " |",
    ]
    for row in body:
        lines.append("| " + " | ".join(row) + " |")

    return "\n".join(lines)


def _not_within_bboxes(obj, bboxes) -> bool:
    """True if the object's center is outside every table bounding box."""
    cx = (obj["x0"] + obj["x1"]) / 2
    cy = (obj["top"] + obj["bottom"]) / 2

    for x0, top, x1, bottom in bboxes:
        if x0 <= cx <= x1 and top <= cy <= bottom:
            return False

    return True


def _boilerplate_lines(pages, threshold_ratio: float = 0.3) -> set:
    """Lines repeated across many pages (headers/footers/verification watermarks).

    Generalizes over any repeated boilerplate (e.g. a signing-service watermark)
    without hardcoding specific strings, so it is not mistaken for real content.
    """
    counter = Counter()
    for page in pages:
        text = page.extract_text() or ""
        for line in {ln.strip() for ln in text.split("\n") if ln.strip()}:
            counter[line] += 1

    cutoff = max(2, int(len(pages) * threshold_ratio))
    return {line for line, count in counter.items() if count >= cutoff}


def _meaningful_text(page, boilerplate: set) -> str:
    """Page text minus boilerplate lines and whitespace."""
    text = page.extract_text() or ""
    lines = [
        ln for ln in text.split("\n")
        if ln.strip() and ln.strip() not in boilerplate
    ]
    return "\n".join(lines).strip()


def _is_scanned_page(page, boilerplate: set) -> bool:
    """A page is scanned if it carries an image but almost no real text."""
    has_image = bool(page.images)
    real_len = len(_meaningful_text(page, boilerplate))
    return has_image and real_len < SCAN_TEXT_THRESHOLD


def _find_caption(page, bbox, max_gap: float = 25.0):
    """Find the narrative line immediately above a table's bounding box.

    Returns a ``(text, line_bbox)`` tuple for the closest text line whose
    bottom sits just above the table top (within ``max_gap`` points). The
    line_bbox lets the caller drop that line from the narrative text so it is
    not duplicated. Returns ``("", None)`` if no suitable caption is found.
    """
    _, table_top, _, _ = bbox

    words = page.extract_words()

    best_line = None
    best_bottom = None

    for word in words:
        if word["bottom"] > table_top:
            continue
        # Track the line closest above the table top.
        if best_bottom is None or word["bottom"] > best_bottom + 3:
            best_bottom = word["bottom"]
            best_line = [word]
        elif abs(word["bottom"] - best_bottom) <= 3:
            best_line.append(word)

    if not best_line:
        return "", None

    # Only treat it as a caption if it sits close above the table; otherwise
    # it is ordinary narrative that should stay in the text stream.
    if table_top - best_bottom > max_gap:
        return "", None

    best_line.sort(key=lambda w: w["x0"])
    text = " ".join(w["text"] for w in best_line).strip()

    line_bbox = (
        min(w["x0"] for w in best_line),
        min(w["top"] for w in best_line),
        max(w["x1"] for w in best_line),
        max(w["bottom"] for w in best_line),
    )

    return text, line_bbox


# A line is a column header (not a data row) if it names a period/year, a
# currency, or a statement column label — matched generically across EN/VI so
# no wording is hardcoded.
_HEADER_LINE_RE = re.compile(
    r"\d{1,2}/\d{1,2}/\d{4}"        # 31/12/2023
    r"|\b(?:19|20)\d{2}\b"          # a 4-digit year: 2023 / 2022
    r"|\bVND\b"
    r"|\bCode\b|\bNote\b"
    r"|M[ãa]\s*s[ốo]"              # "Mã số"
    r"|Thuy[eế]t\s*minh"           # "Thuyết minh"
    r"|Đơn\s*vị",                   # "Đơn vị: VND"
    re.IGNORECASE,
)

# Above this many characters, a "cell" is a blob mis-extraction (a whole text
# column mashed into one cell), not a real financial cell.
_MAX_TABLE_CELL_CHARS = 180

# A cell holds a financial figure: a digit followed by >=4 more digit/.,
# characters (so "17,647,627,338,990" or "5.173.881.628.997" match, but a bare
# year like "2023" or a code like "270" does not).
_DATA_CELL_RE = re.compile(r"\d[\d.,]{4,}")


def _is_data_row(row) -> bool:
    """True if any cell in the row carries a financial figure."""
    return any(_DATA_CELL_RE.search(cell or "") for cell in row)


def _header_lines_above(page, top_y: float, floor_y: float, gap: float = 70.0):
    """Header-like text lines sitting just above a table top.

    pdfplumber often leaves the period/currency header (e.g. "31/12/2023
    1/1/2023 / VND VND") out of the detected table because that band has no
    ruling grid. We recover it from the page words in the strip between
    ``floor_y`` (bottom of the previous table, exclusive) and ``top_y`` (this
    table's top), keeping only header-like lines.

    Returns ``(texts, bboxes)`` — the header strings plus their bounding boxes
    so the caller can drop them from the narrative stream (avoid duplication).
    """
    lower = max(top_y - gap, floor_y)
    words = [w for w in page.extract_words() if lower < w["bottom"] <= top_y]

    lines: dict = {}
    for w in words:
        lines.setdefault(round(w["bottom"]), []).append(w)

    texts, bboxes = [], []
    for key in sorted(lines):
        ws = sorted(lines[key], key=lambda w: w["x0"])
        text = " ".join(w["text"] for w in ws).strip()
        if not text or not _HEADER_LINE_RE.search(text):
            continue
        texts.append(text)
        bboxes.append(
            (
                min(w["x0"] for w in ws),
                min(w["top"] for w in ws),
                max(w["x1"] for w in ws),
                max(w["bottom"] for w in ws),
            )
        )
    return texts, bboxes


def _merge_table_fragments(page):
    """Reconstruct logical tables from pdfplumber's fragmented output.

    pdfplumber's default (line-based) detection keeps columns clean but splits a
    financial statement into many 1-row fragments wherever the ruling lines are
    discontinuous — and drops the shared year/currency header. We keep the
    default column detection and stitch the fragments back:

      1. Fragments in reading order (top -> bottom), empties removed.
      2. Merge consecutive fragments while the column count is unchanged AND no
         fresh header band appears above them; a change in either starts a new
         logical table (so two distinct same-width tables on one page stay
         separate).
      3. Capture the header band above each logical table's first fragment.
      4. Keep only logical tables that actually carry financial figures (drops
         title/furniture blocks pdfplumber mistakes for tables).

    Returns ``(logical_tables, excluded_bboxes)`` where each logical table is a
    dict ``{header, rows, top_bbox}`` and ``excluded_bboxes`` collects every
    region (table bodies + recovered headers) to remove from narrative text.
    """
    raw_tables = page.find_tables()

    # Only regions we actually emit as tables are excluded from the narrative
    # stream. We deliberately do NOT exclude dropped "blob" bboxes: a blob often
    # spans the whole statement and its region contains "loose" data lines that
    # pdfplumber never boxed as table rows (e.g. Goodwill, Share capital,
    # Dividends paid). Those must survive as narrative text so they stay
    # searchable.
    excluded_bboxes = []

    frags = []
    for t in raw_tables:
        rows = [r for r in t.extract() if any(c for c in r)]
        if not rows:
            continue
        # Drop "blob" mis-extractions: pdfplumber sometimes returns a wide table
        # that crams a whole text column (title + every line) into one cell,
        # spatially overlapping the real line-based table. A genuine financial
        # cell is short (a label or a figure); a blob cell runs to hundreds of
        # chars. Dropping it here (before merging) also stops it from stealing
        # the header band and starving the real table of it.
        max_cell = max((len(c or "") for r in rows for c in r), default=0)
        if max_cell > _MAX_TABLE_CELL_CHARS:
            continue
        frags.append((t.bbox, rows))
        excluded_bboxes.append(t.bbox)
    frags.sort(key=lambda f: f[0][1])  # by top (y grows downward)

    logical = []
    prev_bottom = 0.0
    for bbox, rows in frags:
        ncols = max(len(r) for r in rows)
        header_texts, header_bboxes = _header_lines_above(page, bbox[1], prev_bottom)
        excluded_bboxes.extend(header_bboxes)

        start_new = (
            not logical
            or ncols != logical[-1]["ncols"]
            or bool(header_texts)
        )
        if start_new:
            logical.append(
                {
                    "header": header_texts,
                    "rows": list(rows),
                    "ncols": ncols,
                    "top_bbox": bbox,
                }
            )
        else:
            logical[-1]["rows"].extend(rows)

        prev_bottom = bbox[3]

    logical = [lt for lt in logical if any(_is_data_row(r) for r in lt["rows"])]
    return logical, excluded_bboxes


def _extract_text_page(page, document_id: str, filename: str, page_number: int):
    """Native extraction for a text-based page: merged table chunks + narrative.

    Returns ``(table_documents, text_documents)``.
    """
    table_documents = []
    text_documents = []

    logical_tables, excluded_bboxes = _merge_table_fragments(page)

    # --- Tables: one atomic Markdown chunk per reconstructed logical table ---
    for table_index, lt in enumerate(logical_tables):
        markdown = table_to_markdown(lt["rows"])
        if not markdown:
            continue

        caption, caption_bbox = _find_caption(page, lt["top_bbox"])
        if caption_bbox is not None:
            excluded_bboxes.append(caption_bbox)

        parts = []
        if caption and caption not in lt["header"]:
            parts.append(f"## {caption}")
        # Prepend the recovered period/currency header so the LLM can tell which
        # value column is the current year vs the comparative year.
        parts.extend(lt["header"])
        parts.append(markdown)
        content = "\n".join(parts)

        table_documents.append(
            Document(
                page_content=content,
                metadata={
                    "document_id": document_id,
                    "filename": filename,
                    "page": page_number,
                    "content_type": "table",
                    "table_index": table_index,
                    "source": "native",
                },
            )
        )

    # --- Narrative text: exclude table + header + caption regions ---
    if excluded_bboxes:
        filtered = page.filter(
            lambda obj: _not_within_bboxes(obj, excluded_bboxes)
        )
        text = filtered.extract_text()
    else:
        text = page.extract_text()

    if text and text.strip():
        text_documents.append(
            Document(
                page_content=text,
                metadata={
                    "document_id": document_id,
                    "filename": filename,
                    "page": page_number,
                    "content_type": "text",
                    "source": "native",
                },
            )
        )

    return table_documents, text_documents


# A context line worth carrying onto a table chunk: the statement date or the
# currency/unit band. Docling emits these as plain lines just under the ## heading
# (e.g. "Tại ngày 31 tháng 12 năm 2025", "Đơn vị: VND"). Matched generically.
_OCR_CONTEXT_LINE_RE = re.compile(
    r"\bVND\b"
    r"|Đơn\s*vị"                    # "Đơn vị: VND"
    r"|T[ạa]i\s*ngày"              # "Tại ngày ..."
    r"|\d{1,2}/\d{1,2}/\d{4}"      # 31/12/2025
    r"|th[áa]ng\s*\d{1,2}\s*năm\s*\d{4}",  # "tháng 12 năm 2025"
    re.IGNORECASE,
)

# A heading that is the running company name, not a statement title (page 11 carries
# both; the statement title is the more specific one). Skipped when another heading
# exists on the block.
_OCR_COMPANY_HEADING_RE = re.compile(r"CÔNG\s*TY|CÔNGTY", re.IGNORECASE)

# VAS form code in a statement header: "Mẫu số B 01 - DN/HN" (B01=cân đối, B02=KQKD,
# B03=lưu chuyển; B09=notes -> not a statement). Matched on the DIACRITIC-FOLDED text
# (see _fold), so it also survives OCR that strips tone marks ("Mẫu"->"mau"). Requires
# the "mau" prefix to avoid false positives from a stray "b1"; tolerant to spacing and a
# missing leading zero. Only B01/B02/B03 carry a statement kind.
_OCR_FORM_CODE_RE = re.compile(r"mau\s*(?:so\s*)?b\s*0?([123])\b")


def _ocr_section_context(text_block: str) -> str:
    """Section title (+ date/currency lines) recovered from an OCR text block.

    Docling puts the statement heading (``## BẢNG CÂN ĐỐI KẾ TOÁN``) and its
    date/unit lines in a text block right above the table's pipe rows, which
    ``split_markdown_blocks`` then separates out. We pull them back so they can be
    prepended to the following table chunk (mirrors the native ``## {caption}`` +
    header pattern). Returns ``""`` if no ``##`` heading is present.

    Heading choice: the LAST ``##`` line (most specific), skipping a company-name
    heading when a more specific one exists (page 11 has both).
    """
    lines = [ln.strip() for ln in text_block.split("\n") if ln.strip()]

    headings = [ln[2:].strip() for ln in lines if ln.startswith("##")]
    if not headings:
        return ""
    specific = [h for h in headings if not _OCR_COMPANY_HEADING_RE.search(h)]
    title = (specific or headings)[-1]

    # Mặc định lấy heading CUỐI (giả định = tên báo cáo cụ thể nhất). Với vài layout
    # (vd SAB) thứ tự ĐẢO: tên báo cáo ở đầu, dòng cuối là mã biểu mẫu/số thuyết minh
    # -> title mặc định không mang statement_kind. CHỈ khi đó mới quét TOÀN BỘ headings
    # (kể cả dòng bị lọc "công ty", vì OCR có thể nhập chung "Công ty ... Bảng cân đối")
    # tìm heading mang kind. Trang mà [-1] vốn đã có kind (vd DHG) KHÔNG bị đổi.
    if _ocr_statement_kind(title) == "":
        for h in reversed(headings):
            if _ocr_statement_kind(h):
                title = h
                break

    context = [
        ln for ln in lines
        if not ln.startswith("##")
        and "|" not in ln
        and _OCR_CONTEXT_LINE_RE.search(ln)
    ]

    return "\n".join([f"## {title}", *context])


def _section_title_only(section_context: str) -> str:
    """Just the statement title from an ``_ocr_section_context`` string (drop date/unit)."""
    if not section_context:
        return ""
    return section_context.split("\n", 1)[0].lstrip("#").strip()


def _fold(s: str) -> str:
    """Lowercase + bỏ dấu tiếng Việt (đ->d, tone marks removed).

    OCR đôi khi rụng dấu ("LƯU"->"LUU", "TỆ"->"TU"), khiến so khớp có-dấu trượt. Fold cả
    tín hiệu lẫn chữ ký về dạng không dấu để bắt được cả bản OCR sạch lẫn bản mất dấu.
    """
    s = (s or "").lower().replace("đ", "d")
    return "".join(c for c in unicodedata.normalize("NFD", s) if unicodedata.category(c) != "Mn")


# Statement-kind signatures. Matched on the DIACRITIC-FOLDED text so OCR that drops tone
# marks still hits ("luu chuyen", "ket qua", "can d"). Mirrors camelot_extract._statement_kind
# but for VI titles/blocks. The VAS form code (B01/B02/B03) is the most reliable, layout-
# independent signal and is checked last as a fallback.
def _ocr_statement_kind(section_title: str) -> str:
    s = _fold(section_title)
    if "luu chuyen" in s:
        return "cash_flow"
    if "ket qua" in s and "kinh doanh" in s:
        return "income_statement"
    if "bang can" in s or "can d" in s:
        return "balance_sheet"
    return _ocr_form_code_kind(s)


def _ocr_form_code_kind(folded_text: str) -> str:
    """statement_kind từ mã biểu mẫu VAS trong CHUỖI ĐÃ FOLD. "" nếu không có/không phải BCTC."""
    m = _OCR_FORM_CODE_RE.search(folded_text)
    if m:
        return {"1": "balance_sheet", "2": "income_statement", "3": "cash_flow"}[m.group(1)]
    return ""


def _ocr_block_statement_kind(text_block: str) -> str:
    """statement_kind suy từ MỘT text block, dựa trên MÃ BIỂU MẪU xuất hiện bất kỳ đâu.

    Docling xuất tiêu đề/mã biểu mẫu KHÔNG nhất quán: có trang là heading ``##`` (bắt qua
    _ocr_section_context), có trang (vd SAB tr.10 KQKD) chỉ là dòng text thường -> tiêu đề
    lọt lưới. Mã biểu mẫu ("Mẫu B 0X") chỉ nằm ở phần đầu bảng BCTC nên quét CẢ block để bắt
    nó là an toàn (không dùng chữ ký tên-báo-cáo trên cả block vì văn xuôi thuyết minh có thể
    nhắc "lưu chuyển tiền"/"kết quả kinh doanh" -> dương tính giả)."""
    return _ocr_form_code_kind(_fold(text_block))


# Positional fallback for a value column whose header cell is blank (VI year columns).
def _positional_vi(value_index: int) -> str:
    return {0: "số cuối năm", 1: "số đầu năm"}.get(value_index, "giá trị")


# Nhãn cột theo VỊ TRÍ, phân biệt loại báo cáo. Bảng cân đối dùng "số cuối/đầu năm";
# KQKD & LCTT dùng "năm nay/năm trước" (cột trái = kỳ hiện tại). Dùng khi bảng OCR không
# có dòng header năm hợp lệ (vd LCTT bị tách header sang block khác).
def _positional_period(value_index: int, statement_kind: str) -> str:
    if statement_kind == "balance_sheet":
        return _positional_vi(value_index)
    return {0: "năm nay", 1: "năm trước"}.get(value_index, "giá trị")


# Một ô/dòng là HEADER kỳ khi mang dấu hiệu năm / VND / ngày / nhãn cột VAS (mã/thuyết
# minh/số) — KHÔNG phải con số tài chính. Dùng để nhận diện dòng header thật (kể cả header
# 2 dòng "2025"+"VND"), tránh lấy nhầm một dòng dữ liệu làm header.
_OCR_HEADER_MARKER_RE = re.compile(
    r"(?:19|20)\d\d"                 # năm 19xx/20xx
    r"|\bVND\b"
    r"|\d{1,2}/\d{1,2}/\d{4}"        # 31/12/2025
    r"|m[ãa]\s*số|thuyết\s*minh|\bmã\b|\bnote\b|\bcode\b",
    re.IGNORECASE,
)


def _row_is_header(row) -> bool:
    """Dòng header kỳ: có marker header và KHÔNG chứa số tài chính (không phải data row)."""
    if _is_data_row(row):
        return False
    return any(_OCR_HEADER_MARKER_RE.search(c or "") for c in row)


def _compose_ocr_header(header_rows) -> list:
    """Gộp CỘT-THEO-CỘT các dòng header liên tiếp -> 1 header căn cột.

    Ví dụ ['Mã Thuyết','2025','2024'] + ['số minh','VND','VND'] -> ['Mã Thuyết số minh',
    '2025 VND', '2024 VND']. Bỏ ô là con số tài chính (an toàn kép)."""
    ncols = max((len(r) for r in header_rows), default=0)
    header = ["" for _ in range(ncols)]
    for r in header_rows:
        for c in range(min(len(r), ncols)):
            cell = (r[c] or "").strip()
            if cell and not _DATA_CELL_RE.search(cell):
                header[c] = (header[c] + " " + cell).strip()
    return header


# Section headings marking the company's GENERAL-INFO block (name/address/registration/
# ownership/business lines). A text chunk under one of these is tagged
# statement_kind="company_profile" so section-boost can lift it for profile questions
# (which otherwise lose to shareholder/related-party note tables).
_PROFILE_HEADING_KW = (
    "thông tin khái quát", "đặc điểm hoạt động", "hình thức sở hữu",
    "thông tin chung", "thông tin doanh nghiệp", "báo cáo của ban điều hành",
    "người đại diện", "ngành nghề kinh doanh",
)


def _ocr_text_statement_kind(block: str) -> str:
    """Classify an OCR text block by its ``##`` heading(s) -> "company_profile" or ""."""
    for line in block.split("\n"):
        if line.startswith("##") and any(kw in line.lower() for kw in _PROFILE_HEADING_KW):
            return "company_profile"
    return ""


def _parse_markdown_table(block: str) -> list:
    """Parse an OCR Markdown pipe table into ``list[list[cell]]`` (header first).

    Drops the ``|---|`` separator row. Returns ``[]`` if there are no pipe rows.
    """
    rows = []
    for line in block.split("\n"):
        if "|" not in line:
            continue
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        # Separator row: every cell is only dashes/colons/space.
        if cells and all(set(c) <= set("-: ") for c in cells):
            continue
        rows.append(cells)
    return rows


# Nhãn "yếu" của một dòng-số: rỗng, hoặc chỉ là CÔNG THỨC/mã tổng ("(100 = 110 + ...)",
# "100 =", "(200="). VAS in tên chỉ tiêu tổng ở dòng TRÊN, còn dòng-số mang công thức.
_FORMULA_LABEL_RE = re.compile(r"^\(?\s*\d{2,3}\s*=")
# Ô Mã số: 2–4 chữ số thuần (100, 110, 270, 01). KHÁC ô thuyết minh ("8(a)", "IO(a)").
_CODE_CELL_RE = re.compile(r"^\d{2,4}$")


def _is_ocr_label_only(row) -> bool:
    """Dòng chỉ-nhãn: cột 0 có chữ, KHÔNG có ô số tài chính (mã/thuyết minh thì kệ)."""
    return bool(row and (row[0] or "").strip()) and not _is_data_row(row)


def _find_header_col(header, *keys) -> int:
    """Chỉ số cột đầu tiên có header (đã fold) chứa 1 trong keys; -1 nếu không có."""
    for i, h in enumerate(header):
        hf = _fold(h)
        if any(k in hf for k in keys):
            return i
    return -1


def _ocr_row_documents(
    block, section_title, statement_kind, page_number, document_id, filename, table_index
):
    """One natural-language chunk per data row of an OCR table (mirrors camelot_rows).

    Each row carries its year/period HEADER inline (``31/12/2025 VND = ...; 1/1/2025 VND
    = ...``) so a fine-grained chunk still knows which column is current vs comparative.
    The statement title/section stay in metadata (not content) to avoid blurring the
    vector across every row of the same table. Returns ``[]`` if the block has no
    usable data rows (caller falls back to the atomic table chunk).

    MERGE nhãn dòng tổng: VAS tách tên chỉ tiêu tổng ("Tài sản ngắn hạn") ra dòng-chỉ-nhãn
    riêng, dòng-số kế mang nhãn = công thức "(100 = 110 + ...)". Ta giữ nhãn treo từ dòng-
    chỉ-nhãn liền trước và dùng nó khi dòng-số có nhãn YẾU -> chunk có tên chỉ tiêu đúng
    (nếu không, dòng tổng như [100]/[270] mất nhãn -> không truy hồi/đọc được).
    """
    rows = _parse_markdown_table(block)
    if len(rows) < 2:
        return []

    # Header kỳ = các dòng header THẬT liên tiếp ở đầu (không phải data). Nếu rows[0] là
    # data (block dữ liệu LCTT bị tách khỏi header) -> KHÔNG có header, lặp qua cả rows[0]
    # và dùng nhãn cột theo vị trí. TRÁNH lấy một dòng số làm header ("5.651... = value").
    n_header = 0
    while n_header < len(rows) and _row_is_header(rows[n_header]):
        n_header += 1
    header = _compose_ocr_header(rows[:n_header]) if n_header else []
    data_start = n_header if n_header else 0

    code_col = _find_header_col(header, "ma so", "code")
    note_col = _find_header_col(header, "thuyet minh", "note")
    docs = []
    row_index = 0
    pending_label = ""  # tên chỉ tiêu từ dòng-chỉ-nhãn liền trước (cho dòng tổng)
    for r in rows[data_start:]:
        if not _is_data_row(r):
            # Dòng-chỉ-nhãn -> nhớ làm nhãn treo (giữ dòng CỤ THỂ nhất = mới nhất).
            if _is_ocr_label_only(r):
                pending_label = (r[0] or "").strip()
            continue
        values = []
        label_cells = []
        code = ""
        for c, cell in enumerate(r):
            if not cell:
                continue
            if _DATA_CELL_RE.search(cell):
                col_hdr = header[c] if c < len(header) else ""
                # Header hợp lệ = có chữ và KHÔNG phải con số tài chính; nếu không -> nhãn
                # cột theo vị trí + loại báo cáo (năm nay/năm trước cho LCTT/KQKD).
                if col_hdr and not _DATA_CELL_RE.search(col_hdr):
                    hdr = col_hdr
                else:
                    hdr = _positional_period(len(values), statement_kind)
                values.append((hdr, cell))
            elif c == code_col and _CODE_CELL_RE.match(cell):
                code = cell
            elif c == note_col:
                continue  # bỏ mã thuyết minh khỏi nhãn (nhiễu)
            else:
                label_cells.append(cell)
        if not values:
            continue
        own_label = " ".join(label_cells).strip()
        # Nhãn yếu (rỗng / công thức tổng) + có nhãn treo -> lấy tên chỉ tiêu từ nhãn treo.
        if pending_label and (not own_label or _FORMULA_LABEL_RE.search(own_label)):
            label = f"{pending_label} {own_label}".strip()
        else:
            label = own_label or pending_label or section_title or "dòng"
        pending_label = ""  # tiêu thụ / hết hạn sau mỗi dòng-số

        code_str = f"[Mã số {code}] " if code else ""
        pairs = "; ".join(f"{hdr} = {val}" for hdr, val in values)
        content = f"{code_str}{label} (trang {page_number}): {pairs}."
        docs.append(
            Document(
                page_content=content,
                metadata={
                    "document_id": document_id,
                    "filename": filename,
                    "page": page_number,
                    "content_type": "table",
                    "source": "ocr_rows",
                    "statement": section_title,
                    "statement_kind": statement_kind,
                    "table_index": table_index,
                    "row_index": row_index,
                    "row_label": label,
                    "code": code,
                },
            )
        )
        row_index += 1
    return docs


def _extract_ocr_page(file_path, page_number, document_id, filename):
    """OCR a scanned page via the vision model and split into table/text chunks.

    Returns ``(table_documents, text_documents)``.
    """
    table_documents = []
    text_documents = []

    markdown = ocr_page_to_markdown(file_path, page_number)

    # Section context (title + date/unit) carried from the most recent text block
    # onto the following table chunk. OCR runs per page, so this resets each page —
    # fine because DHG repeats the "(Tiếp theo)" heading on every continuation page.
    current_section = ""
    # statement_kind của trang, suy từ mã biểu mẫu trong text block (độc lập với việc tiêu
    # đề có phải heading ## hay không — vá ca SAB tr.10 KQKD chỉ có dòng text thường).
    current_section_kind = ""

    table_index = 0
    for kind, block in split_markdown_blocks(markdown):
        if kind == "table":
            ti = table_index
            table_index += 1
            title = _section_title_only(current_section)
            # Ưu tiên kind từ mã biểu mẫu (mạnh, độc lập layout); fallback tiêu đề ##.
            table_kind = current_section_kind or _ocr_statement_kind(title)

            # Per-row chunking (OCR_TABLE_MODE="rows"): fine-grained, like camelot_rows.
            if OCR_TABLE_MODE == "rows":
                row_docs = _ocr_row_documents(
                    block, title, table_kind,
                    page_number, document_id, filename, ti,
                )
                if row_docs:
                    table_documents.extend(row_docs)
                    continue  # rows emitted; skip atomic fallback

            # Atomic (default): 1 chunk per whole table, optional section prepend.
            content = block
            if OCR_PREPEND_SECTION_TITLE and current_section:
                content = f"{current_section}\n\n{block}"
            table_documents.append(
                Document(
                    page_content=content,
                    metadata={
                        "document_id": document_id,
                        "filename": filename,
                        "page": page_number,
                        "content_type": "table",
                        "table_index": ti,
                        "source": "ocr",
                    },
                )
            )
        else:
            if OCR_PREPEND_SECTION_TITLE:
                section = _ocr_section_context(block)
                if section:
                    current_section = section
            # Bắt mã biểu mẫu (Mẫu B 0X) ở bất kỳ đâu trong block để gán kind cho các
            # bảng phía sau, kể cả khi tiêu đề không được xuất thành heading ##.
            blk_kind = _ocr_block_statement_kind(block)
            if blk_kind:
                current_section_kind = blk_kind
            text_documents.append(
                Document(
                    page_content=block,
                    metadata={
                        "document_id": document_id,
                        "filename": filename,
                        "page": page_number,
                        "content_type": "text",
                        "source": "ocr",
                        # Tag general-info narrative so section-boost can lift it for
                        # company-profile questions ("" for ordinary narrative/notes).
                        "statement_kind": _ocr_text_statement_kind(block),
                    },
                )
            )

    return table_documents, text_documents


# Phát hiện ĐƠN VỊ TIỀN + SCALE ở TẦNG TÀI LIỆU (một lần khi ingest) rồi gắn vào metadata mọi chunk.
# Chạy trên RAW text (native page.extract_text() + OCR markdown) — nơi "$"/"USD"/"in millions"/"VND"
# CÒN NGUYÊN, khác với chunk bảng đã bị strip "$"/scale. Nhờ vậy USD có tín hiệu DƯƠNG (không suy đoán
# "vắng VND = USD"), tách VND/USD sạch, doc VN không bị gắn nhầm "million".
_CUR_VND_RE = re.compile(r"\bVND\b|Đơn\s*vị|Mẫu\s*số\s*B\s*0\d|B0\d-DN|đồng", re.IGNORECASE)
_CUR_USD_RE = re.compile(r"\bUSD\b|US\$|\bin\s+(?:millions?|thousands?)\b", re.IGNORECASE)
_SCALE_MILLION_RE = re.compile(r"in\s+millions?|\(in\s+millions?\)|triệu\s*đồng", re.IGNORECASE)
_SCALE_THOUSAND_RE = re.compile(r"in\s+thousands?|\(in\s+thousands?\)|ngh[ìi]n\s*đồng|ngàn\s*đồng",
                                re.IGNORECASE)


def _detect_currency_scale(text: str) -> tuple[str, str]:
    """(currency, unit_scale) suy từ RAW text tài liệu. currency ∈ {"VND","USD"}; unit_scale ∈
    {"million","thousand",""}. VN-first: chỉ trả "USD" khi tín hiệu USD ÁP ĐẢO tín hiệu VND (doc VN
    có thể nhắc "$"/"USD" lẻ trong thuyết minh -> vẫn phải là VND). Scale chỉ set khi được NÊU RÕ."""
    t = text or ""
    vnd = len(_CUR_VND_RE.findall(t))
    usd = len(_CUR_USD_RE.findall(t)) + t.count("$")
    currency = "USD" if usd > vnd else "VND"
    if _SCALE_MILLION_RE.search(t):
        scale = "million"
    elif _SCALE_THOUSAND_RE.search(t):
        scale = "thousand"
    else:
        scale = ""
    # VN VAS ghi ĐỦ đồng (tuyệt đối) trừ khi nói "triệu/nghìn đồng" -> nếu currency=VND mà scale suy
    # từ "in millions" (tiếng Anh, không áp cho VN) thì bỏ, tránh gắn "million" nhầm cho doc VN.
    if currency == "VND" and scale and not re.search(r"triệu\s*đồng|ngh[ìi]n\s*đồng|ngàn\s*đồng",
                                                     t, re.IGNORECASE):
        scale = ""
    return currency, scale


def ingest_pdf(file_path: str, document_id: str = None):

    document_id = document_id or str(uuid.uuid4())
    filename = os.path.basename(file_path)

    text_documents = []
    table_documents = []
    raw_texts = []  # RAW text mỗi trang -> corpus phát hiện currency/scale (giữ "$"/"in millions")
    num_pages = 0
    num_ocr_pages = 0
    num_text_pages = 0

    # Chọn trình trích bảng cho trang native. Camelot: bảng lấy từ Camelot (chạy
    # 1 lần cho cả file, gom theo trang), narrative vẫn từ pdfplumber.
    #   "camelot"      -> 1 chunk/bảng; "camelot_rows" -> 1 chunk nhỏ mỗi dòng.
    camelot_by_page = {}
    if TABLE_EXTRACTOR in ("camelot", "camelot_rows"):
        from app.ingestion.camelot_extract import (
            extract_all_row_chunks,
            extract_all_tables,
        )

        if TABLE_EXTRACTOR == "camelot_rows":
            camelot_by_page = extract_all_row_chunks(file_path, document_id, filename)
        else:
            camelot_by_page = extract_all_tables(file_path, document_id, filename)

    with pdfplumber.open(file_path) as pdf:
        num_pages = len(pdf.pages)
        boilerplate = _boilerplate_lines(pdf.pages)

        for page in pdf.pages:
            page_number = page.page_number  # 1-indexed
            raw_texts.append(page.extract_text() or "")  # cho phát hiện currency (USD "$" ở đây)

            if _is_scanned_page(page, boilerplate):
                tables, texts = _extract_ocr_page(
                    file_path, page_number, document_id, filename
                )
                num_ocr_pages += 1
            elif TABLE_EXTRACTOR in ("camelot", "camelot_rows"):
                # Bảng: Camelot; narrative: pdfplumber (bỏ nhánh table của nó).
                _, texts = _extract_text_page(
                    page, document_id, filename, page_number
                )
                tables = camelot_by_page.get(page_number, [])
                num_text_pages += 1
            else:
                tables, texts = _extract_text_page(
                    page, document_id, filename, page_number
                )
                num_text_pages += 1

            table_documents.extend(tables)
            text_documents.extend(texts)

    # Split only the narrative text; tables stay atomic.
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
    )

    text_chunks = splitter.split_documents(text_documents)

    all_chunks = text_chunks + table_documents

    # Phát hiện currency/scale MỘT LẦN cho cả tài liệu rồi gắn vào metadata MỌI chunk. Corpus =
    # raw text native (giữ "$"/"in millions" cho Apple/Tesla) + nội dung chunk (giữ "VND" cho doc scan).
    detect_src = "\n".join(raw_texts) + "\n" + "\n".join(d.page_content for d in all_chunks)
    currency, unit_scale = _detect_currency_scale(detect_src)
    for d in all_chunks:
        d.metadata["currency"] = currency
        d.metadata["unit_scale"] = unit_scale

    vector_store = get_vector_store()
    vector_store.add_documents(all_chunks)

    result = {
        "document_id": document_id,
        "filename": filename,
        "num_pages": num_pages,
        "num_text_pages": num_text_pages,
        "num_ocr_pages": num_ocr_pages,
        "num_tables": len(table_documents),
        "num_chunks": len(all_chunks),
        "currency": currency,
        "unit_scale": unit_scale,
    }

    save_last_document(result)

    return result
