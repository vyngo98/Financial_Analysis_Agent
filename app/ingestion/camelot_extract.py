"""Trích bảng cho trang native bằng Camelot (stream, đã tinh chỉnh) — dùng khi
TABLE_EXTRACTOR="camelot". Trang scan vẫn đi OCR (không qua đây).

Camelot stream với edge_tol/row_tol cao cho: cột Code/Note tách riêng, header năm
căn đúng cột, full recall (không cắt cụt). Ta chạy MỘT lần `pages="all"` rồi gom
theo trang (nhanh hơn gọi từng trang).

Import `table_to_markdown`, `_is_data_row` từ pdf_ingestion — nạp lười (lazy) từ
phía ingest_pdf để tránh vòng import.
"""

import re
from collections import defaultdict

import camelot
from langchain_core.documents import Document

from app.config import (
    CAMELOT_EDGE_TOL,
    CAMELOT_ROW_TOL,
    CAMELOT_ROW_TOL_FACTOR,
    CAMELOT_ROW_TOL_MAX,
    CAMELOT_ROW_TOL_MIN,
    CAMELOT_ROW_TOL_MODE,
    TABLE_ROW_FORMAT,
)
from app.ingestion.pdf_ingestion import (
    _DATA_CELL_RE,
    _HEADER_LINE_RE,
    _is_data_row,
    table_to_markdown,
)


def _clean_rows(df):
    """DataFrame Camelot -> list[list[str]], gộp khoảng trắng/xuống dòng trong ô."""
    rows = []
    for row in df.values.tolist():
        rows.append([" ".join(str(c).split()) for c in row])
    return rows


# --------------------------------------------------------------------------- #
# Đọc bảng camelot — hỗ trợ row_tol thích ứng theo LINE PITCH từng trang
# --------------------------------------------------------------------------- #

def _adaptive_row_tol(pitch):
    """pitch (pt) -> row_tol = round(FACTOR*pitch), clamp [MIN, MAX]. Không pitch -> mặc định."""
    if not pitch:
        return CAMELOT_ROW_TOL
    return max(CAMELOT_ROW_TOL_MIN, min(CAMELOT_ROW_TOL_MAX, round(CAMELOT_ROW_TOL_FACTOR * pitch)))


def _page_row_tols(pdf_path):
    """dict[page_number] -> row_tol thích ứng, ước từ median khoảng cách dòng (pdfplumber).

    Chỉ tính khoảng cách 0<gap<40pt (bỏ nhảy đoạn/blank) để median = line pitch thật.
    Trang không đủ dòng -> fallback CAMELOT_ROW_TOL.
    """
    import pdfplumber

    tols = {}
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            tops = sorted(ln["top"] for ln in page.extract_text_lines())
            gaps = sorted(b - a for a, b in zip(tops, tops[1:]) if 0 < b - a < 40)
            pitch = gaps[len(gaps) // 2] if gaps else None
            tols[page.page_number] = _adaptive_row_tol(pitch)
    return tols


def _read_tables(pdf_path):
    """Trả list[camelot.Table] cho cả PDF, tôn trọng CAMELOT_ROW_TOL_MODE.

    fixed    : 1 lần read_pdf(pages="all", row_tol=CAMELOT_ROW_TOL).
    adaptive : gom trang theo bucket row_tol (thường chỉ 2-5 giá trị) rồi read_pdf
               một lần cho mỗi bucket -> vài lần gọi, không phải mỗi trang.
    """
    if CAMELOT_ROW_TOL_MODE != "adaptive":
        return list(camelot.read_pdf(
            pdf_path, pages="all", flavor="stream",
            edge_tol=CAMELOT_EDGE_TOL, row_tol=CAMELOT_ROW_TOL,
        ))

    tols = _page_row_tols(pdf_path)
    buckets = defaultdict(list)
    for page_no, tol in tols.items():
        buckets[tol].append(page_no)

    tables = []
    for tol, pages in buckets.items():
        page_str = ",".join(str(p) for p in sorted(pages))
        tables.extend(camelot.read_pdf(
            pdf_path, pages=page_str, flavor="stream",
            edge_tol=CAMELOT_EDGE_TOL, row_tol=tol,
        ))
    return tables


def extract_all_tables(pdf_path, document_id, filename):
    """Chạy Camelot 1 lần cho cả PDF -> dict[page_number] -> list[Document].

    Chỉ giữ bảng còn dữ liệu tài chính (bỏ khối title mà Camelot nhận nhầm).
    Trang không có bảng (hoặc scan) đơn giản là không có key.
    """
    by_page = {}
    try:
        tables = _read_tables(pdf_path)
    except Exception:
        return by_page

    for t in tables:
        page = t.page
        rows = _clean_rows(t.df)
        if not any(_is_data_row(r) for r in rows):
            continue
        markdown = table_to_markdown(rows)
        if not markdown:
            continue
        idx = len(by_page.get(page, []))
        by_page.setdefault(page, []).append(
            Document(
                page_content=markdown,
                metadata={
                    "document_id": document_id,
                    "filename": filename,
                    "page": page,
                    "content_type": "table",
                    "table_index": idx,
                    "source": "camelot",
                },
            )
        )
    return by_page


# --------------------------------------------------------------------------- #
# camelot_rows — MỘT chunk nhỏ mỗi dòng dữ liệu + header năm (v3)
# --------------------------------------------------------------------------- #

def _is_label_only(row):
    """Chỉ cột 0 có chữ, không có con số tài chính, các cột khác rỗng."""
    return bool(row and row[0]) and not _is_data_row(row) and not any(c for c in row[1:])


def _looks_like_continuation(next_row):
    """Dòng label-only ngay sau data row có phải phần TIẾP của nhãn không?

    Mặc định an toàn: chỉ coi là continuation khi chữ đầu VIẾT THƯỜNG (vd
    "trading securities", "provision of services"). Label-only viết HOA -> coi là
    SECTION mới, không bao giờ dính vào nhãn dòng dữ liệu.
    """
    lbl = next_row[0] if next_row else ""
    return bool(lbl) and lbl[:1].islower()


def _recover_header(rows, first_data):
    """Gộp CỘT-THEO-CỘT các dòng header-like phía trên dòng data đầu tiên.

    Trả list căn cột, vd ["", "Code", "Note", "31/12/2023 VND", "1/1/2023 VND"].
    `_HEADER_LINE_RE` chọn đúng dòng header (năm/VND/Code/Note) và bỏ dòng title rác.
    """
    ncols = max((len(r) for r in rows), default=0)
    header = ["" for _ in range(ncols)]
    for r in rows[:first_data]:
        if not _HEADER_LINE_RE.search(" ".join(r)):
            continue
        for c in range(min(len(r), ncols)):
            cell = r[c]
            # Ô header thật rất ngắn (năm, VND, Code, Note, dd/mm/yyyy). Bỏ ô dài
            # là mảnh title lọt vào (vd "dated 22 December 2014 of the Ministry...").
            if cell and len(cell) <= 24:
                header[c] = (header[c] + " " + cell).strip()

    # B2: bảng SEGMENT có period LẶP (vd "2023 VND" xuất hiện >1 lần vì các cột
    # Domestic|Overseas|Total dùng chung header năm). Gắn nhãn nhóm + tag "total"
    # cho nhóm cuối để LLM không nhầm số segment đầu (nội địa) là số tổng.
    vals = [h for h in header if h]
    if len(vals) != len(set(vals)):
        header = _compose_segment_header(header)
    return header


def _compose_segment_header(header):
    """Gắn nhãn nhóm cho bảng segment (header năm lặp). Nhóm cuối = 'total'.

    ["", "2023 VND","2022 VND","2023 VND","2022 VND","2023 VND","2022 VND"]
      -> ["", "segment1 2023 VND", ..., "total 2023 VND", "total 2022 VND"]
    """
    val_cols = [c for c, h in enumerate(header) if h]
    if not val_cols:
        return header
    # Kích thước nhóm = số period phân biệt trước khi lặp lại (vd 2: 2023,2022).
    uniq = []
    for c in val_cols:
        if header[c] in uniq:
            break
        uniq.append(header[c])
    k = len(uniq) or 1
    ngroups = len(val_cols) // k
    if ngroups < 2:
        return header
    new = list(header)
    for gi in range(ngroups):
        tag = "total" if gi == ngroups - 1 else f"segment{gi + 1}"
        for j in range(k):
            idx = gi * k + j
            if idx < len(val_cols):
                c = val_cols[idx]
                new[c] = f"{tag} {header[c]}"
    return new


def _positional(col_value_index):
    """Nhãn cột dự phòng khi không khôi phục được header."""
    return {0: "current year", 1: "comparative"}.get(col_value_index, "value")


# Token số (kèm ngoặc âm) để tách khỏi label khi camelot gộp cột số vào cột nhãn.
_NUMTOK_RE = re.compile(r"\(?-?\d[\d.,]{4,}\)?")


def _split_label_numbers(cell):
    """Tách ô nhãn bị lẫn số -> (nhãn chữ sạch, [số]).

    Vd 'Short-term investments 1,193,065,962 ... (689,745,197) in shares'
       -> ('Short-term investments in shares',
           ['1,193,065,962', ..., '(689,745,197)']).
    """
    nums = _NUMTOK_RE.findall(cell)
    label = " ".join(_NUMTOK_RE.sub(" ", cell).split())
    return label, nums


def _make_data_record(row, header, section):
    """Tách 1 data row thành record {label, code, note, section, values}."""
    label = row[0] if row else ""
    code = row[1] if len(row) > 1 and row[1] and not _DATA_CELL_RE.search(row[1]) else ""
    note = row[2] if len(row) > 2 and row[2] and not _DATA_CELL_RE.search(row[2]) else ""

    # B1: camelot đôi khi gộp cột số VÀO ô nhãn (row[0]) -> nhãn "rác" kèm nhiều số.
    # Nếu ô nhãn vừa có CHỮ vừa có số -> tách số ra values, giữ phần chữ làm nhãn.
    label_nums = []
    if label and _DATA_CELL_RE.search(label) and any(ch.isalpha() for ch in label):
        label, label_nums = _split_label_numbers(label)

    values = []  # [(header_cột, value)] theo thứ tự cột
    vi = 0
    for c in range(len(row)):
        if c == 0 and label_nums:
            # Số đã nằm trong ô nhãn -> phát ra làm values (thay vì cả chuỗi rác).
            for nv in label_nums:
                hdr = header[c] if c < len(header) and header[c] else _positional(vi)
                values.append((hdr, nv))
                vi += 1
            continue
        if _DATA_CELL_RE.search(row[c] or ""):
            hdr = header[c] if c < len(header) and header[c] else _positional(vi)
            values.append((hdr, row[c]))
            vi += 1

    if not label:
        label = code and f"[{code}]" or section
    return {"label": label, "code": code, "note": note, "section": section, "values": values}


def _format_row_nl(rec, page):
    """Render record thành câu tự nhiên GỌN: nhãn + code/note + page + cặp năm=giá trị.

    KHÔNG chèn statement_title/section (chúng lặp trên mọi dòng cùng nhóm -> nhoè
    vector). Ngữ cảnh đó nằm ở metadata. Dòng subtotal vẫn giữ tên nhóm vì đã được
    gộp vào chính `label` (pre-continuation merge)."""
    if not rec["values"]:
        return ""
    head = rec["label"]
    tags = []
    if rec["code"]:
        tags.append(f"Code {rec['code']}")
    if rec["note"]:
        tags.append(f"Note {rec['note']}")
    if tags:
        head += " [" + ", ".join(tags) + "]"

    pairs = "; ".join(f"{hdr} = {val}" for hdr, val in rec["values"])
    return f"{head} (page {page}): {pairs}."


def _format_row_md(rec, header, page, statement_title):
    """Fallback: bảng markdown mini (header năm + đúng 1 dòng dữ liệu)."""
    ncols = len(header)
    data_row = [""] * ncols
    if ncols:
        data_row[0] = rec["label"]
    # đặt lại code/note/values theo header đã căn
    body = [header, data_row] if any(header) else [data_row]
    # dựng lại dòng dữ liệu đầy đủ từ record: dùng chính giá trị gốc
    # (đơn giản: ghép label + code + note + values)
    flat = [rec["label"], rec["code"], rec["note"]] + [v for _, v in rec["values"]]
    caption = ", ".join(x for x in (statement_title, rec["section"], f"page {page}") if x)
    md = table_to_markdown([header, flat]) if any(header) else table_to_markdown([flat])
    return f"## {caption}\n{md}" if caption else md


def _statement_title(rows):
    """Đoán tên báo cáo từ dòng header-like dài nhất (vd 'CONSOLIDATED STATEMENT...')."""
    best = ""
    for r in rows:
        line = " ".join(c for c in r if c).strip()
        if "STATEMENT" in line.upper() and len(line) > len(best):
            best = line
    # Cắt tiền tố "Form B .." nếu có, giữ từ "CONSOLIDATED"/"STATEMENT" trở đi.
    up = best.upper()
    for anchor in ("CONSOLIDATED", "STATEMENT"):
        pos = up.find(anchor)
        if pos != -1:
            return best[pos:].strip()
    return best.strip()


def _is_prose_label(label: str) -> bool:
    """Nhãn 'giống CÂU VĂN' — nhiều từ (>=8), phân biệt văn xuôi với line-item.

    Dùng ĐỂ loại dòng mà camelot nhận nhầm VĂN XUÔI (báo cáo kiểm toán, MD&A,
    kiện tụng) thành 'dòng bảng' — nhưng CHỈ áp dụng bên trong bảng 1 CỘT (xem
    guard trong extract_all_row_chunks). Line-item/glued-data thật có nhãn NGẮN
    (vd "Cash on hand", "Short-term investments in shares", "Miraka Holdings
    Limited") nên không dính ngưỡng 8 từ; văn xuôi ("We have served as the
    Company's auditor since", "As at 31 December 2023, the Group had ... employees")
    thì dính.
    """
    return len(label.split()) >= 8


# Dòng-tổng ĐẶC TRƯNG để nhận diện báo cáo tài chính HỢP NHẤT chính (v4.6 section boost).
# Kiểm theo THỨ TỰ: cash_flow trước (cụm "... activities" rất đặc thù, dù CF cũng có dòng
# "Net income"/"Profit before tax"), rồi income_statement, rồi balance_sheet.
_STMT_KIND_SIGNATURES = (
    ("cash_flow", ("operating activities", "investing activities", "financing activities")),
    ("income_statement", ("total revenues", "net revenue", "revenue from sales",
                          "income from operations", "net income", "net profit",
                          "profit before tax", "gross profit")),
    ("balance_sheet", ("total assets", "total liabilities and equity",
                       "total liabilities and owners", "total resources",
                       "total equity", "owners")),
)


def _statement_kind(labels_text: str) -> str:
    """Phân loại loại BCTC của một TRANG từ text gộp mọi nhãn dòng (case-insensitive).

    Dùng để boost khi rerank: dòng bảng chính (kind != "") được ưu tiên hơn văn xuôi
    MD&A/thuyết minh (kind = "") khi câu hỏi cùng loại. Trang không phải BCTC chính -> "".
    """
    s = (labels_text or "").lower()
    for kind, sigs in _STMT_KIND_SIGNATURES:
        if any(sig in s for sig in sigs):
            return kind
    return ""


def extract_all_row_chunks(pdf_path, document_id, filename):
    """Camelot -> dict[page] -> list[Document], MỘT Document mỗi DÒNG DỮ LIỆU.

    Mỗi chunk nhỏ (retrieval mịn) mang sẵn header năm căn cột (sửa đọc-lệch-cột).
    """
    by_page = {}
    try:
        tables = _read_tables(pdf_path)
    except Exception:
        return by_page

    # Pass 1: gom tên báo cáo theo trang (từ các bảng title không có data).
    title_by_page = {}
    for t in tables:
        rows = _clean_rows(t.df)
        if any(_is_data_row(r) for r in rows):
            continue
        title = _statement_title(rows)
        if title and t.page not in title_by_page:
            title_by_page[t.page] = title

    # Pass 1b: phân loại statement_kind theo TRANG (union nhãn mọi bảng cùng trang) để
    # chịu được statement tách nhiều bảng (Tesla CF p54=2 bảng, VNM BS p7-8 mỗi trang 1 phía).
    labels_by_page: dict = {}
    for t in tables:
        rows = _clean_rows(t.df)
        joined = " ".join(r[0] for r in rows if r and r[0])
        labels_by_page[t.page] = labels_by_page.get(t.page, "") + " " + joined
    kind_by_page = {pg: _statement_kind(txt) for pg, txt in labels_by_page.items()}

    # Pass 2: mỗi bảng có data -> emit từng dòng.
    for t in tables:
        page = t.page
        rows = _clean_rows(t.df)
        data_idxs = [i for i, r in enumerate(rows) if _is_data_row(r)]
        if not data_idxs:
            continue
        first_data = data_idxs[0]
        header = _recover_header(rows, first_data)
        statement_title = title_by_page.get(page, "")
        # Bảng 1 CỘT = camelot không tìm được cấu trúc cột -> hoặc là VĂN XUÔI (báo
        # cáo kiểm toán, MD&A) bị nhận nhầm, hoặc là bảng thật bị gộp cột (B1 cứu).
        # Chỉ ở bảng 1 cột mới bật prose-guard (bảng đa cột luôn giữ trọn để không
        # bỏ nhầm line-item nhãn dài như "Common stock; $ par value; ...").
        single_col = t.df.shape[1] <= 1

        records = []
        section = ""
        pending = []  # các dòng label-only liên tiếp (heading) đang chờ
        i = 0
        n = len(rows)
        while i < n:
            r = rows[i]
            if _is_data_row(r):
                # PRE-continuation: dòng subtotal có nhãn CHỈ là công thức/rỗng và ngay
                # trên là heading -> heading gần nhất chính là TÊN của dòng này.
                own = r[0] or ""
                label_is_formula = not any(ch.isalpha() for ch in own)
                rec_section = pending[-1] if pending else section
                rec = _make_data_record(r, header, rec_section)
                if label_is_formula and pending:
                    rec["label"] = (pending[-1] + " " + rec["label"]).strip()
                if pending:
                    section = pending[-1]  # heading trở thành section cho các dòng sau
                    pending = []
                # POST-continuation: label-only chữ thường ngay sau -> nối đuôi nhãn.
                j = i + 1
                while j < n and _is_label_only(rows[j]) and _looks_like_continuation(rows[j]):
                    rec["label"] = (rec["label"] + " " + rows[j][0]).strip()
                    j += 1
                # PROSE GUARD: trong bảng 1 CỘT, bỏ dòng có nhãn giống CÂU VĂN (>=8 từ)
                # — đó là văn xuôi bị camelot nhận nhầm thành 'dòng bảng' (nội dung đã có
                # ở native text chunk). Glued-data thật trong bảng 1 cột có nhãn ngắn nên
                # được giữ. Bảng đa cột không bao giờ chạm guard này.
                if single_col and _is_prose_label(rec["label"]):
                    i = j
                    continue
                records.append(rec)
                i = j
                continue
            if _is_label_only(r):
                pending.append(r[0])
            i += 1

        # Tất cả dòng của một bảng chia sẻ cùng table_index (snapshot trước khi append).
        table_index = sum(1 for d in by_page.get(page, []) if d.metadata["row_index"] == 0)
        for k, rec in enumerate(records):
            if TABLE_ROW_FORMAT == "markdown":
                content = _format_row_md(rec, header, page, statement_title)
            else:
                content = _format_row_nl(rec, page)
            if not content:
                continue
            by_page.setdefault(page, []).append(
                Document(
                    page_content=content,
                    metadata={
                        "document_id": document_id,
                        "filename": filename,
                        "page": page,
                        "content_type": "table",
                        "source": "camelot_rows",
                        "statement": statement_title,
                        "table_index": table_index,
                        "row_index": k,
                        "row_label": rec["label"],
                        "code": rec["code"],
                        "note": rec["note"],
                        "section": rec["section"],
                        "statement_kind": kind_by_page.get(page, ""),
                    },
                )
            )
    return by_page
