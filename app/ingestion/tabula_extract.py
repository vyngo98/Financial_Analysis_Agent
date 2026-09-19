"""Trích bảng bằng tabula (Java) rồi GỘP các mảnh bị tabula tách nhầm dựa trên
vị trí bbox, và căn lại cột để header năm về đúng cột giá trị.

Vấn đề gốc: tabula stream đôi khi tách MỘT bảng logic thành nhiều bảng — điển
hình là dải header năm (`Code Note 2023 VND 2022VND`) bị tách khỏi các dòng dữ
liệu (báo cáo KQKD trang 10 của Vinamilk: header bbox chồng lên data, gap âm).
Bảng dữ liệu mất header cột năm -> LLM đọc lệch cột.

`extract_page_logical_tables()` đọc tabula ở `output_format="json"` (có toạ độ
per-cell), cluster các bảng theo bbox proximity, gộp cụm nhiều mảnh thành một
bảng và căn cột lại bằng span-overlap.

Tái sử dụng helper của pdfplumber path (không viết lại): `table_to_markdown`,
`_HEADER_LINE_RE`, `_DATA_CELL_RE`, `_is_data_row`, `_clean_cell`.
"""

import tabula

from app.ingestion.pdf_ingestion import (
    _clean_cell,
    _DATA_CELL_RE,
    _HEADER_LINE_RE,
    _is_data_row,
)

# Hai bảng gộp khi khe dọc <= GAP_MAX (âm = chồng lấn) VÀ overlap ngang đủ lớn.
# 12pt gộp được dải header chồng data (page 10, gap -29) và các mảnh cách ~10pt
# (ma trận biến động vốn trang 53) nhưng KHÔNG gộp bảng cách >=17pt (trang 54).
GAP_MAX = 12.0
HORIZ_MIN = 0.5


def _norm_tables(raw_tables):
    """tabula JSON -> list bản ghi nội bộ có bbox + cells (mỗi cell {l,r,cx,text})."""
    records = []
    for t in raw_tables:
        rows = []
        for row in t.get("data") or []:
            cells = []
            for c in row:
                left = c.get("left", 0.0)
                width = c.get("width", 0.0)
                cells.append(
                    {
                        "l": left,
                        "r": left + width,
                        "cx": left + width / 2.0,
                        "top": c.get("top", 0.0),
                        "text": _clean_cell(c.get("text", "")),
                    }
                )
            rows.append(cells)
        # bỏ bảng rỗng toàn phần
        if not any(cell["text"] for row in rows for cell in row):
            continue
        top = t.get("top", 0.0)
        left = t.get("left", 0.0)
        right = t.get("right", left + t.get("width", 0.0))
        bottom = t.get("bottom", top + t.get("height", 0.0))
        records.append(
            {"top": top, "left": left, "right": right, "bottom": bottom, "rows": rows}
        )
    return records


def _same_cluster(a_left, a_right, a_bottom, b):
    """Bảng b (dưới) có thuộc cùng cụm với vùng đang gom (a_*) không?"""
    gap = b["top"] - a_bottom
    if gap > GAP_MAX:
        return False
    overlap = min(a_right, b["right"]) - max(a_left, b["left"])
    if overlap <= 0:
        return False
    narrower = min(a_right - a_left, b["right"] - b["left"])
    if narrower <= 0:
        return False
    return overlap >= HORIZ_MIN * narrower


def _cluster(records):
    """Sort theo top rồi greedy grow thành các cụm gần nhau."""
    records = sorted(records, key=lambda r: r["top"])
    clusters = []
    for rec in records:
        if not clusters:
            clusters.append([rec])
            continue
        cur = clusters[-1]
        c_left = min(r["left"] for r in cur)
        c_right = max(r["right"] for r in cur)
        c_bottom = max(r["bottom"] for r in cur)
        if _same_cluster(c_left, c_right, c_bottom, rec):
            cur.append(rec)
        else:
            clusters.append([rec])
    return clusters


def _rows_as_text(record):
    """Một bảng đơn -> list[list[str]] nguyên trạng (pass-through)."""
    return [[cell["text"] for cell in row] for row in record["rows"]]


def _vrange_overlap(a, b):
    """Độ chồng lấn theo trục dọc (y) của 2 bản ghi; <=0 nghĩa là rời nhau."""
    return min(a["bottom"], b["bottom"]) - max(a["top"], b["top"])


def _recover_header_texts(rows):
    """Gộp các dòng header (không có số, có token header) thành 1 dòng lên đầu."""
    header_idx = [
        i
        for i, row in enumerate(rows)
        if any(_HEADER_LINE_RE.search(c) for c in row)
        and not any(_DATA_CELL_RE.search(c) for c in row)
    ]
    if not header_idx:
        return rows
    ncols = max((len(r) for r in rows), default=0)
    merged = [""] * ncols
    for i in header_idx:
        for j, c in enumerate(rows[i]):
            if c:
                merged[j] = (merged[j] + " " + c).strip() if merged[j] else c
    rest = [row for i, row in enumerate(rows) if i not in set(header_idx)]
    return [merged] + rest


def _median(xs):
    xs = sorted(xs)
    n = len(xs)
    return xs[n // 2] if n % 2 else (xs[n // 2 - 1] + xs[n // 2]) / 2.0


def _col_centers(record):
    """Tâm x đại diện (median center) của mỗi cột trong một bảng.

    Dùng median center thay vì khung min-left/max-right vì cột nhãn rất rộng (nhãn
    dài chồm sang phải) sẽ overlap mọi thứ; tâm cột thì tách bạch rõ (nhãn ở trái,
    cột giá trị ở phải)."""
    ncols = max((len(row) for row in record["rows"]), default=0)
    centers = []
    for j in range(ncols):
        cs = [row[j]["cx"] for row in record["rows"] if j < len(row) and row[j]["text"]]
        centers.append(_median(cs) if cs else None)
    return centers


def _nearest_col(cell, centers):
    """Chỉ số cột có tâm gần cell nhất theo trục x (None nếu không có cột nào)."""
    best_i, best_d = None, None
    for i, c in enumerate(centers):
        if c is None:
            continue
        d = abs(cell["cx"] - c)
        if best_d is None or d < best_d:
            best_d, best_i = d, i
    return best_i


def _merge_continuation(cluster):
    """Case A — các mảnh RỜI nhau theo dọc (vd trang 53): nối dòng theo thứ tự
    trên->dưới, cùng cấu trúc cột. Rồi khôi phục header."""
    ordered = sorted(cluster, key=lambda r: r["top"])
    rows = [[c["text"] for c in row] for rec in ordered for row in rec["rows"]]
    return _recover_header_texts(rows)


def _merge_header_band(cluster):
    """Case B — có mảnh CHỒNG dọc (vd trang 10): mảnh lớn nhất là bảng dữ liệu
    (giữ nguyên cột do tabula tách, đã sạch); các mảnh chồng còn lại chỉ dùng để
    trích nhãn header, căn theo x về đúng cột giá trị rồi prepend."""
    base = max(cluster, key=lambda r: len(r["rows"]))
    centers = _col_centers(base)
    ncols = len(centers)

    # Thu nhãn header từ các mảnh khác (dòng header: có token header, không có số).
    header_cells = []
    for rec in cluster:
        if rec is base:
            continue
        for row in rec["rows"]:
            texts = [c["text"] for c in row]
            if not any(t for t in texts):
                continue
            if _is_data_row(texts):
                continue
            if any(_HEADER_LINE_RE.search(t) for t in texts):
                header_cells.extend(c for c in row if c["text"])

    header_row = [""] * ncols
    for c in header_cells:
        i = _nearest_col(c, centers)
        if i is not None:
            header_row[i] = (header_row[i] + " " + c["text"]).strip() if header_row[i] else c["text"]

    data_rows = [[c["text"] for c in row] for row in base["rows"]]
    if any(header_row):
        # Nếu base đã tự có dòng header thì không thêm nữa (tránh trùng).
        first = data_rows[0] if data_rows else []
        base_has_header = (
            first
            and any(_HEADER_LINE_RE.search(c) for c in first)
            and not any(_DATA_CELL_RE.search(c) for c in first)
        )
        if not base_has_header:
            return [header_row] + data_rows
    return data_rows


def _merge_cluster(cluster):
    """Cụm nhiều mảnh -> 1 bảng. Phân biệt 2 tình huống bằng chồng lấn dọc."""
    overlapping = any(
        _vrange_overlap(cluster[i], cluster[j]) > 2.0
        for i in range(len(cluster))
        for j in range(i + 1, len(cluster))
    )
    if overlapping:
        return _merge_header_band(cluster)
    return _merge_continuation(cluster)


def extract_page_logical_tables(pdf_path, page, mode="stream"):
    """Trả list bảng logic của một trang (đã gộp mảnh + căn cột).

    Mỗi phần tử: {"rows": list[list[str]], "top_bbox": (x0,top,x1,bottom),
                  "n_fragments": int}. rows[0] là header (nếu khôi phục được).
    """
    raw = tabula.read_pdf(
        pdf_path,
        pages=str(page),
        stream=(mode == "stream"),
        lattice=(mode == "lattice"),
        multiple_tables=True,
        output_format="json",
        silent=True,
    )
    records = _norm_tables(raw)
    if not records:
        return []

    results = []
    for cluster in _cluster(records):
        if len(cluster) == 1:
            # pass-through nguyên trạng: giữ y hệt tabula để không regress trang tốt
            rows = _rows_as_text(cluster[0])
        else:
            rows = _merge_cluster(cluster)
            # chỉ với cụm đã gộp mới lọc: bỏ cụm không còn dữ liệu tài chính
            if not any(_is_data_row(r) for r in rows):
                continue

        x0 = min(r["left"] for r in cluster)
        top = min(r["top"] for r in cluster)
        x1 = max(r["right"] for r in cluster)
        bottom = max(r["bottom"] for r in cluster)
        results.append(
            {
                "rows": rows,
                "top_bbox": (x0, top, x1, bottom),
                "n_fragments": len(cluster),
            }
        )
    return results
