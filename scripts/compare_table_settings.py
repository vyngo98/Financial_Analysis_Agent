"""So sánh các `table_settings` của pdfplumber trên các trang bảng để tìm cách
gom nguyên bảng mà KHÔNG làm hỏng cột số.

Kết luận thực nghiệm (VNM FY2023 — tài liệu native duy nhất trong data hiện tại;
các file còn lại là scan -> đi đường OCR/Docling, không dùng find_tables):

  - default (lines/lines): CỘT SẠCH nhưng phân mảnh bảng thành nhiều mảnh 1-dòng,
    và mất dòng header năm.
  - text/text, Vtext/Hlines: gom được thành 1 bảng nhưng LÀM HỎNG CỘT
    (số bị cắt đôi, vd "60,0 | 74,730,223,299"; nhãn bị vỡ).  => KHÔNG dùng.
  - Khuyến nghị: GIỮ default (cột sạch) + GỘP các mảnh cùng schema trên một trang
    + PREPEND dải header năm ngay phía trên bảng.  -> hàm merge_page_tables() minh hoạ.

Chạy:
    python -m scripts.compare_table_settings                # VNM, các trang mặc định
    python -m scripts.compare_table_settings <pdf> 7 8 10   # tuỳ chọn
"""

import sys
from collections import Counter

import pdfplumber

DEFAULT_PDF = "data/uploads/vinamilk_2023_consolidated_financial_statements.pdf"
DEFAULT_PAGES = [6, 7, 8, 10, 12, 13, 24, 30]

VARIANTS = {
    "default(lines/lines)": {},
    "text/text": {"vertical_strategy": "text", "horizontal_strategy": "text"},
    "Vtext/Hlines": {"vertical_strategy": "text", "horizontal_strategy": "lines"},
    "lines+tol": {"snap_tolerance": 6, "join_tolerance": 6, "intersection_tolerance": 6},
}


def _biggest(tabs):
    best = None
    for t in tabs:
        rows = t.extract()
        r = len(rows)
        if best is None or r > best[0]:
            best = (r, max((len(x) for x in rows), default=0))
    return best or (0, 0)


def compare(page, pno):
    print(f"\n===== page {pno} =====")
    for name, ts in VARIANTS.items():
        tabs = page.find_tables(table_settings=ts) if ts else page.find_tables()
        r, c = _biggest(tabs)
        print(f"  {name:<22} n_tables={len(tabs):<3} biggest={r}x{c}")


def _header_band(page, top_y, gap=70):
    words = [w for w in page.extract_words() if top_y - gap < w["bottom"] <= top_y]
    lines = {}
    for w in words:
        lines.setdefault(round(w["bottom"]), []).append(w)
    out = []
    for k in sorted(lines):
        ws = sorted(lines[k], key=lambda w: w["x0"])
        out.append(" ".join(w["text"] for w in ws))
    return out


def merge_page_tables(page):
    """GIỮ default settings, gộp các mảnh cùng số cột (schema chính) trên trang
    thành một bảng, kèm dải header phía trên. Trả về (header_lines, rows)."""
    frags = [(t.bbox, t.extract()) for t in page.find_tables()]
    frags = [(b, rows) for b, rows in frags if any(any(c for c in r) for r in rows)]
    if not frags:
        return [], []
    schema = Counter(max(len(r) for r in rows) for _, rows in frags).most_common(1)[0][0]
    keep = [(b, rows) for b, rows in frags if max(len(r) for r in rows) == schema]
    keep.sort(key=lambda x: x[0][1])
    merged = [r for _, rows in keep for r in rows if any(c for c in r)]
    header = _header_band(page, keep[0][0][1])
    return header, merged


def demo_merge(page, pno):
    header, rows = merge_page_tables(page)
    print(f"\n----- page {pno}: MERGE (default+gộp) -> 1 bảng, {len(rows)} dòng -----")
    print("  header:", [h for h in header if any(ch.isdigit() for ch in h)])
    for r in rows[:8]:
        print("    |", " | ".join((c or "").replace("\n", " ")[:24] for c in r), "|")


def main(argv):
    pdf_path = argv[0] if argv else DEFAULT_PDF
    pages = [int(x) for x in argv[1:]] if len(argv) > 1 else DEFAULT_PAGES
    pdf = pdfplumber.open(pdf_path)
    print(f"PDF: {pdf_path}  ({len(pdf.pages)} trang)")
    for pno in pages:
        page = pdf.pages[pno - 1]
        compare(page, pno)
        demo_merge(page, pno)
    pdf.close()


if __name__ == "__main__":
    main(sys.argv[1:])
