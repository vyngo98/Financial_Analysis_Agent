"""Extract tables from a PDF with tabula-py (Java tabula) and dump them — to
compare table quality against the pdfplumber native flow (v0/v1) and PyPDFLoader.

Dùng helper `extract_page_logical_tables` (app.ingestion.tabula_extract) để GỘP
các mảnh bị tabula tách nhầm theo bbox proximity + căn lại cột (sửa lỗi trang 10
bị tách header khỏi data). Render Markdown bằng chính `table_to_markdown` của
pipeline để giống hệt cái LLM sẽ đọc.

Tabula có 2 chế độ:
    - lattice : cho bảng CÓ đường kẻ (ruling lines).
    - stream  : cho bảng canh theo khoảng trắng (không kẻ).

Ghi ra:
    <out>/<mode>/table_p<page>_<idx>.csv / .md
    <out>/all_tables_<mode>.md
    <out>/INDEX.md

Usage:
    python -m scripts.export_tabula \
        data/uploads/vinamilk_2023_consolidated_financial_statements.pdf \
        pdf_export/vnm_tabula_merged --modes stream
"""

import argparse
import csv
import os

import pdfplumber

from app.ingestion.pdf_ingestion import table_to_markdown
from app.ingestion.tabula_extract import extract_page_logical_tables


def run_mode(pdf_path, out_dir, mode, num_pages):
    mode_dir = os.path.join(out_dir, mode)
    os.makedirs(mode_dir, exist_ok=True)

    all_md = [f"# Tabula ({mode}, gộp bbox) — {os.path.basename(pdf_path)}\n"]
    total = 0
    # Trích theo từng trang để giữ đúng số trang (pages='all' không gắn page).
    for page in range(1, num_pages + 1):
        try:
            logical = extract_page_logical_tables(pdf_path, page, mode)
        except Exception as e:
            all_md.append(f"\n\n> page {page}: ERROR {e}\n")
            continue

        for idx, lt in enumerate(logical):
            rows = lt["rows"]
            nrows = len(rows)
            ncols = max((len(r) for r in rows), default=0)
            base = f"table_p{page:02d}_{idx}"

            with open(os.path.join(mode_dir, base + ".csv"), "w", encoding="utf-8", newline="") as f:
                csv.writer(f).writerows(rows)

            md = table_to_markdown(rows)
            with open(os.path.join(mode_dir, base + ".md"), "w", encoding="utf-8") as f:
                f.write(
                    f"<!-- tabula {mode} | page {page} | table {idx} | "
                    f"shape=({nrows},{ncols}) | n_fragments={lt['n_fragments']} -->\n\n"
                )
                f.write(md + "\n")
            all_md.append(
                f"\n\n---\n\n## {mode} — page {page}, table {idx} "
                f"(shape=({nrows},{ncols}), gộp {lt['n_fragments']} mảnh)\n\n{md}\n"
            )
            total += 1
        if page % 10 == 0:
            print(f"    ...{mode}: đã qua trang {page}, {total} bảng")

    with open(os.path.join(out_dir, f"all_tables_{mode}.md"), "w", encoding="utf-8") as f:
        f.write("".join(all_md))

    return total


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("pdf")
    ap.add_argument("out_dir")
    ap.add_argument("--modes", nargs="+", default=["lattice", "stream"],
                    choices=["lattice", "stream"])
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    with pdfplumber.open(args.pdf) as pdf:
        num_pages = len(pdf.pages)

    counts = {}
    for mode in args.modes:
        print(f"Đang chạy tabula mode={mode} trên {num_pages} trang ...")
        counts[mode] = run_mode(args.pdf, args.out_dir, mode, num_pages)
        print(f"  -> {counts[mode]} bảng")

    with open(os.path.join(args.out_dir, "INDEX.md"), "w", encoding="utf-8") as f:
        f.write(f"# INDEX — {os.path.basename(args.pdf)} (tabula-py)\n\n")
        f.write("- Loader: `tabula-py` (Java tabula)\n")
        for mode, n in counts.items():
            f.write(f"- Mode **{mode}**: {n} bảng -> `{mode}/`, `all_tables_{mode}.md`\n")

    print("Xong.", counts)


if __name__ == "__main__":
    main()
