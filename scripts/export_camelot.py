"""Extract tables from a PDF with Camelot and dump them — to compare table
quality against pdfplumber (v0/v1), PyPDFLoader, and tabula.

Camelot flavors:
    - lattice : bảng CÓ đường kẻ (dùng Ghostscript + OpenCV).
    - stream  : bảng canh theo khoảng trắng.
Mỗi bảng Camelot kèm `parsing_report` (accuracy, whitespace) và `.page` -> ta ghi
kèm để so chất lượng.

Ghi ra:
    <out>/<flavor>/table_p<page>_<idx>.csv / .md
    <out>/all_tables_<flavor>.md
    <out>/INDEX.md

Usage:
    python -m scripts.export_camelot \
        data/uploads/vinamilk_2023_consolidated_financial_statements.pdf \
        pdf_export/vnm_camelot
"""

import argparse
import os
from collections import defaultdict

import camelot


def _df_to_markdown(df):
    df = df.fillna("")
    try:
        return df.to_markdown(index=False)
    except Exception:
        return df.to_csv(index=False)


def run_flavor(pdf_path, out_dir, flavor, edge_tol=None, row_tol=None):
    flavor_dir = os.path.join(out_dir, flavor)
    os.makedirs(flavor_dir, exist_ok=True)

    # edge_tol/row_tol chỉ áp cho stream. edge_tol lớn -> hết cắt cụt bảng;
    # row_tol lớn -> gom nhãn nhiều dòng thành 1 dòng (bớt vỡ dòng).
    kwargs = {}
    if flavor == "stream":
        if edge_tol is not None:
            kwargs["edge_tol"] = edge_tol
        if row_tol is not None:
            kwargs["row_tol"] = row_tol

    try:
        tables = camelot.read_pdf(pdf_path, pages="all", flavor=flavor, **kwargs)
    except Exception as e:
        print(f"  [flavor={flavor}] LỖI: {e}")
        with open(os.path.join(out_dir, f"all_tables_{flavor}.md"), "w", encoding="utf-8") as f:
            f.write(f"# Camelot ({flavor}) — LỖI: {e}\n")
        return 0

    all_md = [f"# Camelot ({flavor}) — {os.path.basename(pdf_path)} — {tables.n} bảng\n"]
    per_page = defaultdict(int)
    total = 0
    for t in tables:
        page = t.page
        idx = per_page[page]
        per_page[page] += 1

        report = t.parsing_report or {}
        acc = report.get("accuracy")
        ws = report.get("whitespace")
        base = f"table_p{page:02d}_{idx}"

        t.df.to_csv(os.path.join(flavor_dir, base + ".csv"), index=False, header=False)
        md = _df_to_markdown(t.df)
        with open(os.path.join(flavor_dir, base + ".md"), "w", encoding="utf-8") as f:
            f.write(
                f"<!-- camelot {flavor} | page {page} | table {idx} | "
                f"shape={t.df.shape} | accuracy={acc} whitespace={ws} -->\n\n"
            )
            f.write(md + "\n")
        all_md.append(
            f"\n\n---\n\n## {flavor} — page {page}, table {idx} "
            f"(shape={t.df.shape}, accuracy={acc}, whitespace={ws})\n\n{md}\n"
        )
        total += 1

    with open(os.path.join(out_dir, f"all_tables_{flavor}.md"), "w", encoding="utf-8") as f:
        f.write("".join(all_md))
    return total


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("pdf")
    ap.add_argument("out_dir")
    ap.add_argument("--flavors", nargs="+", default=["lattice", "stream"])
    ap.add_argument("--edge-tol", type=int, default=None,
                    help="Camelot stream edge_tol (mặc định 50). Tăng (vd 500) để hết cắt cụt.")
    ap.add_argument("--row-tol", type=int, default=None,
                    help="Camelot stream row_tol (mặc định 2). Tăng (vd 15) để bớt vỡ dòng.")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    counts = {}
    for flavor in args.flavors:
        print(f"Đang chạy camelot flavor={flavor} (edge_tol={args.edge_tol}, row_tol={args.row_tol}) ...")
        counts[flavor] = run_flavor(args.pdf, args.out_dir, flavor, args.edge_tol, args.row_tol)
        print(f"  -> {counts[flavor]} bảng")

    with open(os.path.join(args.out_dir, "INDEX.md"), "w", encoding="utf-8") as f:
        f.write(f"# INDEX — {os.path.basename(args.pdf)} (camelot)\n\n")
        f.write("- Loader: `camelot-py`\n")
        for flavor, n in counts.items():
            f.write(f"- Flavor **{flavor}**: {n} bảng -> `{flavor}/`, `all_tables_{flavor}.md`\n")

    print("Xong.", counts)


if __name__ == "__main__":
    main()
