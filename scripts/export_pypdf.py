"""Load a PDF with LangChain's PyPDFLoader (pypdf) and dump per-page text — to
compare against the pdfplumber-based native flow (v0/v1).

Lưu ý: PyPDFLoader KHÔNG trích bảng. Nó chỉ trả 1 Document text/trang (bọc
pypdf). Script này dump text đó để so chất lượng với bảng của pdfplumber.

Usage:
    python -m scripts.export_pypdf \
        data/uploads/vinamilk_2023_consolidated_financial_statements.pdf \
        pdf_export/vnm_pypdf
"""

import argparse
import os

from langchain_community.document_loaders import PyPDFLoader


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("pdf")
    ap.add_argument("out_dir")
    args = ap.parse_args()

    text_dir = os.path.join(args.out_dir, "text")
    os.makedirs(text_dir, exist_ok=True)

    # PyPDFLoader: 1 Document mỗi trang, metadata['page'] là 0-indexed.
    docs = PyPDFLoader(args.pdf).load()

    all_md = [f"# PyPDFLoader dump — {os.path.basename(args.pdf)}  ({len(docs)} trang)\n"]
    empty_pages = 0
    for d in docs:
        page0 = d.metadata.get("page", 0)
        page1 = page0 + 1  # 1-indexed để khớp metadata 'page' của flow app
        content = d.page_content or ""
        if not content.strip():
            empty_pages += 1
        with open(os.path.join(text_dir, f"page_{page1:02d}.md"), "w", encoding="utf-8") as f:
            f.write(f"<!-- page {page1} | PyPDFLoader (pypdf) | no table extraction -->\n\n")
            f.write(content)
            f.write("\n")
        all_md.append(f"\n\n---\n\n## Trang {page1}\n\n```\n{content}\n```\n")

    with open(os.path.join(args.out_dir, "all_text.md"), "w", encoding="utf-8") as f:
        f.write("".join(all_md))

    with open(os.path.join(args.out_dir, "INDEX.md"), "w", encoding="utf-8") as f:
        f.write(f"# INDEX — {os.path.basename(args.pdf)} (PyPDFLoader)\n\n")
        f.write(f"- Loader: `langchain_community.document_loaders.PyPDFLoader` (pypdf)\n")
        f.write(f"- Số trang / Document: **{len(docs)}**\n")
        f.write(f"- Trang rỗng (không có text-layer, vd trang scan): **{empty_pages}**\n")
        f.write(f"- Bảng trích được: **0** (PyPDFLoader không parse bảng)\n")

    print(f"Xong. pages={len(docs)}, empty={empty_pages}, tables=0 (PyPDFLoader không có bảng)")
    print(f"Xem: {os.path.join(args.out_dir, 'all_text.md')} và INDEX.md")


if __name__ == "__main__":
    main()
