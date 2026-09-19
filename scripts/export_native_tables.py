"""Dump the table (and narrative) chunks that the CURRENT native ingestion flow
(`_extract_text_page`, v1: merge fragments + recover year-header + keep loose
rows) produces for a PDF — so we can eyeball exactly what goes into Qdrant.

Không dùng Docling; chạy đúng đường native của app để soi vấn đề chunk hiện tại.

Ghi ra:
    <out>/table/  - mỗi table-chunk 1 file .md  (table_<page>_<idx>.md)
    <out>/text/   - mỗi narrative-chunk 1 file .md (text_<page>_<idx>.md)
    <out>/all_tables.md   - tất cả table-chunk nối lại để cuộn xem nhanh
    <out>/INDEX.md        - bảng tóm tắt số chunk theo trang

Usage:
    python -m scripts.export_native_tables \
        data/uploads/vinamilk_2023_consolidated_financial_statements.pdf \
        pdf_export/vnm_v1_native
"""

import argparse
import os

import pdfplumber
from langchain_core.documents import Document

from app.ingestion.pdf_ingestion import (
    _boilerplate_lines,
    _extract_text_page,
    _find_caption,
    _is_scanned_page,
    _not_within_bboxes,
    table_to_markdown,
)


def _extract_text_page_v0(page, document_id, filename, page_number):
    """Bản sao đúng logic v0 (baseline): mỗi fragment pdfplumber -> 1 chunk,
    dùng _find_caption, KHÔNG merge, KHÔNG header-band, KHÔNG lọc blob, KHÔNG
    giữ loose rows riêng. Giữ ở đây để export so sánh mà không đụng code production."""
    table_documents = []
    text_documents = []

    tables = page.find_tables()
    excluded_bboxes = [t.bbox for t in tables]

    for table_index, table in enumerate(tables):
        markdown = table_to_markdown(table.extract())
        if not markdown:
            continue
        caption, caption_bbox = _find_caption(page, table.bbox)
        if caption_bbox is not None:
            excluded_bboxes.append(caption_bbox)
        content = f"## {caption}\n{markdown}" if caption else markdown
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

    if excluded_bboxes:
        filtered = page.filter(lambda obj: _not_within_bboxes(obj, excluded_bboxes))
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


EXTRACTORS = {"v0": _extract_text_page_v0, "v1": _extract_text_page}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("pdf")
    ap.add_argument("out_dir")
    ap.add_argument("--flow", choices=["v0", "v1"], default="v1",
                    help="Luồng trích: v0 (baseline) hay v1 (merge+header). Mặc định v1.")
    args = ap.parse_args()

    extract = EXTRACTORS[args.flow]

    table_dir = os.path.join(args.out_dir, "table")
    text_dir = os.path.join(args.out_dir, "text")
    os.makedirs(table_dir, exist_ok=True)
    os.makedirs(text_dir, exist_ok=True)

    pdf = pdfplumber.open(args.pdf)
    boiler = _boilerplate_lines(pdf.pages)

    all_tables_md = []
    index_rows = []
    n_tables = n_texts = n_scanned = 0

    for page in pdf.pages:
        pno = page.page_number
        if _is_scanned_page(page, boiler):
            n_scanned += 1
            index_rows.append((pno, "SCAN (đường OCR — bỏ qua ở đây)", 0, 0))
            continue

        tables, texts = extract(page, "vnm", os.path.basename(args.pdf), pno)

        for t in tables:
            idx = t.metadata.get("table_index", 0)
            fname = f"table_p{pno:02d}_{idx}.md"
            with open(os.path.join(table_dir, fname), "w", encoding="utf-8") as f:
                f.write(f"<!-- page {pno} | table_index {idx} | source=native -->\n\n")
                f.write(t.page_content)
                f.write("\n")
            all_tables_md.append(
                f"\n\n---\n\n## Trang {pno} — table #{idx}\n\n{t.page_content}\n"
            )
            n_tables += 1

        for i, t in enumerate(texts):
            fname = f"text_p{pno:02d}_{i}.md"
            with open(os.path.join(text_dir, fname), "w", encoding="utf-8") as f:
                f.write(f"<!-- page {pno} | narrative chunk {i} | source=native -->\n\n")
                f.write(t.page_content)
                f.write("\n")
            n_texts += 1

        index_rows.append((pno, "native", len(tables), len(texts)))

    with open(os.path.join(args.out_dir, "all_tables.md"), "w", encoding="utf-8") as f:
        f.write(f"# Tất cả table-chunk (native flow v1) — {os.path.basename(args.pdf)}\n")
        f.write("".join(all_tables_md))

    with open(os.path.join(args.out_dir, "INDEX.md"), "w", encoding="utf-8") as f:
        f.write(f"# INDEX — {os.path.basename(args.pdf)} (native flow v1)\n\n")
        f.write(f"- Tổng table-chunk: **{n_tables}**\n")
        f.write(f"- Tổng narrative-chunk: **{n_texts}**\n")
        f.write(f"- Trang scan (đi OCR, không dump ở đây): **{n_scanned}**\n\n")
        f.write("| Trang | Loại | #table | #text |\n|---|---|---|---|\n")
        for pno, kind, nt, nx in index_rows:
            f.write(f"| {pno} | {kind} | {nt} | {nx} |\n")

    print(f"Xong. table={n_tables}, text={n_texts}, scan={n_scanned}")
    print(f"Xem: {os.path.join(args.out_dir, 'all_tables.md')} và INDEX.md")


if __name__ == "__main__":
    main()
