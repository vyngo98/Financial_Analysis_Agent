"""Dump the EXACT chunks that the production ingestion pushed into Qdrant for a
given document_id — so we can eyeball precisely what retrieval sees (text +
table row-chunks), organized by page.

Không trích lại từ PDF: đọc thẳng từ Qdrant (đúng artifact production đang chạy).
Dùng để soi vì sao 1 trang BCTC không lên top-k.

Ghi ra:
    <out>/text/page_XXX.md   - các narrative-chunk của trang (gộp)
    <out>/table/page_XXX.md  - các table row-chunk của trang (gộp)
    <out>/all_text.md        - tất cả narrative-chunk nối lại
    <out>/all_tables.md      - tất cả table row-chunk nối lại
    <out>/INDEX.md           - số chunk theo trang + metadata tóm tắt

Usage:
    python -m scripts.export_ingested_chunks <document_id> <out_dir>
"""

import argparse
import os
from collections import defaultdict

from app.rag.vector_store import build_document_filter, get_client
from app.config import QDRANT_COLLECTION


def scroll_all(client, collection, document_id):
    points = []
    offset = None
    while True:
        batch, offset = client.scroll(
            collection_name=collection,
            scroll_filter=build_document_filter(document_id),
            limit=500,
            offset=offset,
            with_payload=True,
            with_vectors=False,
        )
        points.extend(batch)
        if offset is None:
            break
    return points


def _meta_line(md):
    bits = []
    for key in ("statement", "section", "row_label", "code", "note", "table_index", "row_index"):
        v = md.get(key)
        if v not in (None, "", []):
            bits.append(f"{key}={v}")
    return " | ".join(bits)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("document_id")
    ap.add_argument("out_dir")
    args = ap.parse_args()

    client = get_client()
    points = scroll_all(client, QDRANT_COLLECTION, args.document_id)

    text_dir = os.path.join(args.out_dir, "text")
    table_dir = os.path.join(args.out_dir, "table")
    os.makedirs(text_dir, exist_ok=True)
    os.makedirs(table_dir, exist_ok=True)

    # gom theo (page, content_type), giữ thứ tự row_index/table_index nếu có
    by_page = defaultdict(lambda: {"table": [], "text": []})
    filename = None
    for p in points:
        pc = p.payload.get("page_content", "")
        md = p.payload.get("metadata", {})
        filename = filename or md.get("filename")
        page = md.get("page")
        ctype = md.get("content_type", "text")
        if ctype not in ("table", "text"):
            ctype = "text"
        by_page[page][ctype].append((md, pc))

    def _sort_key(item):
        md = item[0]
        return (md.get("table_index") or 0, md.get("row_index") or 0)

    all_text, all_tables, index_rows = [], [], []
    n_text = n_table = 0

    for page in sorted(by_page.keys(), key=lambda x: (x is None, x)):
        buckets = by_page[page]
        pno = page if page is not None else 0

        tbls = sorted(buckets["table"], key=_sort_key)
        txts = sorted(buckets["text"], key=_sort_key)

        if tbls:
            path = os.path.join(table_dir, f"page_{pno:03d}.md")
            with open(path, "w", encoding="utf-8") as f:
                f.write(f"# Trang {pno} — table row-chunks ({len(tbls)})\n\n")
                for md, pc in tbls:
                    ml = _meta_line(md)
                    f.write(f"---\n<!-- {ml} -->\n\n{pc}\n\n")
                    all_tables.append(f"\n\n---\n### p{pno} · {ml}\n\n{pc}\n")
            n_table += len(tbls)

        if txts:
            path = os.path.join(text_dir, f"page_{pno:03d}.md")
            with open(path, "w", encoding="utf-8") as f:
                f.write(f"# Trang {pno} — narrative chunks ({len(txts)})\n\n")
                for i, (md, pc) in enumerate(txts):
                    ml = _meta_line(md)
                    f.write(f"---\n<!-- chunk {i}{(' | ' + ml) if ml else ''} -->\n\n{pc}\n\n")
                    all_text.append(f"\n\n---\n### p{pno} · chunk {i}\n\n{pc}\n")
            n_text += len(txts)

        index_rows.append((pno, len(tbls), len(txts)))

    with open(os.path.join(args.out_dir, "all_tables.md"), "w", encoding="utf-8") as f:
        f.write(f"# Tất cả table row-chunk — {filename} (document_id={args.document_id})\n")
        f.write("".join(all_tables))

    with open(os.path.join(args.out_dir, "all_text.md"), "w", encoding="utf-8") as f:
        f.write(f"# Tất cả narrative chunk — {filename} (document_id={args.document_id})\n")
        f.write("".join(all_text))

    with open(os.path.join(args.out_dir, "INDEX.md"), "w", encoding="utf-8") as f:
        f.write(f"# INDEX — {filename}\n\n")
        f.write(f"- document_id: `{args.document_id}`\n")
        f.write(f"- Tổng chunk: **{len(points)}** (table **{n_table}**, narrative **{n_text}**)\n")
        f.write(f"- Số trang có chunk: **{len(index_rows)}**\n\n")
        f.write("| Trang | #table-chunk | #text-chunk |\n|---|---|---|\n")
        for pno, nt, nx in index_rows:
            f.write(f"| {pno} | {nt} | {nx} |\n")

    print(f"Xong. chunk={len(points)} (table={n_table}, text={n_text}), trang={len(index_rows)}")
    print(f"Xem: {os.path.join(args.out_dir, 'INDEX.md')}, all_tables.md, all_text.md")


if __name__ == "__main__":
    main()
