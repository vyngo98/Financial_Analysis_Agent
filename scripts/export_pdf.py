"""Export a (possibly scanned) PDF to structured files using Docling.

Writes results under an output directory with three sub-folders:
    <out>/text/    - full Markdown + per-page plain text
    <out>/table/   - each detected table as Markdown + CSV
    <out>/figure/  - each detected picture as PNG

Usage:
    python -m scripts.export_pdf <pdf_path> [out_dir] [--pages A-B]
"""

import argparse
import os
import warnings

warnings.filterwarnings("ignore")

from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import EasyOcrOptions, PdfPipelineOptions
from docling.document_converter import DocumentConverter, PdfFormatOption

from app.config import OCR_LANG


def build_converter():
    options = PdfPipelineOptions()
    options.do_ocr = True
    options.do_table_structure = True
    options.generate_picture_images = True  # keep cropped picture images
    options.images_scale = 2.0
    options.ocr_options = EasyOcrOptions(lang=list(OCR_LANG))
    return DocumentConverter(
        format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=options)}
    )


def _page_of(item):
    try:
        return item.prov[0].page_no
    except Exception:
        return None


def export_pdf(pdf_path, out_dir="pdf_export", page_range=None):
    text_dir = os.path.join(out_dir, "text")
    table_dir = os.path.join(out_dir, "table")
    figure_dir = os.path.join(out_dir, "figure")
    for d in (text_dir, table_dir, figure_dir):
        os.makedirs(d, exist_ok=True)

    converter = build_converter()
    kwargs = {"page_range": page_range} if page_range else {}
    result = converter.convert(pdf_path, **kwargs)
    doc = result.document

    # ---- TEXT: full markdown + per-page plain text ----
    with open(os.path.join(text_dir, "full_document.md"), "w") as f:
        f.write(doc.export_to_markdown())

    pages = {}
    for item in doc.texts:
        page = _page_of(item) or 0
        pages.setdefault(page, []).append(item.text)
    for page, lines in sorted(pages.items()):
        with open(os.path.join(text_dir, f"page_{page:03d}.txt"), "w") as f:
            f.write("\n".join(lines))

    # ---- TABLES: markdown + csv ----
    n_tables = 0
    for i, table in enumerate(doc.tables):
        page = _page_of(table)
        base = f"table_{i:03d}_p{page}"
        try:
            md = table.export_to_markdown(doc)
        except TypeError:
            md = table.export_to_markdown()
        with open(os.path.join(table_dir, base + ".md"), "w") as f:
            f.write(md)
        try:
            try:
                df = table.export_to_dataframe(doc)
            except TypeError:
                df = table.export_to_dataframe()
            df.to_csv(os.path.join(table_dir, base + ".csv"), index=False)
        except Exception:
            pass
        n_tables += 1

    # ---- FIGURES: png ----
    n_figures = 0
    for i, pic in enumerate(doc.pictures):
        page = _page_of(pic)
        img = None
        try:
            img = pic.get_image(doc)
        except Exception:
            img = None
        if img is None:
            continue
        img.save(os.path.join(figure_dir, f"figure_{i:03d}_p{page}.png"))
        n_figures += 1

    summary = {
        "pdf": os.path.basename(pdf_path),
        "pages": doc.num_pages() if hasattr(doc, "num_pages") else None,
        "text_pages_written": len(pages),
        "tables": n_tables,
        "figures": n_figures,
        "out_dir": out_dir,
    }
    print("EXPORT DONE:", summary)
    return summary


def _parse_pages(s):
    if not s:
        return None
    if "-" in s:
        a, b = s.split("-", 1)
        return (int(a), int(b))
    return (int(s), int(s))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("pdf_path")
    parser.add_argument("out_dir", nargs="?", default="pdf_export")
    parser.add_argument("--pages", default=None, help="e.g. 1-5 or 3")
    args = parser.parse_args()
    export_pdf(args.pdf_path, args.out_dir, _parse_pages(args.pages))
