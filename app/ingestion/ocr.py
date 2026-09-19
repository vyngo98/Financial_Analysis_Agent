"""OCR for scanned PDF pages using Docling (layout + TableFormer + EasyOCR).

Chosen over a vision-LLM because it is deterministic (it will not invent or drop
financial numbers), reconstructs table structure via TableFormer, and runs on
CPU — leaving the GPU free for the chat LLM. Isolated here so the OCR engine can
be swapped without touching ingestion routing.
"""

from app.config import OCR_LANG

_converter = None


def _get_converter():
    """Lazily build (and cache) a Docling converter with Vietnamese OCR."""
    global _converter
    if _converter is None:
        from docling.datamodel.base_models import InputFormat
        from docling.datamodel.pipeline_options import (
            EasyOcrOptions,
            PdfPipelineOptions,
        )
        from docling.document_converter import DocumentConverter, PdfFormatOption

        options = PdfPipelineOptions()
        options.do_ocr = True
        options.do_table_structure = True
        options.ocr_options = EasyOcrOptions(lang=list(OCR_LANG))

        _converter = DocumentConverter(
            format_options={
                InputFormat.PDF: PdfFormatOption(pipeline_options=options)
            }
        )
    return _converter


def ocr_page_to_markdown(pdf_path: str, page_number: int) -> str:
    """OCR a single page (1-indexed) of a scanned PDF into Markdown.

    Tables are reconstructed as Markdown tables; numbers are preserved verbatim.
    """
    result = _get_converter().convert(
        pdf_path, page_range=(page_number, page_number)
    )
    return result.document.export_to_markdown().strip()


def split_markdown_blocks(markdown: str):
    """Split OCR Markdown into ("table" | "text", block) segments.

    Consecutive lines containing a pipe form a table block (kept atomic);
    everything else is accumulated as narrative text. Mirrors the "atomic
    table chunk" philosophy used for native PDF tables.
    """
    blocks = []
    lines = markdown.split("\n")
    buffer = []
    buffer_kind = None

    def flush():
        if buffer and "\n".join(buffer).strip():
            blocks.append((buffer_kind, "\n".join(buffer).strip()))

    for line in lines:
        kind = "table" if "|" in line and line.strip() else "text"
        # Blank lines stay with the current block.
        if not line.strip():
            if buffer:
                buffer.append(line)
            continue
        if buffer_kind is None:
            buffer_kind = kind
        if kind != buffer_kind:
            flush()
            buffer = []
            buffer_kind = kind
        buffer.append(line)

    flush()
    return blocks
