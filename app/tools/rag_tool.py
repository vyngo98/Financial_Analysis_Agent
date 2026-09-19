from langchain.tools import tool
from langchain_core.runnables import RunnableConfig

from app.config import ADAPTIVE_DECOMPOSE, RAG_TOP_K
from app.ingestion.pdf_ingestion import get_latest_document_id
from app.rag.retrieval import retrieve
from app.rag.adaptive import _looks_like_computed_metric, decompose_retrieve


@tool
def search_pdf_knowledge_base(
    query: str,
    config: RunnableConfig = None,
) -> str:
    """
    Search the uploaded PDF knowledge base.

    Use this tool whenever the user asks a question
    about the uploaded financial report. The document being
    queried is resolved automatically from the current session,
    so only provide a search query.

    Args:
        query: The search query describing the information needed.
    """

    # The document to search is context, not something the model picks.
    # Prefer an explicit id passed via config; otherwise fall back to the
    # most recently uploaded document.
    configurable = (config or {}).get("configurable") or {}
    document_id = configurable.get("document_id") or get_latest_document_id()

    if not document_id:
        return (
            "No document has been uploaded yet. "
            "Please upload a PDF before asking questions."
        )

    # Câu hỏi cần TÍNH % (tỷ suất/tăng trưởng): phân rã lấy đủ cả tử lẫn mẫu vào context
    # (mode-independent, tự fallback về retrieve khi không phân rã được). Câu thường đi
    # thẳng retrieve() như cũ (tiền lọc keyword nên không thêm chi phí).
    if ADAPTIVE_DECOMPOSE and _looks_like_computed_metric(query):
        results = decompose_retrieve(query, document_id, k=RAG_TOP_K)
    else:
        results = retrieve(query, document_id, k=RAG_TOP_K)

    if not results:
        return "No relevant information was found in the uploaded PDF."

    # Optional per-request sink: the caller (API) may pass a mutable list via
    # config to collect structured citations for display in the UI. Not part of
    # the text returned to the LLM.
    sources_sink = configurable.get("sources_sink")

    # ĐƠN VỊ TIỀN của tài liệu (gắn khi ingest) -> chèn đầu context để Rule 11 gắn đúng currency/scale
    # (thay hardcode 'VND'). Fallback detect trên text nếu chunk cũ chưa có metadata.
    cur = next((d.metadata.get("currency") for d in results if d.metadata.get("currency")), "")
    if cur:
        scale = next((d.metadata.get("unit_scale") for d in results
                      if d.metadata.get("currency")), "") or ""
    else:
        from app.ingestion.pdf_ingestion import _detect_currency_scale
        cur, scale = _detect_currency_scale(" ".join(d.page_content for d in results))
    if cur == "VND":
        currency_line = "ĐƠN VỊ TIỀN CỦA TÀI LIỆU: VND (số tuyệt đối)."
    else:
        scale_txt = f", số liệu tính theo {scale.upper()} ({scale})" if scale else ""
        add_word = f"'{scale} {cur}'" if scale else f"'{cur}'"
        currency_line = (
            f"ĐƠN VỊ TIỀN CỦA TÀI LIỆU: {cur}{scale_txt}. Gắn {add_word} sau con số, KHÔNG dùng 'VND' "
            f"(EPS/tỷ lệ giữ đơn vị của dòng)."
        )

    context = [currency_line]

    for doc in results:

        page = doc.metadata.get("page", "unknown")
        content_type = doc.metadata.get("content_type", "text")
        label = "TABLE" if content_type == "table" else "TEXT"

        if isinstance(sources_sink, list):
            sources_sink.append({"page": page, "content_type": content_type})

        context.append(
            f"""
SOURCE PAGE: {page} [{label}]

CONTENT:
{doc.page_content}
"""
        )

    return "\n\n".join(context)