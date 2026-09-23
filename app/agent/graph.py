"""Agentic Adaptive RAG — LangGraph orchestration (v7).

Bật khi `AGENT_GRAPH=1`. Thay cho `create_agent` (tool-calling), một StateGraph điều phối
grade/rewrite/route/decompose/compute thành CÁC NODE TƯỜNG MINH. Python điều phối, KHÔNG phụ
thuộc LLM gọi tool (né nút thắt qwen2:7b không đáng tin khi gọi tool / tự nhẩm sai).

Luồng:
    route ──computed?──▶ decompose ─▶ compute ─▶ generate
       │                                             ▲
       └──factual?──▶ retrieve ─▶ grade ─(rewrite loop)┘

Điểm khác adaptive.py (chôn trong chokepoint `retrieve()`):
  - Vòng điều phối là graph, mỗi bước là node log/debug được.
  - Câu TÍNH TOÁN: con số tính bằng Python (từ operand ĐÃ XÁC MINH) là CHÂN LÝ — node
    `generate` chỉ để LLM diễn đạt lại, post-check giữ nguyên con số. Không verify được
    operand -> ABSTAIN (rơi về factual, không bịa số).

Interface giữ nguyên `.invoke({"messages": [...]}, config={"configurable": {...}})` -> đọc
`result["messages"][-1].content`, nên api.py / run_eval.py KHÔNG phải sửa. Reuse toàn bộ helper
LLM/tính trong app/rag/adaptive.py (chỉ re-wire thành node, không viết lại logic).
"""

import logging
import re
from typing import Annotated, TypedDict

from langchain_core.documents import Document
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages

from app.config import (
    ADAPTIVE_GRADER,
    ADAPTIVE_MAX_RETRIES,
    LLM_MODEL,
    OLLAMA_BASE_URL,
    OPERAND_CANDIDATES,
    RAG_TOP_K,
)
from app.ingestion.pdf_ingestion import get_latest_document_id
from app.rag.adaptive import (
    _OPERAND_STOP,
    _current_prior,
    _first_value,
    _period_values,
    _fmt_pct,
    _looks_like_computed_metric,
    _rerank_merge,
    analyze_query,
    decompose_metric,
    grade_retrieval,
    rewrite_query,
)
from app.rag.retrieval import hybrid_search

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# State
# --------------------------------------------------------------------------- #

class RagState(TypedDict, total=False):
    messages: Annotated[list, add_messages]  # lịch sử + AIMessage đáp án cuối
    question: str            # câu hỏi mới nhất (rút từ messages[-1])
    document_id: str | None
    qkind: str | None        # statement_kind từ analyzer -> lái section-boost
    route: str               # "factual" | "computed"
    docs: list               # ngữ cảnh cuối cùng dùng để sinh đáp / trích dẫn
    grade: dict
    retries: int
    spec: dict               # {op, parts} từ decompose_metric (đường computed)
    operand_docs: list       # list[list[Document]] — chunk đã khớp mỗi toán hạng
    compute: dict | None     # {operands_verified, result_str, pct_token} hoặc None


# --------------------------------------------------------------------------- #
# Answer LLM (sinh đáp án cuối; tách khỏi analyzer/grader/rewriter của adaptive.py)
# --------------------------------------------------------------------------- #

_ANSWER_LLM = None


def _get_answer_llm():
    global _ANSWER_LLM
    if _ANSWER_LLM is None:
        from langchain_ollama import ChatOllama

        _ANSWER_LLM = ChatOllama(model=LLM_MODEL, temperature=0, base_url=OLLAMA_BASE_URL)
    return _ANSWER_LLM


# System prompt cho sinh đáp án factual. Reuse nguyên bộ rule của agent tool-calling
# (cách đọc dòng/dấu/cột vẫn áp dụng), chỉ đổi Rule 1: ngữ cảnh được CUNG CẤP SẴN bên dưới
# (graph đã truy hồi) thay vì bảo LLM gọi tool.
from app.agent.financial_agent import SYSTEM_PROMPT as _AGENT_SYSTEM_PROMPT  # noqa: E402

_GEN_SYSTEM_PROMPT = _AGENT_SYSTEM_PROMPT.replace(
    """1. When the user asks about an uploaded PDF,
   use the search_pdf_knowledge_base tool. Pass only a
   search query; the current document is selected
   automatically, so never ask for or invent a document id.""",
    """1. The relevant lines from the uploaded document are
   provided below under "RETRIEVED CONTEXT". Answer the
   user's question using ONLY those lines. Never ask for a
   document id.""",
)

# System prompt cho đường COMPUTED (đã có con số tính sẵn tất định).
_COMPUTE_SYSTEM_PROMPT = """Bạn là trợ lý phân tích báo cáo tài chính.

Bạn được cho SẴN một kết quả phần trăm ĐÃ TÍNH CHÍNH XÁC bằng máy (tất định, đúng tuyệt đối).

Nhiệm vụ: viết MỘT câu trả lời ngắn gọn, tự nhiên cho câu hỏi của người dùng.

Quy tắc BẮT BUỘC:
1. GIỮ NGUYÊN con số phần trăm đã cho — KHÔNG tính lại, KHÔNG làm tròn khác, KHÔNG đổi số.
2. Nêu rõ chiều tăng/giảm nếu kết quả cho biết.
3. Viết bằng CÙNG ngôn ngữ với câu hỏi (câu tiếng Việt -> trả lời tiếng Việt; tiếng Anh ->
   tiếng Anh). Không trộn ngôn ngữ.
4. Ngắn gọn, không thêm phân tích thừa."""


# --------------------------------------------------------------------------- #
# Helpers dùng chung
# --------------------------------------------------------------------------- #

def _configurable(config) -> dict:
    return (config or {}).get("configurable") or {}


def _history_before_last(messages: list[BaseMessage]) -> list[BaseMessage]:
    """Các lượt trước (đa lượt) — bỏ câu hỏi hiện tại (message cuối)."""
    return list(messages[:-1]) if len(messages) > 1 else []


# --------------------------------------------------------------------------- #
# Nodes
# --------------------------------------------------------------------------- #

def route_node(state: RagState, config) -> dict:
    """Rút câu hỏi + document_id, phân loại (analyze) và định tuyến factual/computed."""
    messages = state["messages"]
    question = messages[-1].content if messages else ""
    conf = _configurable(config)
    document_id = conf.get("document_id") or get_latest_document_id()

    analysis = analyze_query(question)
    qkind = analysis.get("statement_kind")
    route = "computed" if _looks_like_computed_metric(question) else "factual"
    logger.info("graph.route: route=%s qkind=%s analysis=%s", route, qkind, analysis)

    return {
        "question": question,
        "document_id": document_id,
        "qkind": qkind,
        "route": route,
        "retries": 0,
    }


def retrieve_node(state: RagState) -> dict:
    """Đường factual: hybrid_search (+section-boost theo qkind)."""
    if not state.get("document_id"):
        return {"docs": []}
    docs = hybrid_search(
        state["question"], state["document_id"], RAG_TOP_K, qkind=state.get("qkind")
    )
    return {"docs": docs}


def grade_node(state: RagState) -> dict:
    grade = grade_retrieval(state["question"], state.get("docs") or [])
    logger.info("graph.grade(retry %d): %s", state.get("retries", 0), grade)
    return {"grade": grade}


def rewrite_node(state: RagState) -> dict:
    """Rewrite -> retrieve lại -> rerank-merge (giữ k) theo query GỐC."""
    docs = state.get("docs") or []
    new_query = rewrite_query(state["question"], docs)
    logger.info("graph.rewrite -> %r", new_query)
    new_docs = hybrid_search(
        new_query, state["document_id"], RAG_TOP_K, qkind=state.get("qkind")
    )
    merged = _rerank_merge(state["question"], list(docs) + list(new_docs), RAG_TOP_K)
    return {"docs": merged, "retries": state.get("retries", 0) + 1}


def _label_tokens(doc: Document) -> set:
    """Token nhãn của chunk (row_label + đầu nội dung), để khớp toán hạng."""
    text = (doc.metadata.get("row_label") or "") + " " + doc.page_content[:100]
    return set(re.findall(r"\w+", text.lower()))


def _content_tokens(part: str) -> list[str]:
    """Token nội-dung của mô tả toán hạng (bỏ stopword/năm/số/1 ký tự)."""
    return [t for t in re.findall(r"\w+", (part or "").lower())
            if t not in _OPERAND_STOP and not t.isdigit() and len(t) > 1]


def _match_operand_in_pool(part: str, pool: list[Document]) -> Document | None:
    """Tìm chunk DÒNG-CÓ-SỐ khớp toán hạng `part` trong `pool`, XỬ LÝ TÊN RIÊNG tổng quát.

    Nút thắt: LLM hay chèn tên công ty vào mô tả ("...của Dược Hậu Giang") — token này KHÔNG
    có trong nhãn dòng chỉ tiêu nên luật all-token loại nhầm dòng đúng. Cách tổng quát (không
    hardcode danh sách công ty): chỉ YÊU CẦU khớp những token của `part` mà THỰC SỰ xuất hiện
    trong nhãn của ÍT NHẤT một dòng-CÓ-SỐ của pool (vocab). Token tên riêng không nằm trong
    vocab (nhãn dòng số không chứa tên công ty) -> tự động bị bỏ; token metric (doanh/thu/
    thuần/lợi/nhuận/gộp...) được giữ -> vẫn phân biệt đúng dòng."""
    value_docs = [d for d in pool if _first_value([d]) is not None]
    if not value_docs:
        return None
    vocab: set = set()
    for d in value_docs:
        vocab |= _label_tokens(d)
    req = [t for t in _content_tokens(part) if t in vocab]
    if not req:  # không còn token metric nào để phân biệt -> không khẳng định
        return None
    # Các dòng khớp ĐỦ token metric. Nhiều dòng có thể khớp (dòng BÁO CÁO CHÍNH và dòng THUYẾT
    # MINH SEGMENT cùng mang nhãn "gross profit"/"net revenue"). Ưu tiên tổng quát (không hardcode
    # tên section):
    #   (1) nhãn có ÍT TOKEN THỪA nhất ngoài `req` -> dòng chỉ tiêu TỔNG/CHÍNH ("Net revenue",
    #       "Total gross profit") thắng dòng đủ điều kiện thừa ("Segment gross profit", "Gross
    #       profit energy generation and storage segment"). Đây là tín hiệu mạnh & khái quát.
    #   (2) ÍT period-value hơn -> né dòng breakdown segment nhiều cột (4-6 giá trị) mà
    #       `_first_value`/`_current_prior` sẽ lấy nhầm giá trị của MỘT segment.
    #   (3) tie-break theo THỨ TỰ retrieval (pool đã xếp theo độ liên quan).
    req_set = set(req)
    candidates = [
        (i, d) for i, d in enumerate(value_docs)
        if all(t in _label_tokens(d) for t in req)
    ]
    if not candidates:
        return None

    def _extra_label_tokens(doc: Document) -> int:
        label = doc.metadata.get("row_label") or doc.page_content[:80]
        return len(set(_content_tokens(label)) - req_set)

    candidates.sort(key=lambda it: (
        _extra_label_tokens(it[1]),
        len(_period_values(it[1].page_content)),
        it[0],
    ))
    return candidates[0][1]


def decompose_node(state: RagState) -> dict:
    """Đường computed: phân rã -> truy hồi + XÁC MINH từng toán hạng. Không phân rã được
    -> đặt docs = hybrid_search thường (compute sẽ abstain -> generate factual)."""
    question, document_id = state["question"], state.get("document_id")
    if not document_id:
        return {"docs": [], "spec": {}, "operand_docs": []}

    spec = decompose_metric(question)
    parts = spec.get("parts") or []
    base = hybrid_search(question, document_id, RAG_TOP_K, qkind=state.get("qkind"))
    if not parts:
        return {"docs": base, "spec": {}, "operand_docs": []}

    logger.info("graph.decompose: op=%s parts=%s", spec.get("op"), parts)
    result: list[Document] = list(base)
    seen = {d.page_content for d in base}
    operand_docs: list[list[Document]] = []
    for p in parts:
        # Pool = full-query base + sub-query (nới OPERAND_CANDIDATES vì dòng đúng có thể ngoài
        # top-3, vd "Doanh thu thuần" bị "Lợi nhuận thuần" đè, thực ở rank 7).
        pool = list(base) + hybrid_search(p, document_id, OPERAND_CANDIDATES)
        d = _match_operand_in_pool(p, pool)
        matches = [d] if d is not None else []
        if d is not None and d.page_content not in seen:
            seen.add(d.page_content)
            result.append(d)
        operand_docs.append(matches)

    return {"docs": result, "spec": spec, "operand_docs": operand_docs}


def compute_node(state: RagState) -> dict:
    """Tính % TẤT ĐỊNH từ operand đã xác minh. Chỉ khẳng định khi đủ số + hợp lý;
    ngược lại operands_verified=False (ABSTAIN, không bịa)."""
    spec = state.get("spec") or {}
    op = spec.get("op")
    operand_docs = state.get("operand_docs") or []
    compute = {"operands_verified": False, "result_str": "", "pct_token": ""}
    try:
        if op == "ratio" and len(operand_docs) >= 2:
            a = _first_value(operand_docs[0])
            b = _first_value(operand_docs[1])
            if a is not None and b not in (None, 0):
                pct = a / b * 100.0
                # Guard hợp lý: biên/tỷ trọng phải trong [0,100]. Ngoài khoảng =
                # gần như chắc chọn nhầm operand (bug 167,5%) -> abstain.
                if 0.0 <= pct <= 100.0:
                    token = f"{_fmt_pct(pct)}%"
                    compute = {
                        "operands_verified": True,
                        "pct_token": token,
                        "result_str": f"≈ {token} (= {a:.0f} / {b:.0f})",
                    }
        elif op == "growth" and operand_docs:
            cp = _current_prior(operand_docs[0])
            if cp and cp[1] != 0:
                cur, prior = cp
                pct = (cur - prior) / prior * 100.0
                chieu = "tăng" if pct >= 0 else "giảm"
                token = f"{_fmt_pct(abs(pct))}%"
                compute = {
                    "operands_verified": True,
                    "pct_token": token,
                    "result_str": f"{chieu} ≈ {token} "
                    f"(kỳ này {cur:.0f} so kỳ trước {prior:.0f})",
                }
    except Exception as exc:  # noqa: BLE001 - không bao giờ làm hỏng luồng
        logger.warning("graph.compute lỗi: %s", exc)
    logger.info("graph.compute: %s", compute)
    return {"compute": compute}


def generate_node(state: RagState, config) -> dict:
    """Sinh AIMessage đáp án cuối. Computed+verified: con số Python là chân lý (LLM chỉ
    diễn đạt, post-check giữ số). Ngược lại: factual từ docs."""
    conf = _configurable(config)
    sources_sink = conf.get("sources_sink")

    if not state.get("document_id"):
        return {"messages": [AIMessage(
            "No document has been uploaded yet. "
            "Please upload a PDF before asking questions."
        )]}

    docs = state.get("docs") or []
    # Populate citation side-channel từ ngữ cảnh cuối (cả hai đường).
    if isinstance(sources_sink, list):
        for doc in docs:
            sources_sink.append({
                "page": doc.metadata.get("page", "unknown"),
                "content_type": doc.metadata.get("content_type", "text"),
            })

    compute = state.get("compute")
    if compute and compute.get("operands_verified"):
        return {"messages": [_generate_computed(state, compute)]}

    if not docs:
        return {"messages": [AIMessage(
            "No relevant information was found in the uploaded PDF."
        )]}
    return {"messages": [_generate_factual(state, docs)]}


def _doc_currency_scale(docs: list[Document]) -> tuple[str, str]:
    """(currency, unit_scale) của tài liệu từ metadata chunk (gắn khi ingest). Fallback detect trên
    page_content nếu chunk cũ chưa có metadata (chưa re-ingest)."""
    for d in docs:
        cur = d.metadata.get("currency")
        if cur:
            return cur, d.metadata.get("unit_scale", "") or ""
    from app.ingestion.pdf_ingestion import _detect_currency_scale
    return _detect_currency_scale(" ".join(d.page_content for d in docs))


def _currency_directive(currency: str, scale: str) -> str:
    """Dòng chỉ thị ĐƠN VỊ TIỀN chèn vào prompt để Rule 11 gắn đúng currency/scale (thay hardcode VND)."""
    if currency == "VND":
        return "ĐƠN VỊ TIỀN CỦA TÀI LIỆU: VND (số tuyệt đối). Gắn 'VND' sau con số."
    scale_txt = f", số liệu tính theo {scale.upper()} ({scale})" if scale else ""
    add_word = f" '{scale} {currency}'" if scale else f" '{currency}'"
    return (
        f"ĐƠN VỊ TIỀN CỦA TÀI LIỆU: {currency}{scale_txt}. "
        f"Gắn{add_word} sau con số tiền tệ, KHÔNG dùng 'VND'. "
        f"Lãi trên cổ phiếu (EPS)/tỷ lệ giữ đơn vị của chính dòng đó (vd '{currency}/cổ phiếu'), "
        f"KHÔNG áp scale."
    )


def _generate_factual(state: RagState, docs: list[Document]) -> AIMessage:
    context = "\n\n".join(
        f"SOURCE PAGE: {d.metadata.get('page', 'unknown')} "
        f"[{'TABLE' if d.metadata.get('content_type') == 'table' else 'TEXT'}]\n\n"
        f"CONTENT:\n{d.page_content}"
        for d in docs
    )
    currency, scale = _doc_currency_scale(docs)
    user = (
        f"CÂU HỎI / QUESTION:\n{state['question']}\n\n"
        f"{_currency_directive(currency, scale)}\n\n"
        f"RETRIEVED CONTEXT:\n{context}"
    )
    llm_messages = (
        [SystemMessage(_GEN_SYSTEM_PROMPT)]
        + _history_before_last(state["messages"])
        + [HumanMessage(user)]
    )
    resp = _get_answer_llm().invoke(llm_messages)
    content = resp.content if isinstance(resp.content, str) else str(resp.content)
    return AIMessage(content)


def _generate_computed(state: RagState, compute: dict) -> AIMessage:
    """LLM diễn đạt lại kết quả tính sẵn. Post-check: nếu output không GIỮ NGUYÊN con số %
    thì dùng thẳng câu template Python (con số là chân lý, không bao giờ bị đổi)."""
    token = compute["pct_token"]
    canonical = f"{compute['result_str']}"
    user = (
        f"CÂU HỎI / QUESTION:\n{state['question']}\n\n"
        f"KẾT QUẢ ĐÃ TÍNH SẴN (giữ nguyên con số): {canonical}"
    )
    try:
        resp = _get_answer_llm().invoke(
            [SystemMessage(_COMPUTE_SYSTEM_PROMPT), HumanMessage(user)]
        )
        out = resp.content if isinstance(resp.content, str) else str(resp.content)
        if token in out:  # post-check: con số nguyên vẹn
            return AIMessage(out)
        logger.info("graph.generate(computed): LLM đổi/mất số -> dùng canonical")
    except Exception as exc:  # noqa: BLE001
        logger.warning("graph.generate(computed) LLM lỗi -> canonical: %s", exc)
    return AIMessage(canonical)


# --------------------------------------------------------------------------- #
# Edges (điều kiện)
# --------------------------------------------------------------------------- #

def _route_branch(state: RagState) -> str:
    return "decompose" if state.get("route") == "computed" else "retrieve"


def _grade_branch(state: RagState) -> str:
    """Đủ / hết retry / tắt grader -> generate; chưa đủ & còn retry -> rewrite."""
    if not ADAPTIVE_GRADER:
        return "generate"
    grade = state.get("grade") or {}
    if grade.get("sufficient", True):
        return "generate"
    if state.get("retries", 0) < ADAPTIVE_MAX_RETRIES:
        return "rewrite"
    return "generate"


# --------------------------------------------------------------------------- #
# Build
# --------------------------------------------------------------------------- #

def build_rag_graph():
    """Compile StateGraph. Trả object có `.invoke({"messages": [...]}, config=...)`."""
    g = StateGraph(RagState)
    g.add_node("route", route_node)
    g.add_node("retrieve", retrieve_node)
    g.add_node("grade", grade_node)
    g.add_node("rewrite", rewrite_node)
    g.add_node("decompose", decompose_node)
    g.add_node("compute", compute_node)
    g.add_node("generate", generate_node)

    g.add_edge(START, "route")
    g.add_conditional_edges("route", _route_branch,
                            {"retrieve": "retrieve", "decompose": "decompose"})
    g.add_edge("retrieve", "grade")
    g.add_conditional_edges("grade", _grade_branch,
                            {"rewrite": "rewrite", "generate": "generate"})
    g.add_edge("rewrite", "grade")
    g.add_edge("decompose", "compute")
    g.add_edge("compute", "generate")
    g.add_edge("generate", END)

    return g.compile()
