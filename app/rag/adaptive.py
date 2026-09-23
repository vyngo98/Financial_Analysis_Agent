"""Adaptive RAG — Phase 1 (v6).

Bật khi `RETRIEVAL_MODE="adaptive"`. Nhúng vào chokepoint `retrieve()` nên tool +
eval + agent dùng chung, KHÔNG đổi contract `retrieve(query, document_id, k) -> [Document]`.

Luồng:
    query -> Query Analyzer (1 LLM call, JSON: type/complexity/statement_kind/strategy)
          -> hybrid_search(query, k, qkind=statement_kind)          # Retrieve + Reranker
          -> Retrieval Grader (1 LLM call: ngữ cảnh ĐỦ trả lời chưa?)
             - đủ  -> trả top-k
             - chưa-> rewrite_query -> hybrid_search lại -> rerank-merge (giữ k) -> chấm lại
                      (bounded ADAPTIVE_MAX_RETRIES; hết retry -> trả best-so-far)

Vì sao Phase 1 chỉ Analyzer + Grader + Rewrite (Decomposition/HyDE để Phase 2):
  - Theo PERFORMANCE_REPORT, retrieval đã ổn; nút thắt là câu suy luận/mơ hồ + chống
    bịa số. Grader bắt "chưa đủ" để rewrite→thử lại; Analyzer.statement_kind lái
    section-boost chuẩn cho câu TIẾNG VIỆT (keyword _classify_query hay trơ với VI/OCR).

An toàn: MỌI lỗi LLM/parse -> fallback về hybrid_search thuần (không bao giờ làm hỏng
truy hồi). Mỗi tầng bật/tắt độc lập qua config để đo đóng góp riêng và giảm độ trễ.
"""

import json
import logging
import re

from langchain_core.documents import Document

from app.config import (
    ADAPTIVE_ANALYZER,
    ADAPTIVE_GRADER,
    ADAPTIVE_LLM_MODEL,
    ADAPTIVE_MAX_RETRIES,
    OLLAMA_BASE_URL,
)
from app.rag.retrieval import (
    _classify_query,
    _dedup,
    get_reranker,
    hybrid_search,
)

logger = logging.getLogger(__name__)

# statement_kind hợp lệ (khớp metadata chunk + _QUERY_KIND_KEYWORDS trong retrieval.py).
_VALID_KINDS = {
    "balance_sheet",
    "income_statement",
    "cash_flow",
    "company_profile",
}

_LLM = None


def _get_adaptive_llm():
    """Lazy singleton ChatOllama cho analyzer/grader/rewriter (tách khỏi LLM sinh đáp án)."""
    global _LLM
    if _LLM is None:
        from langchain_ollama import ChatOllama

        _LLM = ChatOllama(model=ADAPTIVE_LLM_MODEL, temperature=0, base_url=OLLAMA_BASE_URL)
    return _LLM


def _parse_json(text: str) -> dict:
    """Rút khối {...} đầu tiên rồi json.loads. {} nếu không parse được (không raise).

    qwen2:7b hay bọc JSON trong chữ thừa/markdown fence -> lấy khối ngoặc nhọn đầu tiên.
    Cùng cách xử lý như parse_judge_json trong app/eval/run_eval.py.
    """
    m = re.search(r"\{.*\}", text or "", re.DOTALL)
    if not m:
        return {}
    try:
        obj = json.loads(m.group(0))
        return obj if isinstance(obj, dict) else {}
    except (ValueError, TypeError):
        return {}


def _ask_llm(system: str, user: str) -> str:
    """Gọi 1 lượt chat, trả content chuỗi ('' nếu lỗi)."""
    llm = _get_adaptive_llm()
    resp = llm.invoke(
        [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]
    )
    content = resp.content
    return content if isinstance(content, str) else str(content)


# --------------------------------------------------------------------------- #
# Query Analyzer
# --------------------------------------------------------------------------- #

_ANALYZER_SYSTEM = """Bạn phân tích CÂU HỎI về một báo cáo tài chính để định tuyến truy hồi.
Trả về ĐÚNG một JSON một dòng, KHÔNG thêm chữ nào khác:
{"query_type": "...", "complexity": "...", "statement_kind": "...", "strategy": "..."}

- query_type: "factual" (hỏi 1 con số/dữ kiện cụ thể) | "reasoning" (cần tính toán/suy luận,
  vd tỉ lệ %, tăng trưởng) | "semantic" (mô tả/định nghĩa, câu chữ mơ hồ).
- complexity: "simple" (một chỉ tiêu) | "complex" (nhiều chỉ tiêu / nhiều bước).
- statement_kind: báo cáo chứa câu trả lời — "balance_sheet" (bảng cân đối: tài sản, nợ,
  vốn chủ sở hữu, hàng tồn kho) | "income_statement" (kết quả KD: doanh thu, giá vốn, lợi
  nhuận, EPS) | "cash_flow" (lưu chuyển tiền: dòng tiền HĐKD/đầu tư/tài chính, cổ tức) |
  "company_profile" (tên/trụ sở/kiểm toán/kỳ báo cáo) | "none" (không rõ).
- strategy: "simple" | "complex" | "semantic" (gợi ý cách xử lý)."""


def analyze_query(query: str) -> dict:
    """Phân loại câu hỏi -> {query_type, complexity, statement_kind, strategy}.

    Fallback khi tắt/LLM lỗi/parse lỗi: strategy="simple", statement_kind lấy từ
    _classify_query (keyword). statement_kind chuẩn hoá về _VALID_KINDS hoặc None.
    """
    fallback = {
        "query_type": "factual",
        "complexity": "simple",
        "statement_kind": _classify_query(query),
        "strategy": "simple",
    }
    if not ADAPTIVE_ANALYZER:
        return fallback
    try:
        raw = _ask_llm(_ANALYZER_SYSTEM, f"CÂU HỎI:\n{query}")
        data = _parse_json(raw)
        if not data:
            return fallback
        kind = data.get("statement_kind")
        if kind not in _VALID_KINDS:
            # LLM trả "none"/rác -> thử keyword để không mất section-boost.
            kind = _classify_query(query)
        return {
            "query_type": data.get("query_type") or "factual",
            "complexity": data.get("complexity") or "simple",
            "statement_kind": kind,
            "strategy": data.get("strategy") or "simple",
        }
    except Exception as exc:  # noqa: BLE001 - không bao giờ làm hỏng truy hồi
        logger.warning("analyze_query lỗi, dùng fallback: %s", exc)
        return fallback


# --------------------------------------------------------------------------- #
# Retrieval Grader
# --------------------------------------------------------------------------- #

_GRADER_SYSTEM = """Bạn chấm xem NGỮ CẢNH truy hồi có ĐỦ để trả lời CÂU HỎI không.
Chỉ dựa vào ngữ cảnh được cho, KHÔNG dùng kiến thức ngoài.
Trả về ĐÚNG một JSON một dòng:
{"sufficient": true hoặc false, "reason": "giải thích ngắn"}

- sufficient=true CHỈ khi có ít nhất một dòng/đoạn khớp TRỰC TIẾP điều câu hỏi cần
  (đúng chỉ tiêu, đúng con số/dữ kiện). KHÔNG suy đoán, KHÔNG bịa.
- sufficient=false nếu ngữ cảnh chỉ liên quan mơ hồ, thiếu con số/chỉ tiêu được hỏi,
  hoặc lạc chủ đề."""


def _format_context(docs: list[Document]) -> str:
    blocks = []
    for i, d in enumerate(docs, start=1):
        page = d.metadata.get("page", "?")
        blocks.append(f"[{i}] (trang {page})\n{d.page_content}")
    return "\n\n".join(blocks)


def grade_retrieval(query: str, docs: list[Document]) -> dict:
    """Chấm TỔNG THỂ top-k có đủ trả lời câu hỏi không -> {sufficient, reason}.

    Rỗng docs -> sufficient=False. LLM/parse lỗi -> sufficient=True (fail-open: không
    ép rewrite khi không chắc, tránh kéo thêm nhiễu; câu thật sự thiếu đã hiếm)."""
    if not docs:
        return {"sufficient": False, "reason": "no docs"}
    try:
        user = f"CÂU HỎI:\n{query}\n\nNGỮ CẢNH:\n{_format_context(docs)}"
        data = _parse_json(_ask_llm(_GRADER_SYSTEM, user))
        if "sufficient" not in data:
            return {"sufficient": True, "reason": "grader parse-fail (fail-open)"}
        return {
            "sufficient": bool(data.get("sufficient")),
            "reason": str(data.get("reason", ""))[:200],
        }
    except Exception as exc:  # noqa: BLE001
        logger.warning("grade_retrieval lỗi, fail-open: %s", exc)
        return {"sufficient": True, "reason": f"grader error: {exc}"}


# --------------------------------------------------------------------------- #
# Query Rewriter
# --------------------------------------------------------------------------- #

_REWRITER_SYSTEM = """Bạn viết lại CÂU HỎI để truy hồi báo cáo tài chính TỐT HƠN.
Giữ nguyên ý định; làm rõ bằng thuật ngữ kế toán chuẩn (VAS), thêm loại báo cáo / năm
nếu suy ra được. KHÔNG trả lời câu hỏi, KHÔNG bịa con số.
Chỉ in ĐÚNG câu hỏi đã viết lại trên một dòng, KHÔNG thêm nhãn/tiêu đề/giải thích."""

# Nhãn dẫn mà model hay chèn trước câu viết lại (qwen2:7b: "CÁCH VIẾT LẠI:", "Câu hỏi
# viết lại:"...). part TRƯỚC ':' đầu tiên ngắn + không có chữ số/'?' => coi là nhãn, bỏ đi.
_LABEL_RE = re.compile(r"^[^:?\d]{1,30}:\s*")


def _clean_rewrite_line(line: str) -> str:
    line = line.strip().strip('"').strip("*").strip()
    line = _LABEL_RE.sub("", line).strip()  # bỏ nhãn dẫn nếu có
    return line


def rewrite_query(query: str, docs: list[Document]) -> str:
    """Viết lại query cho lần retrieve kế. Trả query gốc nếu rỗng/lỗi/không hợp lệ.

    qwen2:7b hay chèn dòng nhãn ("CÁCH VIẾT LẠI:") -> bỏ dòng chỉ-có-nhãn và tách
    phần sau ':'. Query viết lại phải đủ dài (>= 8 ký tự); nếu không, giữ query gốc.
    """
    try:
        hint = ""
        if docs:
            kinds = {d.metadata.get("statement_kind") for d in docs if d.metadata.get("statement_kind")}
            if kinds:
                hint = f"\n(Ngữ cảnh vừa truy hồi thuộc: {', '.join(sorted(k for k in kinds if k))})"
        out = _ask_llm(_REWRITER_SYSTEM, f"CÂU HỎI:\n{query}{hint}")
        for raw in out.splitlines():
            line = _clean_rewrite_line(raw)
            if len(line) >= 8:  # bỏ dòng nhãn cụt ("CÁCH VIẾT LẠI:") / rỗng
                return line
        return query
    except Exception as exc:  # noqa: BLE001
        logger.warning("rewrite_query lỗi, giữ query gốc: %s", exc)
        return query


# --------------------------------------------------------------------------- #
# Rerank-merge (gộp 2 pool truy hồi, KHÔNG tăng k)
# --------------------------------------------------------------------------- #

def _rerank_merge(query: str, docs: list[Document], k: int) -> list[Document]:
    """Dedup (old ∪ new) rồi cross-encoder rerank theo query GỐC -> top-k.

    Giữ k=RAG_TOP_K (không phình pool cho LLM sinh đáp án — k lớn gây bịa số ở câu âm-tính)."""
    pool = _dedup(docs)
    if len(pool) <= k:
        return pool
    reranker = get_reranker()
    scores = [float(s) for s in reranker.predict([(query, d.page_content) for d in pool])]
    order = sorted(range(len(pool)), key=lambda i: scores[i], reverse=True)
    return [pool[i] for i in order[:k]]


# --------------------------------------------------------------------------- #
# Query decomposition cho câu TÍNH TOÁN (tỷ suất / tăng trưởng %) — Phase 2
# --------------------------------------------------------------------------- #

# Tiền lọc KEYWORD (rẻ): chỉ câu khớp mới gọi LLM phân rã -> câu thường KHÔNG bị đụng.
_COMPUTED_METRIC_KW = (
    "%", "phần trăm", "phan tram", "biên lợi nhuận", "bien loi nhuan", "tỷ suất",
    "ty suat", "tỷ lệ", "ty le", "tỷ trọng", "ty trong", "chiếm bao nhiêu",
    "tăng bao nhiêu", "giảm bao nhiêu", "tăng hay giảm", "tăng trưởng", "tang truong",
    "margin", "growth", "share of", "percentage", "percent",
)


def _looks_like_computed_metric(query: str) -> bool:
    s = (query or "").lower()
    return any(kw in s for kw in _COMPUTED_METRIC_KW)


_DECOMPOSE_SYSTEM = """Bạn phân rã một CÂU HỎI cần TÍNH phần trăm thành các chỉ tiêu cần tra.
Trả về ĐÚNG một JSON một dòng:
{"op": "ratio" hoặc "growth", "parts": ["...", "..."]}

NGÔN NGỮ (BẮT BUỘC): viết mô tả trong "parts" CÙNG ngôn ngữ với CÂU HỎI, dùng thuật ngữ kế
toán chuẩn của ngôn ngữ đó. Câu tiếng Anh -> thuật ngữ tiếng Anh (gross profit, net revenue /
total net sales, net income, cost of sales...). Câu tiếng Việt -> thuật ngữ VAS (lợi nhuận gộp,
doanh thu thuần, lợi nhuận sau thuế...). TUYỆT ĐỐI không trộn hai ngôn ngữ trong một mô tả.

- op="ratio": tỷ suất/tỷ trọng = tử / mẫu × 100 (vd biên lợi nhuận gộp, tỷ trọng doanh thu).
  parts = [mô tả chỉ tiêu TỬ, mô tả chỉ tiêu MẪU], kèm năm nếu có.
  VI: "biên lợi nhuận gộp 2025" -> {"op":"ratio","parts":["lợi nhuận gộp 2025","doanh thu thuần 2025"]}
  EN: "gross profit margin 2023" -> {"op":"ratio","parts":["gross profit 2023","net revenue 2023"]}
- op="growth": mức tăng/giảm so kỳ trước = (kỳ này − kỳ trước)/kỳ trước × 100.
  parts = [mô tả MỘT chỉ tiêu] (giá trị hai năm nằm cùng một dòng).
  VI: "doanh thu tăng bao nhiêu % so 2024" -> {"op":"growth","parts":["doanh thu thuần"]}
  EN: "how much did revenue grow vs 2022" -> {"op":"growth","parts":["net revenue"]}
Mỗi mô tả trong "parts" CHỈ gồm thuật ngữ kế toán (+ năm nếu cần); KHÔNG chèn tên công ty / tên
riêng / mã ticker (vd bỏ "của Sabeco", "Vinamilk") vì nhãn chỉ tiêu BCTC không chứa chúng.
Chỉ dùng thuật ngữ kế toán chuẩn; KHÔNG bịa số."""


def decompose_metric(query: str) -> dict:
    """Phân rã câu tính % -> {op, parts}. {} nếu tắt/không phải câu tính/parse lỗi."""
    if not ADAPTIVE_ANALYZER or not _looks_like_computed_metric(query):
        return {}
    try:
        data = _parse_json(_ask_llm(_DECOMPOSE_SYSTEM, f"CÂU HỎI:\n{query}"))
        op = data.get("op")
        parts = data.get("parts")
        if op not in ("ratio", "growth") or not isinstance(parts, list):
            return {}
        parts = [str(p).strip() for p in parts if str(p).strip()]
        return {"op": op, "parts": parts} if parts else {}
    except Exception as exc:  # noqa: BLE001
        logger.warning("decompose_metric lỗi: %s", exc)
        return {}


# Rút các giá trị cột (theo thứ tự) từ một chunk row: "... = 9.300.560.824.792; ... =
# (6.787.749.735.923)" -> [9300560824792.0, -6787749735923.0]. Ngoặc = số ÂM.
# Nhận cả dấu phân tách nghìn KIỂU CHẤM (VAS/tiếng Việt: 9.300.560) LẪN KIỂU PHẨY (tiếng Anh:
# 24,544,731,615,410 — bản VNM/aapl/tsla), vì cùng regex phục vụ cả hai. Số BCTC là số nguyên
# nên bỏ MỌI dấu phân tách (`.`/`,`) là an toàn (không có case thập phân thật).
_PERIOD_VAL_RE = re.compile(r"=\s*(\()?\s*(\d[\d.,]{4,})\s*(\))?")


def _period_values(text: str) -> list[float]:
    vals = []
    for m in _PERIOD_VAL_RE.finditer(text or ""):
        digits = re.sub(r"[.,]", "", m.group(2))
        if not digits.isdigit():
            continue
        v = float(digits)
        if m.group(1) or m.group(3):
            v = -v
        vals.append(v)
    return vals


def _fmt_pct(pct: float) -> str:
    return f"{pct:.1f}".replace(".", ",")


# Từ vô nghĩa khi khớp nhãn toán hạng với chunk (năm/đơn vị/từ chung).
# LƯU Ý: token tên công ty (LLM hay chèn "...của Sabeco/Dược Hậu Giang") được xử lý TỔNG QUÁT
# ở graph `_match_operand_in_pool` (chỉ giữ token có trong nhãn dòng-CÓ-SỐ), không hardcode ở đây.
_OPERAND_STOP = frozenset(
    "nam năm vnd trong cua của va và the of in for năm2025 2025 2024 2023 2022".split()
)


def _operand_matches(part: str, doc: Document) -> bool:
    """Chunk có ĐÚNG là toán hạng được yêu cầu không? Yêu cầu MỌI token nội-dung của mô tả part
    xuất hiện dưới dạng TỪ NGUYÊN VẸN trong nhãn chunk. Dùng token nguyên vẹn (không chuỗi con)
    để "thu" không khớp trong "thuần"; và yêu cầu ĐỦ token để phân biệt "doanh thu thuần" với
    "giảm trừ doanh thu" / "lợi nhuận thuần" (thiếu 'thuần' hoặc 'doanh'/'thu' -> loại)."""
    toks = [t for t in re.findall(r"\w+", (part or "").lower())
            if t not in _OPERAND_STOP and not t.isdigit() and len(t) > 1]
    if not toks:
        return True
    label_toks = set(re.findall(r"\w+", ((doc.metadata.get("row_label") or "")
                                         + " " + doc.page_content[:100]).lower()))
    return all(t in label_toks for t in toks)


def _first_value(docs: list[Document]) -> float | None:
    """Giá trị KỲ HIỆN TẠI (cột đầu) từ chunk đầu tiên có số. None nếu không có."""
    for d in docs:
        v = _period_values(d.page_content)
        if v:
            return v[0]
    return None


def _current_prior(docs: list[Document]) -> tuple[float, float] | None:
    """(kỳ này, kỳ trước) cho câu growth. Ưu tiên chunk có ĐỦ 2 cột (VAS thường); nếu các
    chunk chỉ có 1 giá trị (OCR tách năm nay/năm trước) thì ghép giá trị đầu của 2 chunk."""
    for d in docs:
        v = _period_values(d.page_content)
        if len(v) >= 2:
            return v[0], v[1]
    singles = [_period_values(d.page_content)[0] for d in docs if _period_values(d.page_content)]
    if len(singles) >= 2:
        return singles[0], singles[1]
    return None


def _compute_hint(op: str, parts: list[str], operand_docs: list[list[Document]]) -> str | None:
    """Tính % TẤT ĐỊNH -> dòng '[KẾT QUẢ TÍNH SẴN]'. None nếu không đủ số (không bịa).

    `operand_docs[i]` = các chunk ĐÃ KHỚP toán hạng parts[i] (đã lọc `_operand_matches` ở
    decompose_retrieve nên không cần guard lại). Không phụ thuộc tool-calling của LLM."""
    try:
        if op == "ratio" and len(operand_docs) >= 2:
            a = _first_value(operand_docs[0])
            b = _first_value(operand_docs[1])
            if a is not None and b not in (None, 0):
                return (f"[KẾT QUẢ TÍNH SẴN] Tỷ suất ≈ {_fmt_pct(a / b * 100)}% "
                        f"(= {a:.0f} / {b:.0f}).")
        if op == "growth" and operand_docs:
            cp = _current_prior(operand_docs[0])
            if cp and cp[1] != 0:
                cur, prior = cp
                pct = (cur - prior) / prior * 100
                chieu = "tăng" if pct >= 0 else "giảm"
                return (f"[KẾT QUẢ TÍNH SẴN] {chieu} ≈ {_fmt_pct(abs(pct))}% "
                        f"(kỳ này {cur:.0f} so kỳ trước {prior:.0f}).")
    except Exception as exc:  # noqa: BLE001
        logger.warning("_compute_hint lỗi: %s", exc)
    return None


def decompose_retrieve(query: str, document_id: str, k: int) -> list[Document]:
    """Truy hồi cho câu tính %: lấy ĐỦ mỗi toán hạng (sub-query) + ngữ cảnh query gốc, và
    CHÈN dòng '[KẾT QUẢ TÍNH SẴN]' tính sẵn (tất định) lên đầu.

    Đảm bảo cả TỬ lẫn MẪU cùng vào context (nút thắt câu tỷ suất) + không phụ thuộc LLM gọi
    tool tính. Fallback hybrid_search khi không phân rã được / lỗi."""
    try:
        spec = decompose_metric(query)
        parts = spec.get("parts") or []
        if not parts:
            return hybrid_search(query, document_id, k)
        op = spec.get("op")
        logger.info("decompose: op=%s parts=%s", op, parts)

        # NỀN = full-query, GIỮ NGUYÊN thứ tự (không nhấc sub-query lên đầu).
        base = hybrid_search(query, document_id, k)
        result: list[Document] = list(base)
        seen: set = {d.page_content for d in base}

        # Với mỗi toán hạng: tìm chunk khớp trong base; nếu THIẾU -> augment (sub-query) và
        # APPEND vào CUỐI (không đảo thứ tự base).
        operand_docs: list[list[Document]] = []
        for p in parts:
            matches = [d for d in base if _operand_matches(p, d)]
            if not matches:
                for d in hybrid_search(p, document_id, 3):
                    if _operand_matches(p, d):
                        matches.append(d)
                        if d.page_content not in seen:
                            seen.add(d.page_content)
                            result.append(d)
                        break
            operand_docs.append(matches)

        hint = _compute_hint(op, parts, operand_docs)
        if hint:
            logger.info("decompose: %s", hint)
            result.insert(0, Document(
                page_content=hint,
                metadata={"page": "—", "content_type": "text", "source": "computed"},
            ))
        return result
    except Exception as exc:  # noqa: BLE001
        logger.warning("decompose_retrieve lỗi, fallback hybrid_search: %s", exc)
        return hybrid_search(query, document_id, k)


# --------------------------------------------------------------------------- #
# Orchestrator
# --------------------------------------------------------------------------- #

def adaptive_retrieve(query: str, document_id: str, k: int) -> list[Document]:
    """Adaptive RAG Phase 1. Fallback về hybrid_search thuần khi có lỗi bất kỳ."""
    try:
        analysis = analyze_query(query)
        qkind = analysis.get("statement_kind")
        logger.info("adaptive: analysis=%s", analysis)

        docs = hybrid_search(query, document_id, k, qkind=qkind)

        if not ADAPTIVE_GRADER:
            return docs

        grade = grade_retrieval(query, docs)
        logger.info("adaptive: grade=%s", grade)

        retries = 0
        while not grade["sufficient"] and retries < ADAPTIVE_MAX_RETRIES:
            new_query = rewrite_query(query, docs)
            logger.info("adaptive: rewrite -> %r", new_query)
            new_docs = hybrid_search(new_query, document_id, k, qkind=qkind)
            # Rerank-merge theo query GỐC (giữ đúng ý định), cắt về k.
            docs = _rerank_merge(query, list(docs) + list(new_docs), k)
            grade = grade_retrieval(query, docs)
            logger.info("adaptive: grade(retry %d)=%s", retries + 1, grade)
            retries += 1

        return docs
    except Exception as exc:  # noqa: BLE001 - không bao giờ làm hỏng truy hồi
        logger.warning("adaptive_retrieve lỗi, fallback hybrid_search: %s", exc)
        return hybrid_search(query, document_id, k)
