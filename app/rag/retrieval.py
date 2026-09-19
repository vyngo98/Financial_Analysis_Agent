"""Truy hồi dùng chung cho tool lẫn eval (v4).

Hai chế độ (config `RETRIEVAL_MODE`):
  - "dense"         : chỉ embedding similarity_search (hành vi cũ).
  - "hybrid_rerank" : hợp (dense top-N ∪ BM25 top-M) -> cross-encoder rerank -> top-k.

Vì sao hybrid + rerank:
  - Dense (bi-encoder) bỏ SÓT dòng khớp theo từ khoá lexical (vd "Payments of
    dividends [Code 36]" không lọt top-50). BM25 kéo nó vào pool.
  - Cross-encoder chấm lại từng cặp (query, chunk) nên đẩy đúng dòng lên top-k,
    sửa các ca gold nằm rank 2-6 hoặc chunk nhãn cụt gây nhiễu.
k CUỐI CÙNG vẫn = k truyền vào; ta chỉ nâng chất top-k chứ không tăng số chunk đưa
cho LLM (tránh lỗi k lớn -> nhiễu -> bịa số đã thấy ở eval).

rank_bm25 chạy IN-MEMORY: scroll toàn bộ chunk của document (1 lần, cache theo
document_id) rồi build BM25 — không cần re-ingest hay đổi schema Qdrant.
"""

import re

from langchain_core.documents import Document

from app.config import (
    BM25_CANDIDATES,
    DENSE_CANDIDATES,
    QDRANT_COLLECTION,
    RERANK_MODEL,
    RETRIEVAL_MODE,
    SECTION_BOOST,
    SECTION_BOOST_WEIGHT,
)
from app.rag.vector_store import (
    build_document_filter,
    get_client,
    get_vector_store,
)

# --------------------------------------------------------------------------- #
# Cache module-level (mỗi tiến trình)
# --------------------------------------------------------------------------- #
_RERANKER = None
_CORPUS_CACHE: dict = {}   # document_id -> list[Document]
_BM25_CACHE: dict = {}     # document_id -> (BM25Okapi, list[Document])


def get_reranker():
    """Lazy singleton CrossEncoder. Tải model 1 lần (giống pattern model OCR)."""
    global _RERANKER
    if _RERANKER is None:
        from sentence_transformers import CrossEncoder

        _RERANKER = CrossEncoder(RERANK_MODEL, max_length=512)
    return _RERANKER


# --------------------------------------------------------------------------- #
# BM25 keyword (in-memory, per document)
# --------------------------------------------------------------------------- #

# \w+ (Unicode, mặc định cho str pattern Python 3) để giữ chữ TIẾNG VIỆT CÓ DẤU.
# [a-z0-9]+ (cũ) băm nát chữ có dấu ("tài sản" -> ['t','i','s','n']) khiến BM25 MÙ hoàn
# toàn với nội dung tiếng Việt -> dòng bảng chính không lọt pool. Với tiếng Anh \w+ ≈ cũ.
_TOK_RE = re.compile(r"\w+")

# Stopword tiếng Anh (function words + từ hỏi). Không lọc -> query như "total assets
# as at 31 December 2023" bị cụm ngày + từ chung chiếm sóng BM25, chôn dòng "TOTAL
# ASSETS [Code 270]" (đã đo: rank 32 -> rank 1 sau khi lọc).
_EN_STOP = frozenset(
    "a an the of in on at as to for and or is are was were be been being by with from "
    "this that these those which what when where who whom how much many did do does had "
    "has have i you we they it its their his her s during within into over under than "
    "then so such also not no all any each more most other some there here".split()
)


def _tok(text: str) -> list[str]:
    """Tokenize lexical: lowercase, giữ chữ+số (kể cả '110', '36', 'kpmg')."""
    return _TOK_RE.findall((text or "").lower())


def _tok_filtered(text: str, stop: frozenset) -> list[str]:
    """Như _tok nhưng bỏ stopword + token toàn chữ số (năm/mã đứng lẻ)."""
    return [w for w in _tok(text) if w not in stop and not w.isdigit()]


def _doc_corpus(document_id: str) -> list[Document]:
    """Scroll toàn bộ chunk của 1 document -> list[Document]. Cache theo id."""
    if document_id in _CORPUS_CACHE:
        return _CORPUS_CACHE[document_id]

    client = get_client()
    flt = build_document_filter(document_id)
    docs: list[Document] = []
    offset = None
    while True:
        points, offset = client.scroll(
            collection_name=QDRANT_COLLECTION,
            scroll_filter=flt,
            limit=256,
            offset=offset,
            with_payload=True,
            with_vectors=False,
        )
        for p in points:
            payload = p.payload or {}
            content = payload.get("page_content") or payload.get("text") or ""
            metadata = payload.get("metadata") or {}
            docs.append(Document(page_content=content, metadata=metadata))
        if offset is None:
            break

    _CORPUS_CACHE[document_id] = docs
    return docs


def _bm25_top(query: str, document_id: str, m: int) -> list[Document]:
    """Top-m Document theo điểm BM25 trên corpus của document."""
    if m <= 0:
        return []
    if document_id in _BM25_CACHE:
        bm25, corpus, stop = _BM25_CACHE[document_id]
    else:
        from collections import Counter

        from rank_bm25 import BM25Okapi

        corpus = _doc_corpus(document_id)
        if not corpus:
            return []
        # Stoplist = stopword tiếng Anh + boilerplate lặp nhiều (document-frequency
        # > 25%: "vinamilk/consolidated/statements/code/vnd/2023..."). Bỏ chúng để
        # BM25 tập trung vào từ khoá phân biệt (total assets, gross profit...).
        raw = [_tok(d.page_content) for d in corpus]
        df = Counter()
        for t in raw:
            for w in set(t):
                df[w] += 1
        cutoff = 0.25 * len(corpus)
        stop = _EN_STOP | {w for w, c in df.items() if c > cutoff}
        bm25 = BM25Okapi([_tok_filtered(d.page_content, stop) for d in corpus])
        _BM25_CACHE[document_id] = (bm25, corpus, stop)

    scores = bm25.get_scores(_tok_filtered(query, stop))
    ranked = sorted(range(len(corpus)), key=lambda i: scores[i], reverse=True)
    return [corpus[i] for i in ranked[:m] if scores[i] > 0]


# --------------------------------------------------------------------------- #
# Hybrid + rerank
# --------------------------------------------------------------------------- #

def _dedup(docs: list[Document]) -> list[Document]:
    """Khử trùng theo nội dung, giữ thứ tự xuất hiện đầu tiên."""
    seen = set()
    out = []
    for d in docs:
        key = d.page_content
        if key in seen:
            continue
        seen.add(key)
        out.append(d)
    return out


# --------------------------------------------------------------------------- #
# Section-aware boost (v4.6): phân loại "loại câu hỏi" theo báo cáo tài chính chính
# --------------------------------------------------------------------------- #

# Thứ tự kiểm: CF -> IS -> BS (cụm "... activities" đặc thù CF nhất; nhiều câu IS/BS
# mới rơi xuống sau). Đã kiểm trên 59 câu ground-truth: 48/48 statement đúng kind,
# 0 câu company_profile/operational bị phân loại nhầm.
# Từ khoá EN + VI. Câu VI trước đây luôn -> None (chỉ có keyword EN) nên section-boost
# không bao giờ bắn cho tài liệu tiếng Việt dù chunk đã có statement_kind. Cụm VI dùng
# thuật ngữ VAS chuẩn (không chép câu hỏi), tránh trùng câu company-profile/âm-tính:
# KHÔNG dùng "cổ phiếu" trần (bảo vệ câu "giá cổ phiếu trên sàn"); "tài sản" KHÁC "tài chính".
_QUERY_KIND_KEYWORDS = (
    ("cash_flow", ("cash flow", "cash used", "cash provided", "operating activities",
                   "investing activities", "financing activities", "capital expenditure",
                   "purchases of", "dividend", "repurchas", "proceeds from",
                   # VI:
                   "lưu chuyển tiền", "dòng tiền", "hoạt động đầu tư",
                   "hoạt động tài chính", "cổ tức", "chi trả cổ tức")),
    ("income_statement", ("revenue", "gross profit", "gross margin", "operating income",
                          "income from operations", "income before", "net income",
                          "net profit", "profit before tax", "profit after tax",
                          "earnings per share", " eps", "research and development", "r&d",
                          "sg&a", "selling", "cost of sales", "cost of goods",
                          "cost of revenue", "operating expense", "expense", "gross",
                          # VI:
                          "doanh thu", "doanh thu thuần", "giá vốn", "lợi nhuận gộp",
                          "lợi nhuận trước thuế", "lợi nhuận sau thuế", "trước thuế",
                          "sau thuế", "biên lợi nhuận", "lãi cơ bản trên cổ phiếu",
                          "chi phí bán hàng",
                          "chi phí quản lý")),
    ("balance_sheet", ("total assets", "current assets", "current liabilit",
                       "total liabilit", "liabilit", "inventor", "equity",
                       "retained earnings", "goodwill", "accounts receivable",
                       "accounts payable", "property, plant", "par value", "stockholder",
                       "intangible", "deferred tax asset", "cash and cash equivalents",
                       "short-term", "share capital", "owners",
                       # VI:
                       "tài sản", "tổng tài sản", "nợ phải trả", "nợ ngắn hạn",
                       "nợ dài hạn", "vốn chủ sở hữu", "vốn góp", "vốn cổ phần",
                       "hàng tồn kho", "tương đương tiền", "phải thu", "phải trả",
                       "đầu tư tài chính")),
    # company_profile: câu hỏi thông tin doanh nghiệp (tên/trụ sở/kiểm toán/kỳ báo cáo).
    # Đặt CUỐI: câu statement khớp CF/IS/BS trước; câu profile thuần không có từ statement.
    ("company_profile", ("công ty nào", "trụ sở", "địa chỉ", "đặt tại",
                         "đơn vị kiểm toán", "năm tài chính kết thúc",
                         "ngành nghề kinh doanh")),
)


def _classify_query(query: str) -> str | None:
    """Đoán loại BCTC câu hỏi nhắm tới -> khớp `statement_kind` của chunk. None nếu không rõ
    (company_profile/operational...) -> KHÔNG boost, giữ nguyên hành vi."""
    s = (query or "").lower()
    for kind, kws in _QUERY_KIND_KEYWORDS:
        if any(w in s for w in kws):
            return kind
    return None


# Sentinel: qkind KHÔNG truyền -> tự phân loại bằng _classify_query (hành vi cũ).
# Truyền qkind=None tường minh -> KHÔNG boost (đã phân loại và ra "không rõ").
_AUTO_QKIND = object()


def hybrid_search(
    query: str,
    document_id: str,
    k: int,
    dense_n: int = None,
    bm25_n: int = None,
    qkind=_AUTO_QKIND,
) -> list[Document]:
    """Hợp dense top-N ∪ BM25 top-M -> cross-encoder rerank (+section boost) -> top-k.

    `qkind`: loại BCTC câu hỏi nhắm tới, dùng cho section-boost. Mặc định (_AUTO_QKIND)
    tự phân loại bằng `_classify_query`. Adaptive RAG truyền statement_kind từ Query
    Analyzer (LLM, hiểu tiếng Việt) để lái boost chuẩn hơn keyword thuần.
    """
    dense_n = DENSE_CANDIDATES if dense_n is None else dense_n
    bm25_n = BM25_CANDIDATES if bm25_n is None else bm25_n

    vector_store = get_vector_store()
    dense = vector_store.similarity_search(
        query=query,
        k=dense_n,
        filter=build_document_filter(document_id),
    )
    sparse = _bm25_top(query, document_id, bm25_n)

    pool = _dedup(list(dense) + list(sparse))
    if not pool:
        return []
    if len(pool) <= k:
        return pool

    reranker = get_reranker()
    scores = [float(s) for s in reranker.predict([(query, d.page_content) for d in pool])]

    # SECTION BOOST: cross-encoder hay dìm dòng bảng BCTC chính cô đọng xuống dưới văn xuôi
    # MD&A/thuyết minh cùng số. Chuẩn hoá điểm về [0,1] rồi CỘNG WEIGHT cho chunk có
    # statement_kind khớp loại câu hỏi -> nhấc nhóm khớp-section lên top-k, giữ thứ tự
    # cross-encoder trong nhóm. Query không phân loại được -> không boost.
    if qkind is _AUTO_QKIND:
        qkind = _classify_query(query) if SECTION_BOOST else None
    elif not SECTION_BOOST:
        qkind = None
    if qkind:
        lo, hi = min(scores), max(scores)
        rng = (hi - lo) or 1.0
        scores = [
            (s - lo) / rng
            + (SECTION_BOOST_WEIGHT if pool[i].metadata.get("statement_kind") == qkind else 0.0)
            for i, s in enumerate(scores)
        ]

    order = sorted(range(len(pool)), key=lambda i: scores[i], reverse=True)
    return [pool[i] for i in order[:k]]


def retrieve(query: str, document_id: str, k: int) -> list[Document]:
    """Điểm truy hồi DUY NHẤT cho tool + eval. Dispatch theo RETRIEVAL_MODE."""
    if RETRIEVAL_MODE == "adaptive":
        # Import trong hàm: adaptive.py import ngược lại hybrid_search/_classify_query.
        from app.rag.adaptive import adaptive_retrieve

        return adaptive_retrieve(query, document_id, k)

    if RETRIEVAL_MODE == "hybrid_rerank":
        return hybrid_search(query, document_id, k)

    vector_store = get_vector_store()
    return vector_store.similarity_search(
        query=query,
        k=k,
        filter=build_document_filter(document_id),
    )
