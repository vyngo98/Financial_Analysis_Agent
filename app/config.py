import os
from dotenv import load_dotenv

load_dotenv()

LLM_PROVIDER = os.getenv("LLM_PROVIDER", "ollama")

OLLAMA_BASE_URL = os.getenv(
    "OLLAMA_BASE_URL",
    "http://localhost:11434"
)

LLM_MODEL = os.getenv(
    "LLM_MODEL",
    "qwen2:7b"
)

EMBEDDING_MODEL = os.getenv(
    "EMBEDDING_MODEL",
    "nomic-embed-text"
)

# Model cho LLM-as-judge trong eval (app/eval/run_eval.py). Mặc định = LLM_MODEL
# (không đổi hành vi cũ). Đặt riêng để GHIM judge cố định khi so sánh nhiều answer-
# model: nhóm câu % chỉ được chấm bằng judge, nếu judge đổi theo answer-model thì so
# sánh vô nghĩa (lẫn thay-đổi-đáp-án với độ-khó-judge) + model tự chấm dễ thiên vị.
JUDGE_LLM_MODEL = os.getenv("JUDGE_LLM_MODEL", LLM_MODEL)

# Số chunk truy hồi mỗi lần search (k). Giữ 5: eval cho thấy k=12 làm câu âm-tính
# (không có trong BCTC) bị nhiễu -> model bịa số; k=5 cho judge/negative tốt nhất.
RAG_TOP_K = int(os.getenv("RAG_TOP_K", "5"))

# Chế độ truy hồi (v4.1 — mặc định hybrid_rerank):
#   "hybrid_rerank" (mặc định) — hợp (dense top-N ∪ BM25 top-M, BM25 lọc stopword)
#                    rồi cross-encoder rerank -> k. Retrieval 81%, numeric 75%.
#   "dense"         — chỉ embedding similarity_search (hành vi cũ v3.3).
#   "adaptive"      — (v6, opt-in) Adaptive RAG Phase 1: Query Analyzer (phân loại
#                    câu + statement_kind) -> hybrid_search -> Retrieval Grader; nếu
#                    ngữ cảnh CHƯA ĐỦ thì rewrite query -> retrieve lại (bounded).
#                    Xem app/rag/adaptive.py. Fallback về hybrid_search khi lỗi LLM.
# k CUỐI CÙNG vẫn = RAG_TOP_K; hybrid+rerank chỉ nâng CHẤT top-k, không tăng số chunk.
RETRIEVAL_MODE = os.getenv("RETRIEVAL_MODE", "hybrid_rerank")
# Cross-encoder reranker (đa ngữ, có tiếng Việt, chạy CPU). Tải 1 lần về HF cache.
RERANK_MODEL = os.getenv("RERANK_MODEL", "BAAI/bge-reranker-v2-m3")
# Số ứng viên lấy trước khi rerank: dense (embedding) + bm25 (keyword).
# 50/30 (v5.2): chunk cần thiết đôi khi nằm ngoài top-25/15 (vd chunk địa chỉ DHG ở dense
# rank 38 / bm25 rank 26) -> pool cũ cắt mất, section-boost không cứu được vì boost chỉ SẮP
# XẾP LẠI pool chứ không THÊM ứng viên. Nới pool để recall đủ; top-k cuối (RAG_TOP_K) vẫn 5.
DENSE_CANDIDATES = int(os.getenv("DENSE_CANDIDATES", "50"))
BM25_CANDIDATES = int(os.getenv("BM25_CANDIDATES", "30"))

# Section-aware boost (v4.6): cross-encoder hay dìm dòng bảng BCTC chính cô đọng xuống
# dưới văn xuôi MD&A/thuyết minh cùng số. Boost cộng điểm cho chunk có statement_kind
# khớp "loại câu hỏi" (balance_sheet/income_statement/cash_flow) sau khi rerank.
#   final = minmax(cross_score) + WEIGHT * (chunk.statement_kind == query_kind)
# WEIGHT=1.0 (đã tune): cross-encoder dìm dòng bảng RẤT mạnh, W nhỏ (0.3) chỉ cứu 4/10
# câu; W=1.0 (chunk khớp-section vượt hẳn nhóm không khớp) cứu 7/10 (đo trên Tesla). Câu
# âm-tính vẫn an toàn vì pool của chúng hầu như không có chunk khớp-section để nhấc.
SECTION_BOOST = os.getenv("SECTION_BOOST", "1") not in ("0", "false", "False", "")
SECTION_BOOST_WEIGHT = float(os.getenv("SECTION_BOOST_WEIGHT", "1.0"))

# Adaptive RAG (v6, chỉ tác dụng khi RETRIEVAL_MODE="adaptive"). Analyzer + Grader +
# rewrite-loop là các LLM call PHỤ (ngoài LLM sinh câu trả lời) chạy trong retrieve().
#   ADAPTIVE_LLM_MODEL  — model cho analyzer/grader/rewriter (mặc định = LLM_MODEL).
#   ADAPTIVE_ANALYZER   — bật Query Analyzer (phân loại + statement_kind lái section-boost).
#                         Tắt -> đi thẳng hybrid_search, section-boost dùng _classify_query cũ.
#   ADAPTIVE_GRADER     — bật Retrieval Grader + vòng rewrite. Tắt -> chỉ analyze rồi retrieve 1 lần.
#   ADAPTIVE_MAX_RETRIES— số lần rewrite->retrieve lại tối đa khi grader nói CHƯA ĐỦ. Giữ 1:
#                         nhiều retry kéo thêm chunk tangential -> nguy cơ model bịa số (câu âm-tính).
ADAPTIVE_LLM_MODEL = os.getenv("ADAPTIVE_LLM_MODEL", LLM_MODEL)
ADAPTIVE_ANALYZER = os.getenv("ADAPTIVE_ANALYZER", "1") not in ("0", "false", "False", "")
ADAPTIVE_GRADER = os.getenv("ADAPTIVE_GRADER", "1") not in ("0", "false", "False", "")
ADAPTIVE_MAX_RETRIES = int(os.getenv("ADAPTIVE_MAX_RETRIES", "1"))
# Phase 2 — Query decomposition + tính % tất định cho câu tỷ suất/tăng trưởng.
# MẶC ĐỊNH TẮT: eval cho thấy độ chính xác retrieval TOÁN HẠNG chưa đủ (vd sub-query "doanh
# thu thuần" bị "lợi nhuận thuần" đè top-1) -> auto-compute ra % SAI một cách tự tin (167,5%
# thay 35,9%) và làm regress câu growth vốn đã đúng nhờ cột kỳ v6.3. Bật lại khi có operand-
# retrieval chính xác hơn hoặc model mạnh hơn (qwen2.5). Guard token-overlap giảm rủi ro khi bật.
ADAPTIVE_DECOMPOSE = os.getenv("ADAPTIVE_DECOMPOSE", "0") not in ("0", "false", "False", "")
# Số ứng viên truy hồi cho MỖI toán hạng (sub-query) khi phân rã câu tính % (decompose).
# Nới từ 3 lên 10: dòng toán hạng đúng đôi khi bị dòng gần-đồng-nghĩa đè (vd "Doanh thu
# thuần" bị "Lợi nhuận thuần" đè, thực ở rank 7). Cổng _operand_matches vẫn chặn dòng sai
# metric nên nới k chỉ tăng recall, không kéo nhầm. hybrid_search rerank toàn pool sẵn nên
# k lớn hơn KHÔNG thêm chi phí tính toán.
OPERAND_CANDIDATES = int(os.getenv("OPERAND_CANDIDATES", "10"))

# Agentic Adaptive RAG (v7, opt-in). Bật -> create_financial_agent() trả về một LangGraph
# StateGraph điều phối route/retrieve/grade/rewrite/decompose/compute thành CÁC NODE tường
# minh (Python điều phối, KHÔNG phụ thuộc LLM gọi tool). Câu tính % dùng con số tính tất
# định làm CHÂN LÝ (LLM chỉ diễn đạt lại). Xem app/agent/graph.py. Tắt (mặc định) -> giữ
# nguyên agent tool-calling cũ (create_agent + middleware repair). Gate để A/B eval.
AGENT_GRAPH = os.getenv("AGENT_GRAPH", "0") not in ("0", "false", "False", "")

QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY", "")
QDRANT_COLLECTION = os.getenv("QDRANT_COLLECTION", "financial_reports")

# Trình trích BẢNG cho trang native (text-layer):
#   "camelot_rows" (MẶC ĐỊNH) — Camelot stream, MỘT chunk NHỎ mỗi dòng dữ liệu + header năm
#                    (retrieval mịn như v0 + sửa đọc-lệch-cột). Sinh chunk dạng `nhãn = giá_trị`
#                    + metadata row_label -> đường compute % TẤT ĐỊNH (agentic graph) đọc được
#                    native docs như VNM. Xem camelot_extract.py. CẦN camelot-py + ghostscript.
#   "pdfplumber"   — find_tables + merge fragment + recover header. Sinh bảng Markdown ATOMIC
#                    (không có `= giá_trị`/row_label) -> compute % KHÔNG chạy. Dùng làm FALLBACK
#                    opt-in khi camelot sập trên PDF khó (vd 10-K lớn kiểu Tesla).
#   "camelot"      — Camelot stream, MỘT chunk/bảng (Code/Note tách, header năm).
# Trang scan luôn đi OCR bất kể cờ này.
TABLE_EXTRACTOR = os.getenv("TABLE_EXTRACTOR", "camelot_rows")
# Camelot stream tuning: edge_tol lớn -> hết cắt cụt.
CAMELOT_EDGE_TOL = int(os.getenv("CAMELOT_EDGE_TOL", "500"))
# row_tol = ngưỡng gộp DÒNG theo trục y. Đặt NHỎ (6) làm mặc định phổ quát:
#   - row_tol lớn (15) gộp nhầm 2 dòng chỉ tiêu độc lập trên bảng KHÍT (Tesla 10-K,
#     line pitch ~13pt) VÀ làm camelot SẬP phát hiện cột ở vài trang (vd Vinamilk p43
#     -> 1 cột, số dính vào nhãn). row_tol nhỏ tránh cả hai: mỗi dòng vật lý = 1 grid
#     row, cột ổn định trên cả layout khít (Tesla) lẫn VAS giãn rộng (Vinamilk).
#   - Nhãn bị xuống dòng (wrapped label của VAS) tách thành dòng label-only, được
#     `_looks_like_continuation`/`_is_label_only` nối lại POST-hoc (xem camelot_extract.py).
CAMELOT_ROW_TOL = int(os.getenv("CAMELOT_ROW_TOL", "6"))
# Chế độ row_tol cho camelot stream:
#   "fixed"    (mặc định) — dùng hằng CAMELOT_ROW_TOL (nhỏ) cho mọi trang.
#   "adaptive" — (THỬ NGHIỆM, đã bị hằng-nhỏ thay thế làm mặc định) dò LINE PITCH từng
#              trang rồi đặt row_tol = round(FACTOR * pitch). Không dùng nữa vì pitch lớn
#              (Vinamilk ~28pt -> tol~17) đẩy vào vùng SẬP CỘT; giữ lại làm tuỳ chọn.
CAMELOT_ROW_TOL_MODE = os.getenv("CAMELOT_ROW_TOL_MODE", "fixed")
CAMELOT_ROW_TOL_FACTOR = float(os.getenv("CAMELOT_ROW_TOL_FACTOR", "0.6"))
CAMELOT_ROW_TOL_MIN = int(os.getenv("CAMELOT_ROW_TOL_MIN", "2"))
CAMELOT_ROW_TOL_MAX = int(os.getenv("CAMELOT_ROW_TOL_MAX", "20"))
# Định dạng chunk cho camelot_rows: "nl" (câu tự nhiên, mặc định) | "markdown".
TABLE_ROW_FORMAT = os.getenv("TABLE_ROW_FORMAT", "nl")

# OCR (scanned PDFs) via Docling + EasyOCR. Deterministic, table-aware, CPU-ok.
OCR_LANG = os.getenv("OCR_LANG", "vi").split(",")
# A page whose real (non-boilerplate) text is shorter than this and that
# carries a full-page image is treated as scanned and routed through OCR.
SCAN_TEXT_THRESHOLD = int(os.getenv("SCAN_TEXT_THRESHOLD", "80"))
# Prepend the section title (+ date/currency line) to each OCR table chunk's
# page_content, mirroring the native path's "## {caption}" + header. Docling puts
# the statement heading (## BẢNG CÂN ĐỐI KẾ TOÁN ...) in a SEPARATE text block, so
# the bare pipe-table chunk otherwise loses all section context -> balance-sheet
# retrieval misses. Toggle off to reproduce the pre-v4.7 bare-table behaviour.
OCR_PREPEND_SECTION_TITLE = os.getenv("OCR_PREPEND_SECTION_TITLE", "1") not in (
    "0", "false", "False", "",
)
# Cách chunk BẢNG cho trang scan (OCR):
#   "atomic" (mặc định) — 1 chunk/bảng (cả bảng Markdown), có thể prepend section-title.
#   "rows"             — 1 chunk NHỎ mỗi DÒNG dữ liệu, dạng câu tự nhiên kèm HEADER năm
#                        căn theo cột (mô phỏng camelot_rows cho native). Retrieval mịn
#                        hơn: dòng bảng chính cạnh tranh sòng phẳng với dòng thuyết minh.
# Row chunk mang metadata statement_kind (phân loại VN theo section-title) để sẵn sàng
# cho section-boost (hiện query-classifier chỉ tiếng Anh nên boost chưa bắn cho câu VN).
# Mặc định "rows": eval DHG cho retrieval 26.9%->61.5%, judge 39.3%->57.1%, bảng cân đối
# 0%->40% so với "atomic" (v4.8-dhg-ocr-rows). Chỉ ảnh hưởng trang scan (OCR).
OCR_TABLE_MODE = os.getenv("OCR_TABLE_MODE", "rows")

UPLOAD_DIR = "data/uploads"
LAST_DOCUMENT_PATH = "data/last_document.json"
# LAST_DOCUMENT_PATH = "/home/dongbui-5070ti/Workspace/Financial_Langchain_Agent/data/last_document.json"
DOCUMENTS_PATH = "data/documents.json"
SESSIONS_DIR = "data/sessions"