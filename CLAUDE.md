# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Tổng quan

Agent phân tích báo cáo tài chính (RAG) dựa trên LangChain + Ollama (chạy LLM & embedding cục bộ) + Qdrant làm vector store. Luồng: nạp PDF → chia chunk → nhúng vào Qdrant → agent trả lời câu hỏi bằng cách truy vấn knowledge base qua tool.

## Lệnh thường dùng

```bash
# Cài phụ thuộc
pip install -r requirements.txt

# Trình trích bảng native mặc định là camelot_rows -> cần ghostscript (system dep):
sudo apt-get install -y ghostscript

# Chạy Ollama và kéo model trước khi chạy app (mặc định trong config.py)
ollama pull qwen3:8b
ollama pull nomic-embed-text

# Cần một Qdrant server đang chạy. Local nhanh nhất qua Docker:
docker run -d --name qdrant -p 6333:6333 -p 6334:6334 qdrant/qdrant
# hoặc trỏ tới server thật bằng .env: QDRANT_URL, QDRANT_API_KEY, QDRANT_COLLECTION

# OCR cho PDF scan (Docling + EasyOCR) tự kích hoạt khi phát hiện trang scan.
# Lần chạy OCR đầu tiên sẽ tải model (layout/TableFormer/EasyOCR vi) tự động.

# Chạy web UI (upload PDF, chat, quản lý session) — mở http://localhost:8000
uvicorn app.api:app --reload --port 8000

# CLI (phải chạy như module từ thư mục gốc do dùng import tuyệt đối "app.*")
python -m app.main ingest data/uploads/vinamilk_2023_consolidated_financial_statements.pdf
python -m app.main ask "What was Vinamilk's net revenue in 2023?"
```

Chưa có test, linter, hay build script trong repo.

## Kiến trúc

Import trong toàn bộ code dùng đường dẫn tuyệt đối bắt đầu bằng `app.` (ví dụ `from app.rag.vector_store import get_vector_store`), nên **luôn chạy từ thư mục gốc** với `python -m app.main`, không phải `python app/main.py`.

Ba tầng chính, tất cả chia sẻ chung cấu hình và vector store:

1. **Ingestion** — [app/ingestion/pdf_ingestion.py](app/ingestion/pdf_ingestion.py): `ingest_pdf(file_path, document_id=None)` **nhận diện scan theo từng trang** rồi route: trang text → pdfplumber (native), trang scan → OCR.
   - **Nhận diện scan**: pass 1 gom các dòng lặp trên >30% số trang = boilerplate (header/footer/watermark, vd dòng xác thực chữ ký số) — tổng quát, không hardcode. Trang là **scan** nếu có ảnh lớn và text "thật" (sau khi bỏ boilerplate) < `SCAN_TEXT_THRESHOLD`. Điều này chặn lỗi PDF scan có watermark mỏng tạo ra chunk rác.
   - **Native (text page)** — `_extract_text_page`: **table** (`page.find_tables()` + `table.extract()`) thành chunk Markdown atomic (không qua splitter), prepend caption từ dòng ngay trên bảng (`_find_caption`); **text** narrative lọc bỏ vùng bbox bảng (`page.filter` + `_not_within_bboxes`) rồi chia `RecursiveCharacterTextSplitter` (1000/overlap 200). `source="native"`.
   - **OCR (scan page)** — `_extract_ocr_page` → [app/ingestion/ocr.py](app/ingestion/ocr.py) dùng **Docling** (layout + TableFormer + EasyOCR `vi`) OCR từng trang (`page_range`) ra Markdown, `split_markdown_blocks` tách bảng/narrative. **Deterministic** (không bịa số), chạy CPU. `source="ocr"`.
   - Mọi chunk mang metadata `document_id`, `filename`, `page` (**1-indexed**), `content_type` (`table`/`text`), `source` (`native`/`ocr`). Trả về dict gồm `document_id`, `num_pages`, `num_text_pages`, `num_ocr_pages`, `num_tables`, `num_chunks`. `document_id` là khóa truy vấn lại tài liệu.

2. **RAG / Vector store** — [app/rag/vector_store.py](app/rag/vector_store.py): điểm truy cập duy nhất tới Qdrant (`QdrantVectorStore` + `QdrantClient`). Collection mặc định `"financial_reports"` (cấu hình qua `QDRANT_COLLECTION`), kết nối server qua `QDRANT_URL`/`QDRANT_API_KEY`. `get_vector_store()` tự tạo collection nếu chưa có (dò số chiều vector động qua `embed_query`, distance COSINE) + payload index cho `metadata.document_id`. Helper `build_document_filter(document_id)` trả về Qdrant `Filter` trên **`metadata.document_id`** (QdrantVectorStore lưu metadata dưới key `metadata`). Mọi tầng khác gọi `get_vector_store()` / `build_document_filter()` thay vì đụng Qdrant trực tiếp.

4. **Web API + Frontend** — [app/api.py](app/api.py) là FastAPI app: tạo agent **một lần** rồi tái dùng (document chọn theo request qua `config`). Endpoints `/api/documents` (list + upload), `/api/sessions` (CRUD + `/messages` để chat). Upload chạy `ingest_pdf` trong **BackgroundTasks** (OCR scan có thể mất nhiều phút): document được đăng ký ngay với `status="processing"`, xong thì `update_document(status="ready", ...counts)` hoặc `"failed"`; frontend poll `/api/documents` và chỉ cho tạo session khi `ready`. Chat truyền toàn bộ lịch sử session cho agent (hỏi đa lượt) và inject `document_id` của session qua `config`. [app/frontend/](app/frontend/) là SPA HTML/CSS/JS thuần (3 khu: Documents / Chat / Sessions), phục vụ tại `/` và `/static`. [app/storage.py](app/storage.py) lưu registry đa tài liệu (`data/documents.json`) và session (`data/sessions/<id>.json`) bằng JSON — tách biệt với Qdrant (chỉ giữ metadata + lịch sử hội thoại).

3. **Agent + Tools** — [app/agent/financial_agent.py](app/agent/financial_agent.py) tạo agent qua `create_agent` (LangChain) với `ChatOllama`, `temperature=0`, và một tool duy nhất. Tool [app/tools/rag_tool.py](app/tools/rag_tool.py) `search_pdf_knowledge_base(query, config)` chạy `similarity_search` (k=5) **có filter theo `document_id`** để cô lập tài liệu. Mỗi kết quả gắn nhãn `[TABLE]`/`[TEXT]` theo metadata `content_type` để LLM biết đang đọc bảng Markdown. System prompt buộc agent chỉ trả lời dựa trên kết quả tool và không bịa số liệu.

### Cơ chế chọn `document_id` (LLM không thấy tham số này)

`document_id` là **ngữ cảnh phiên**, không phải thứ LLM điền. Tool nhận nó qua `RunnableConfig` (`config["configurable"]["document_id"]`) — tham số kiểu `RunnableConfig` được LangChain tự inject, không nằm trong schema mà LLM thấy. Nếu config không có, tool fallback về `get_latest_document_id()` đọc `data/last_document.json` (`LAST_DOCUMENT_PATH`). File này được `ingest_pdf` ghi qua `save_last_document()` sau mỗi lần nạp, nên phiên query (tách khỏi phiên ingest) tự biết tài liệu mới nhất mà user không cần truyền id. Nếu chưa nạp gì, tool trả thông báo yêu cầu upload.

## Cấu hình

[app/config.py](app/config.py) đọc biến môi trường qua `python-dotenv` (tạo file `.env` ở gốc để ghi đè). Các khóa: `LLM_PROVIDER`, `OLLAMA_BASE_URL`, `LLM_MODEL`, `EMBEDDING_MODEL` (mặc định `nomic-embed-text`); Qdrant: `QDRANT_URL` (mặc định `http://localhost:6333`), `QDRANT_API_KEY` (rỗng cho local), `QDRANT_COLLECTION` (mặc định `financial_reports`); OCR: `OCR_LANG` (mặc định `vi`, phân tách bằng dấu phẩy), `SCAN_TEXT_THRESHOLD` (ngưỡng ký tự để coi trang là scan). `UPLOAD_DIR`, `LAST_DOCUMENT_PATH`, `DOCUMENTS_PATH`, `SESSIONS_DIR` đang hardcode.

**Truy hồi** (`retrieve()` dispatch theo `RETRIEVAL_MODE`): `"hybrid_rerank"` (mặc định), `"dense"`, hoặc `"adaptive"`. Chế độ `adaptive` (Adaptive RAG Phase 1, [app/rag/adaptive.py](app/rag/adaptive.py)) chạy **Query Analyzer** (phân loại câu + `statement_kind` để lái section-boost, hiểu tiếng Việt tốt hơn keyword `_classify_query`) → `hybrid_search` → **Retrieval Grader**; nếu ngữ cảnh chưa đủ thì **rewrite query → truy hồi lại** (bounded `ADAPTIVE_MAX_RETRIES`, mặc định 1) rồi rerank-merge giữ nguyên `k`. Đây là các LLM call PHỤ (ngoài LLM sinh đáp án), bật/tắt qua `ADAPTIVE_ANALYZER`/`ADAPTIVE_GRADER`, model qua `ADAPTIVE_LLM_MODEL` (mặc định = `LLM_MODEL`). Mọi lỗi LLM/parse → fallback về `hybrid_search` thuần (không làm hỏng truy hồi). Opt-in: `RETRIEVAL_MODE=adaptive`.
