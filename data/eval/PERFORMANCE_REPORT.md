# Báo cáo đánh giá hiệu năng hệ thống RAG tài chính

Tài liệu sống — mỗi lần cải thiện hệ thống thêm **một mục phiên bản mới** vào đầu
phần "Lịch sử phiên bản", giữ nguyên các mục cũ để so sánh.

- Cách sinh số liệu: `python -m app.eval.run_eval -g data/eval/qa_ground_truth_vnm.json --version <vX> --note "<mô tả thay đổi>"`
- Report thô mỗi lần chạy tự lưu ở `data/eval/results/eval_report_<version>_<timestamp>.{json,md}`.

## Các chỉ số được chấm

| Chỉ số | Ý nghĩa | Cách tính |
|---|---|---|
| **Retrieval hit-rate@k** | Trong `k` chunk truy hồi có trúng `expected_pages` không | số câu hit / số câu có expected_pages |
| **mean recall@k** | Tỉ lệ trang mong đợi được truy hồi (trung bình) | trung bình \|expected ∩ retrieved\| / \|expected\| |
| **Numeric accuracy** | Câu trả lời chứa **đúng con số chính** (chuẩn hoá dấu `. , `, bỏ năm) | số câu khớp / số câu có số tài chính |
| **Judge accuracy** | LLM-as-judge chấm đúng/sai so ground_truth (chấp nhận ±1đ% cho câu suy luận) | số câu đúng / số câu chấm |
| **Âm-tính** | Câu bẫy: hệ thống nói "không tìm thấy", KHÔNG bịa số | số câu xử lý đúng / số câu âm-tính |

---

## Lịch sử phiên bản

### `v7.5-currency` ⭐⭐ — Sửa lỗi ĐƠN VỊ TIỀN (VND-vs-USD) + scale "million" bằng metadata tất định (2026-09-19)

Eval v7.4 lộ nhóm lỗi chi phối tài liệu USD: khâu generate **gắn "VND" cho tài liệu USD** (Apple sai
9/12 câu, Tesla 5/10) — model trả ĐÚNG số nhưng sai đơn vị tiền. **Nguồn:** `SYSTEM_PROMPT` Rule 11
hardcode *"followed by 'VND'"* ([app/agent/financial_agent.py](../../app/agent/financial_agent.py));
graph kế thừa qua `_GEN_SYSTEM_PROMPT`. Không có detection/metadata currency nào; doc USD lại bị strip
hết `$`/`USD`/`million` khi trích bảng.

**Fix (detect tất định ở INGEST, doc-level):**
1. `_detect_currency_scale(text)` ([app/ingestion/pdf_ingestion.py](../../app/ingestion/pdf_ingestion.py))
   chạy trên **RAW text** (native `page.extract_text()` + OCR markdown — nơi `$`/`USD`/`in millions`/`VND`
   CÒN NGUYÊN) → `(currency, unit_scale)`. Tín hiệu **dương cho cả 2 currency** (không suy đoán). Stamp
   `metadata["currency"]` + `["unit_scale"]` vào MỌI chunk (1 vòng post-process trong `ingest_pdf`).
2. Generate đọc currency/scale từ metadata rồi chèn dòng `ĐƠN VỊ TIỀN…` vào prompt
   ([app/agent/graph.py](../../app/agent/graph.py) `_doc_currency_scale`/`_currency_directive`); fallback
   detect trên page_content cho chunk cũ.
3. Rule 11/12/15 trung tính currency (USD→"43,009 million USD"; VND→"…VND"); EPS/tỷ lệ giữ đơn vị dòng.
   Số GIỮ NGUYÊN (scale chỉ là từ đơn vị, không nhân/chia). Parity ở
   [app/tools/rag_tool.py](../../app/tools/rag_tool.py) cho path tool-calling.

Detect xác nhận: Apple=(USD,million), Tesla=(USD,million), VNM/SAB/DHG/GAS=(VND,"").

#### BẢNG ĐẦY ĐỦ — v7.5-currency (`AGENT_GRAPH=1`, qwen2:7b, re-ingest 6 doc)

| Tài liệu | Judge (v7.4→**v7.5**) | Retrieval | Numeric | Compute | Âm-tính |
|---|---|---|---|---|---|
| **DHG** (scan) | 96.4% → **96.4%** (27/28) | 96.2% | 100% | 2/2 | 2/2 |
| **SAB** (scan) | 93.1% → **89.7%** (26/29) | 81.5% | 71.4% | 2/2 | 1/2 |
| **VNM** (native) | 92.9% → **89.3%** (25/28) | 84.6% | 80.0% | 2/2 | 1/2 |
| **GAS** (scan) | 89.3% → **89.3%** (25/28) | 84.6% | 90.5% | 2/2 | 1/2 |
| **TSLA** (native) | 69.7% → **78.8%** (26/33) ▲+9.1 | 61.3% | 75.0% | 2/3 | 2/2 |
| **AAPL** (native) | 63.6% → **69.7%** (23/33) ▲+6.1 | 96.8% | 76.0% | 1/3 | 1/2 |
| **TỔNG** | 83.2% → **84.9%** (152/179) ▲+1.7 | — | — | 11/14 | 8/12 |

#### Kết quả & kiểm hồi quy (ràng buộc chính)
- **USD thắng rõ:** Tesla **+9.1pp**, Apple **+6.1pp**. Đáp án nay ghi "… million USD" thay vì "VND"
  — Apple `cash_flow` 50→83.3%, `balance_sheet` 75→87.5%; Tesla `balance_sheet` 33→… tăng đều.
- **KHÔNG hồi quy lỗi currency ở doc VN:** detect VND đúng cho cả 4 doc VN, đáp án vẫn ghi "VND"
  (không gắn "million" thừa). DHG/GAS giữ y nguyên.
- **VNM −3.6 / SAB −3.4 là NHIỄU qwen2, KHÔNG phải currency:** các câu tụt đều thuộc lớp khác —
  VNM Q9 (retrieval-miss), Q24 (thiếu năm), Q28 (negative sai năm); SAB Q8/Q15 (đọc nhầm số), Q28
  (negative bịa). Thêm 1 dòng directive làm model nhỏ đảo ~1 câu/doc (đã đo: rút gọn directive lại
  đảo câu khác — whack-a-mole của qwen2:7b, không tách được bằng câu chữ prompt). Currency của VN
  còn nguyên.
- Net tổng **+1.7pp**. Nút thắt còn lại của USD **không còn là currency** mà là **đọc nhầm dòng/số**
  trong bảng dày (Apple income_statement, Tesla) + EPS + camelot sập 10-K — điểm yếu qwen2:7b.

### `v7.4` ⭐⭐⭐ — Native PDF compute + operand-precision + bộ test GAS (BẢNG ĐẦY ĐỦ 6 tài liệu) (2026-09-19)

Gói 3 thay đổi để **native text-layer PDF (VNM/Apple/Tesla) tính được % như doc OCR**, cộng bộ test mới GAS:

1. **Flip mặc định `TABLE_EXTRACTOR=pdfplumber → camelot_rows`** ([app/config.py](../../app/config.py)): trang
   native giờ sinh chunk **1-dòng-1-chunk** dạng `nhãn = giá_trị` + metadata `row_label` (thay bảng
   Markdown atomic không có `= value`/`row_label` mà compute_node không đọc được). Giữ `pdfplumber`
   opt-in làm van an toàn (camelot có thể sập cột trên 10-K lớn). Cần `camelot-py` + ghostscript
   (đã thêm requirements.txt + CLAUDE.md).
2. **`_period_values` nhận SỐ DẤU PHẨY** ([app/rag/adaptive.py](../../app/rag/adaptive.py) L309): regex đổi
   `[\d.]` → `\d[\d.,]{4,}` và strip cả `.`/`,`. Đây là blocker ĐỘC LẬP: doc tiếng Anh (VNM/AAPL/TSLA)
   dùng dấu phẩy `24,544,731,...`, regex cũ chỉ đọc dấu chấm VAS → trả rỗng → abstain. **Chính vì thế
   OCR-VN (dấu chấm) chạy mà native-EN (dấu phẩy) không.**
3. **Operand-precision segment-vs-total** ([app/agent/graph.py](../../app/agent/graph.py) `_match_operand_in_pool`):
   trong các dòng khớp token metric, ưu tiên **(1) nhãn ít TOKEN THỪA nhất** ngoài `req` (dòng tổng
   "Net revenue"/"Total gross profit" thắng "Segment gross profit") → **(2) ít period-value hơn** (né
   breakdown segment 4-6 cột) → **(3) rank retrieval**. Sửa VNM Q25 gross margin **48,5% → 40,7%** (chọn
   nhầm net-revenue-segment1 p65 → total p10). Bẫy: heuristic "ít value nhất" ĐƠN THUẦN sai (dòng
   segment energy 1-value của Tesla lọt trên total 3-value) → token-thừa phải là khoá CHÍNH.

**Bộ test MỚI: `qa_ground_truth_gas.json`** — PV GAS BCTC hợp nhất kiểm toán 2025 (scan/OCR), 28 câu, số
đối chiếu TRỰC TIẾP từ ảnh PDF gốc (10 cân đối / 8 KQKD / 3 LCTT / 3 công ty / 2 suy luận / 2 âm-tính).

#### BẢNG HIỆU NĂNG ĐẦY ĐỦ — v7.4 (`AGENT_GRAPH=1`, qwen2:7b) trên TẤT CẢ tài liệu đã eval

| Tài liệu | Loại trích | Judge | Retrieval@k | Numeric | Compute (%) | Âm-tính |
|---|---|---|---|---|---|---|
| **DHG** (Dược Hậu Giang) | scan/OCR-rows | **96.4%** (27/28) | 96.2% | 100% | **100%** (2/2) | 100% (2/2) |
| **SAB** (Sabeco) | scan/OCR-rows | **93.1%** (27/29) | 81.5% | 81.0% | **100%** (2/2) | 50% (1/2) |
| **VNM** (Vinamilk) | native/camelot_rows | **92.9%** (26/28) | 84.6% | 80.0% | **100%** (2/2) | 50% (1/2) |
| **GAS** (PV GAS) ⟵ mới | scan/OCR-rows | **89.3%** (25/28) | 84.6% | 95.2% | **100%** (2/2) | 50% (1/2) |
| **TSLA** (Tesla 10-K) | native/camelot_rows | **69.7%** (23/33) | 61.3% | 75.0% | 66.7% (2/3) | 100% (2/2) |
| **AAPL** (Apple) | native/camelot_rows | **63.6%** (21/33) | 96.8% | 84.0% | 33.3% (1/3) | 50% (1/2) |
| **TỔNG** (6 doc) | — | **83.2%** (149/179) | — | — | 78.6% (11/14) | 66.7% (8/12) |

Ghi chú: Compute (%) = category suy_luan/reasoning; doc VN đạt **8/8 = 100%**, doc EN 3/6 (điểm yếu
decompose EN + currency). Âm-tính = các câu negative_not_found (2 câu/doc × 6 = 12).

#### Kết quả then chốt
- **Compute % native ĐÃ THÔNG:** VNM 0→**100%** (gross margin đúng tất định 40,7%). Doc VN (VNM/SAB/DHG/GAS)
  compute **8/8 = 100%**.
- **GAS lần đầu = 89.3%**, compute 100%, bang_can_doi/ket_qua 100% — fix khái quát tốt cho doc scan MỚI.
- **Không hồi quy:** flip camelot_rows **retrieval-neutral** trên AAPL/TSLA (y hệt baseline camelot cũ:
  AAPL 96.8%, TSLA 61.3%); operand-precision **không đụng** SAB/DHG (giữ 93.1%/96.4%). TSLA judge còn
  TĂNG 48.5% (v5.4 tool-calling) → 66.7% (v7.3) → **69.7%** (v7.4).

#### Điểm yếu còn lại (độc lập, ngoài phạm vi gói này)
- **Compute EN chưa tất định:** VNM Q26 / TSLA-AAPL reasoning — `decompose_metric` (LLM) dịch "net profit
  after tax"→"net income" LỆCH nhãn VAS/tài liệu ("Net profit") → `_match_operand_in_pool` trả None →
  rơi về LLM tự nhẩm (VNM Q26 4,54% vs 5,15%, judge tha ±1pp). Cần map thuật ngữ EN↔nhãn tài liệu.
- **AAPL judge thấp (63.6%):** lỗi **currency VND-vs-USD** trong generate (model trả VND cho doc USD) —
  có sẵn từ trước, retrieval AAPL vẫn 96.8%.
- **TSLA income_statement retr 10%:** camelot **sập cột trên 10-K lớn** (130 trang) — giới hạn đã biết.
- **Âm-tính:** VNM Q28 / SAB Q28 / GAS Q27 / AAPL bịa số câu bẫy — điểm yếu qwen2:7b.

Cách tái lập mỗi doc: `AGENT_GRAPH=1 python -m app.eval.run_eval -g data/eval/qa_ground_truth_<doc>.json --version v7.4-<doc>`
(doc native re-ingest camelot_rows lần đầu; sau đó `--skip-ingest`). Report thô ở `data/eval/results/eval_report_v7.4-*`.

### `v7.2-operand-recall` ⭐⭐ — Sửa recall TOÁN HẠNG câu tỷ suất (calc VN 4/4 = 100%) (2026-09-15)

Vá lỗi cuối của v7.0 (SAB Q26 biên LN gộp abstain). Trace cho thấy 2 nguyên nhân trong
`decompose_node` ([app/agent/graph.py](../../app/agent/graph.py)): **① tử số** bị `_operand_matches`
loại vì LLM chèn tên công ty ("...của Sabeco/Dược Hậu Giang") — token tên riêng không có trong nhãn
chỉ tiêu; **③ mẫu số** ("Doanh thu thuần") rớt khỏi pool k=3 (bị "Lợi nhuận thuần" đè, thực ở rank 7).

Fix (giữ qwen2:7b): (A) nới sub-query mỗi toán hạng `3 → OPERAND_CANDIDATES=10`
([app/config.py](../../app/config.py)); `hybrid_search` rerank toàn pool sẵn nên **chi phí 0**.
(B) Thay khớp toán hạng bằng `_match_operand_in_pool` **TỔNG QUÁT**: chỉ yêu cầu khớp token của mô tả
mà THỰC SỰ xuất hiện trong nhãn các dòng-CÓ-SỐ của pool → token tên riêng tự động bị bỏ, token metric
(doanh/thu/thuần/lợi/nhuận/gộp) được giữ. **Không hardcode danh sách công ty** (v7.1 từng thử stop-list
"sabeco/dhg..." → brittle, làm **DHG Q25 lật BAD** vì thiếu "dược/hậu/giang" → bỏ cách đó). Chỉ nhận
chunk có `_first_value` ≠ None (bỏ chunk khớp token nhưng là văn xuôi). (C) Prompt decompose cấm chèn
tên riêng (phòng vệ tầng 2). Opt-in sau `AGENT_GRAPH=1`.

Eval `AGENT_GRAPH=1 --skip-ingest` (SAB + DHG, qwen2:7b) so v7.0:

| Tài liệu | Chỉ số | v7.0 | **v7.2** |
|---|---|---|---|
| **SAB** | Judge | 89.7% | **93.1%** (27/29) |
| | `suy_luan` (%) | 50% | **100%** (2/2) |
| **DHG** | Judge | 96.4% | **96.4%** (27/28) |
| | `suy_luan` (%) | 100% | **100%** (2/2) |
| **Tổng** | Judge | 93.0% | **94.7%** (54/57, cao nhất) |
| | `suy_luan` (%) | 75% (3/4) | **100%** (4/4) |

**Cả 4 câu calc VN ra con số CHÍNH XÁC tuyệt đối** (compute tất định): SAB 35,9% & giảm 18,8%;
DHG 47,6% & tăng 9,4% — đều khớp ground-truth. **Không hồi quy**: mọi category factual + âm-tính giữ
nguyên (SAB judge +3.4pp nhờ đúng Q26; DHG phục hồi từ v7.1). Còn lại chỉ: SAB Q16 (đọc nhầm dòng
"doanh thu bán bia" — số đúng ở rank1 context), SAB Q28 (bịa giá cổ phiếu câu âm-tính) — cả hai là
điểm yếu qwen2:7b, ngoài phạm vi fix này.

**Ngoài phạm vi:** doc TIẾNG ANH (bảng Markdown native) vẫn cần bộ trích `_period_values` cho định
dạng `| … |` → calc EN còn abstain (an toàn). `v7.1-operand-recall` bị thay thế bởi bản này (đã có
hồi quy DHG do stop-list brittle).

### `v7.0-agentic-graph` ⭐⭐ — Agentic Adaptive RAG bằng LangGraph (compute % tất định là chân lý) (2026-09-14)

**Đổi KIẾN TRÚC agent** (không đổi model — vẫn qwen2:7b, `.env` không override → cô lập biến sạch). Thay
agent tool-calling bằng một **LangGraph StateGraph** điều phối `route → {retrieve→grade→rewrite loop |
decompose→compute} → generate` thành CÁC NODE tường minh. Python điều phối, **KHÔNG phụ thuộc LLM gọi
tool** (né nút thắt qwen2:7b chập chờn/không gọi calculator). Câu tính % dùng **con số tính tất định làm
CHÂN LÝ** (LLM chỉ diễn đạt lại, post-check giữ nguyên số); operand không verify được → **ABSTAIN** (rơi về
factual, không bịa); guard ratio ∈ [0,100]. Opt-in qua cờ `AGENT_GRAPH=1`, giữ đường cũ để A/B. Code:
[app/agent/graph.py](../../app/agent/graph.py) (mới), dispatch ở
[app/agent/financial_agent.py](../../app/agent/financial_agent.py), cờ ở
[app/config.py](../../app/config.py). Reuse helper analyzer/grader/rewriter/decompose/compute trong
[app/rag/adaptive.py](../../app/rag/adaptive.py) (chỉ re-wire) + fix `_DECOMPOSE_SYSTEM` sinh operand
ĐÚNG ngôn ngữ câu hỏi (Anh→Anh, Việt→Việt).

Eval `AGENT_GRAPH=1 --skip-ingest` (SAB + DHG, qwen2:7b) so baseline production v6.3:

| Tài liệu | Chỉ số | v6.3 | **v7.0** | Δ |
|---|---|---|---|---|
| **SAB** | Judge | 69.0% | **89.7%** | **+20.7pp** (cao nhất từ trước) |
| | Numeric | 71.4% | **81.0%** | +9.5pp |
| | `bang_can_doi` / `ket_qua` / `luu_chuyen` judge | 80% / 62.5% / 66.7% | **100% / 87.5% / 100%** | tăng đều |
| | `suy_luan` (%) | 0% | **50%** (1/2) | +50pp |
| | Retrieval / Âm-tính | 81.5% / 50% | 81.5% / 50% | = |
| **DHG** | Judge | 96.4% | **96.4%** | = (giữ đỉnh) |
| | `suy_luan` (%) | 50% | **100%** (2/2) | +50pp |
| | Numeric / Âm-tính | 100% / 100% | 100% / 100% | = |

**Thắng rõ, không hồi quy đâu cả.** Hai đòn bẩy: (1) **factual tăng mạnh** vì graph LUÔN truy hồi rồi nạp
context — bỏ phụ thuộc qwen2:7b tự gọi search tool (trước hay bỏ qua/rò tool-call ra text). (2) **calc: 3/4
câu VN đúng với con số CHÍNH XÁC tuyệt đối** nhờ compute tất định: DHG margin **47,6%**, DHG growth **9,4%**,
SAB growth **18,8%** — đều khớp ground-truth. Đây là lần đầu nhóm % thắng **nhất quán** (v6.4/v6.5 hint bắn
chập chờn vì LLM tự chọn tin; nay con số là chân lý, LLM không đổi được).

**Còn lại:** SAB Q26 biên LN gộp — compute **abstain** (không verify được operand "lợi nhuận gộp"/"doanh thu
thuần" trong chunk SAB) → trả số LN gộp thô thay vì % (an toàn: không bịa %, nhưng vẫn sai). Lớp lỗi
operand-precision cũ [[retrieval-granularity-bottleneck]].

**Phạm vi:** mới eval 2 doc TIẾNG VIỆT (nơi chunk dạng `= số` OCR/rows hợp với bộ trích `_period_values`).
Doc TIẾNG ANH (aapl/tsla/vnm) dùng **bảng Markdown native** → `_period_values` chưa trích được → calc EN
**abstain an toàn → factual** (chưa cải thiện, không hồi quy). Đòn bẩy tiếp: **bộ trích bảng Markdown** cho
compute (để calc EN chạy) + lọc dòng segment/sub-item; điều tra operand SAB Q26.

**Trạng thái:** cờ `AGENT_GRAPH` **mặc định TẮT** (production = tool-calling cũ), bật `=1` để dùng graph.

### `v6.5-decomp-fix` — Sửa reorder decompose (full-query nền + augment) — VẪN MẶC ĐỊNH TẮT (2026-09-13)

Sửa lỗi thiết kế v6.4 (sub-query chiếm slot đầu, đảo thứ tự full-query → phá câu growth). Đổi
`decompose_retrieve`: **full-query làm nền giữ thứ tự** + **augment-if-missing (append cuối)**; `_operand_matches`
đổi sang **khớp TOÀN BỘ token nguyên vẹn**; `_compute_hint` growth chịu giá trị ở 1 hoặc 2 chunk. Kiểm chạy
lẻ: Q27 growth → hint "giảm 18,8%" đúng, thứ tự full-query giữ nguyên.

Eval `ADAPTIVE_DECOMPOSE=1 --skip-ingest` (so v6.3/v6.4): reasoning **KHÔNG cải thiện net** — SAB suy_luan
0% (Q25 margin thắng lẻ nhưng eval Q26/Q27 BAD), DHG 50% (**Q25 margin OK** — thắng nhất quán), VNM reasoning
0%. SAB numeric 71.4→81.0% nhưng judge phẳng 69.0%; DHG 96.4%; VNM 82.1%. Âm-tính tổng 83.3% (VNM Q28 bịa
"revenue target" — câu KHÔNG có "%", KHÔNG qua decompose → **nhiễu thuần, không do Phase 2**).

**Kết luận (giữ TẮT, dừng tinh chỉnh):** cơ chế hint tính-sẵn **bắn KHÔNG nhất quán** vì `decompose_metric`
(LLM) không tất định — mô tả part lúc ngắn lúc dài → guard all-token lúc khớp lúc trượt (Q27 chạy lẻ ra hint
đúng nhưng trong eval model lại "không tìm thấy"). Guard lỏng thì lấy nhầm toán hạng (v6.4: 167,5%), chặt thì
trượt chunk đúng (v6.5) — whack-a-mole. Thắng nhất quán duy nhất: **DHG margin** (toán hạng sạch). Không đủ
để bật mặc định. **Nút thắt thật = model**: qwen2:7b (a) không gọi calculator, (b) LLM-decomposer/judge nhiễu.
→ Đòn bẩy tiếp: **qwen2.5:7b-instruct** (giải cả tool-calling lẫn số học lẫn ổn định sub-call). Feature giữ
sau cờ `ADAPTIVE_DECOMPOSE=0`, bật lại khi đổi model.

### `v6.4-decomp-calc` — Phase 2: Query decomposition + calculator (câu %) — MẶC ĐỊNH TẮT (2026-09-12)

Thử **Phase 2** cho câu tỷ suất/tăng trưởng %: Query decomposition (tách sub-query lấy đủ tử+mẫu) +
calculator tất định. Vì qwen2:7b **không chịu gọi tool** calculate (đo thực: bỏ qua tool, tự nhẩm sai),
chuyển sang **tính sẵn server-side** rồi chèn dòng `[KẾT QUẢ TÍNH SẴN]` vào context (LLM chỉ đọc lại), kèm
**guard bigram** chống lấy nhầm toán hạng (vd sub-query "doanh thu thuần" bị "lợi nhuận thuần" đè top-1 →
suýt ra 167,5% thay 35,9%). Gate bằng `ADAPTIVE_DECOMPOSE` (**mặc định 0/TẮT**). Code:
[app/rag/adaptive.py](../../app/rag/adaptive.py) (`decompose_metric`/`decompose_retrieve`/`_compute_hint`),
[app/tools/calc_tool.py](../../app/tools/calc_tool.py), wiring ở rag_tool + financial_agent (rule 16).

Eval **ADAPTIVE_DECOMPOSE=1** (thí nghiệm, `--skip-ingest`) vs baseline TẮT (SAB v6.3 / DHG v6.3 / VNM v6.2):

| Tài liệu | Judge (OFF→ON) | Âm-tính (OFF→ON) | reasoning/% judge |
|---|---|---|---|
| SAB | 69.0% → 72.4% | 50% → 100% | 50% (Q26 margin OK, Q27 growth HỎNG) |
| DHG | 92.9% → 96.4% | 100% → 100% | 50% |
| VNM | 85.7% → **82.1%** | 100% → **50%** | 50% |

**Kết luận (âm tính — giữ TẮT):** nhóm % **KHÔNG cải thiện net** (~50% cả hai chế độ) — decomposition sửa
câu tỷ suất (SAB Q26) nhưng phá câu growth (SAB Q27) vốn đã đúng nhờ cột kỳ v6.3. Tệ hơn, BẬT gây **regress
VNM** (judge −3.6pp, **âm-tính 100%→50%**: câu âm-tính có chữ "%" bị kéo vào đường tính → bịa số). Dao động
SAB/DHG phần lớn là nhiễu qwen2 (âm-tính SAB 50→100 không do decomposition). Nút thắt thật: (1) qwen2 không
gọi tool; (2) retrieval TOÁN HẠNG chưa đủ chính xác cho thuật ngữ VAS gần giống + OCR doanh-thu-thuần lỗi.

**Trạng thái**: tính năng đã có, **guard an toàn**, **mặc định TẮT** (hành vi production = v6.3, không hồi
quy). Bật lại (`ADAPTIVE_DECOMPOSE=1`) khi có **model mạnh hơn (qwen2.5)** hoặc **operand-retrieval chính
xác hơn** (khớp mã số/nhãn chính xác). Calculator tool tất định đã sẵn cho model biết gọi tool.

### `v6.3-cf-header` ⭐ — Header-recovery cột kỳ cho chunk OCR (vá LCTT judge 0%) (2026-09-12)

Vá **model đọc sai cột kỳ bảng lưu chuyển tiền tệ** (SAB `luu_chuyen` judge 0% dù retrieval 100%).
Gốc rễ ([app/ingestion/pdf_ingestion.py](../../app/ingestion/pdf_ingestion.py) `_ocr_row_documents`): lấy `header = rows[0]` cứng. Bảng LCTT
3 cột có header năm **tách 2 dòng** (`['Mã Thuyết','2025','2024']`+`['số minh','VND','VND']`) và bị Docling
**tách sang table-block khác** với dữ liệu → block dữ liệu có `rows[0]` là một **dòng số** → nhãn cột =
"5.651.965.788.427" (số rác) thay vì "2025". Model không biết cột nào là năm nay → đọc sai.

Sửa: **header-recovery** — chỉ dòng header THẬT (không phải data, có marker năm/VND/mã/thuyết minh) mới
làm header, gộp header đa dòng; **không bao giờ dùng dòng số làm header**; fallback nhãn cột theo VỊ TRÍ +
loại báo cáo (`_positional_period`): BS → "số cuối/đầu năm", KQKD/LCTT → **"năm nay/năm trước"** (cột trái
= kỳ hiện tại). Giữ nguyên merge nhãn tổng + `[Mã số]` của v6.2. Kiểm chứng OCR thật tr.12/13:
mã 20 HĐKD "năm nay = 3.898.884.563.124", mã 40 HĐTC "(6.787.749.735.923)", mã 36 cổ tức "(6.512.668.096.116)".

| Tài liệu | Chỉ số | v6.2 | **v6.3** | Δ |
|---|---|---|---|---|
| **SAB** (scan) | Numeric | 61.9% | **71.4%** | **+9.5pp** |
| | Judge | 58.6% | **69.0%** | **+10.4pp** |
| | `luu_chuyen` judge | **0%** | **66.7%** | **+66.7pp** |
| | Retrieval / Âm-tính | 81.5% / 50% | 81.5% / 50% | = |
| **DHG** (scan) | Numeric | 100% | 100% | = |
| | Judge | 96.4% | 92.9% | −3.6pp (1 câu `suy_luan`, nhiễu) |
| | Retrieval / Âm-tính | 96.2% / 100% | 96.2% / 100% | = |

**SAB judge 69.0% — cao nhất từ trước tới nay** (55.2% baseline → 69.0%). `luu_chuyen` **0%→66.7%** (Q23
HĐKD, Q24 HĐTC OK; Q25 cổ tức còn sai). `bang_can_doi` 80% & `ket_qua` 62.5% **không hồi quy**. DHG mọi
nhóm BCTC vẫn 100%, chỉ `suy_luan` −1 câu (reasoning, không liên quan nhãn cột → nhiễu qwen2). Âm-tính giữ.
`dump_failures` bắn cho SAB (9 câu, vượt best 62.1%).

**Còn lại (SAB)**: Q5 "TỔNG TÀISẢN" retrieval MISS (nhãn OCR dính chữ); Q8 tiền [110], Q15/17/18 KQKD, Q25
cổ tức, Q26 suy_luan — chủ yếu model đọc/suy luận số nhiều-chữ-số. Đòn bẩy tiếp: model mạnh hơn (qwen2.5).

### `v6.2-rowlabel` ⭐ — Merge nhãn dòng TỔNG cho chunk OCR + format Mã số + prompt đọc cột-kỳ (2026-09-12)

Sửa lỗi **model đọc sai dòng/cột bảng nhiều dòng số lớn** — nút thắt còn lại sau v6.1. Ba thay đổi:
1. **Merge nhãn dòng tổng** ([app/ingestion/pdf_ingestion.py](../../app/ingestion/pdf_ingestion.py) `_ocr_row_documents`): VAS tách
   tên chỉ tiêu tổng ("Tài sản ngắn hạn") ra dòng-chỉ-nhãn riêng, dòng-số kế mang nhãn = công thức
   "(100 = 110 + ...)"; OCR-parsing cũ bỏ dòng-chỉ-nhãn (`if not values`) → dòng tổng [100]/[200]/[270]
   mất tên → không truy hồi/đọc được. Nay giữ **nhãn treo** và gán cho dòng-số có nhãn YẾU (công thức/mã),
   có guard không cướp nhãn dòng mạnh. (camelot native đã có; đường OCR nay mới có.)
2. **Format `[Mã số NNN]`**: tách ô Mã số khỏi nhãn, bỏ cột thuyết minh khỏi nhãn.
3. **SYSTEM_PROMPT rule 12–15** ([app/agent/financial_agent.py](../../app/agent/financial_agent.py)): đọc **cột kỳ hiện tại**
   (không lấy năm trước), **khớp đúng Mã số + nhãn** (phân biệt "X" với "X khác"/thuyết minh), trả **MỘT**
   số, **sanity độ lớn** (số tổng phải lớn). Query-time, áp mọi tài liệu.

Đã kiểm chứng OCR thật: dòng tổng SAB nay tồn tại với nhãn+value đúng (Mã 100=22.140.977.869.887,
200=10.456..., 270=32.597...). Eval `hybrid_rerank`, re-ingest SAB+DHG; VNM `--skip-ingest`.

| Tài liệu | Chỉ số | v6.1 | **v6.2** | Δ |
|---|---|---|---|---|
| **SAB** (scan) | Retrieval | 81.5% | 81.5% | = |
| | Numeric | 47.6% | **61.9%** | **+14.3pp** |
| | Judge | 51.7% | **58.6%** | **+6.9pp** |
| | Âm-tính | 50% | 50% | = |
| **DHG** (scan) | Numeric | 90.5% | **100%** (21/21) | **+9.5pp** |
| | Judge | 85.7% | **96.4%** (27/28) | **+10.7pp** |
| | Retrieval / Âm-tính | 96.2% / 100% | 96.2% / 100% | = |
| **VNM** (native) | Retr/Num/Judge/Âm | 84.6/85.0/85.7/100 | **y hệt** | = (prompt không hồi quy) |

**Thắng rõ, không hồi quy đâu cả.** SAB `bang_can_doi` judge **50%→80%** (Q6 TS ngắn hạn [100] & Q7 TS
dài hạn [200] từ BAD→OK — đúng nhờ recover dòng tổng). DHG hưởng lợi mạnh (cùng lỗi dòng tổng VAS):
**numeric tuyệt đối 100%, judge 96.4% — cao nhất từ trước tới nay** (`dump_failures` bắn, vượt best 92.9%).
VNM native y hệt → prompt rule 12–15 an toàn cho tài liệu đã tốt.

**Còn lại (SAB)**: `luu_chuyen` judge 0% (đọc bảng LCTT nhiều dòng — cấu trúc khác), `suy_luan` 0% (tính
%), Q5 "TỔNG TÀISẢN" retrieval MISS (nhãn OCR dính chữ "TÀISẢN" ≠ query "Tổng cộng tài sản"). Hướng tiếp:
LCTT/suy-luận + có thể chuẩn hoá nhãn OCR / model mạnh hơn (qwen2.5).

### `v6.1-ocr-kind` — Vá gán `statement_kind` cho chunk OCR (SAB) (2026-09-12)

Sửa [app/ingestion/pdf_ingestion.py](../../app/ingestion/pdf_ingestion.py) để chunk OCR của SAB
được gán `statement_kind` (trước: `ocr_rows` 0/691 có kind → section-boost trơ, gốc rễ
`ocr-section-boost-inert`). Ba thay đổi (chỉ đường OCR, đã kiểm chứng trên OCR thật trang 6/10/12):
1. **`_ocr_statement_kind` so khớp BỎ DẤU** (`_fold`): OCR rụng dấu ("LƯU"→"LUU") làm chữ ký
   có-dấu trượt (SAB tr.12 CF). Fold cả tín hiệu lẫn chữ ký.
2. **Chữ ký mã biểu mẫu VAS** `B01→balance_sheet, B02→income_statement, B03→cash_flow`
   (`B09`=thuyết minh→""): tín hiệu mạnh, độc lập layout, chịu nhiễu OCR (`_OCR_FORM_CODE_RE`).
3. **`_ocr_block_statement_kind` quét mã biểu mẫu ở BẤT KỲ đâu trong text block** +
   `current_section_kind` trong `_extract_ocr_page`: vá SAB tr.10 KQKD nơi tiêu đề/mã bị Docling
   xuất thành dòng text thường (không `##`) nên `_ocr_section_context` bỏ sót. `_ocr_section_context`
   cũng ưu tiên heading mang kind khi `[-1]` rỗng (guard: KHÔNG đổi trang vốn đã đúng như DHG).

**Coverage sau re-ingest**: SAB `ocr_rows` có `statement_kind` **0 → 127 (18%)** (BS/IS/CF); DHG
127→131 (gần như không đổi — vốn đã chạy).

| Tài liệu | Chỉ số | trước | **v6.1** | Δ |
|---|---|---|---|---|
| **SAB** (hybrid, vs v5.4-baseline) | **Retrieval** | 74.1% | **81.5%** | **+7.4pp** |
| | Numeric | 52.4% | 47.6% | −4.8pp (1 câu) |
| | Judge | 55.2% | 51.7% | −3.5pp (1 câu) |
| | Âm-tính | 50% | 50% | = |
| **DHG** (hybrid, vs v6.0-control) | Retrieval | 96.2% | 96.2% | = |
| | Judge | 89.3% | 85.7% | −3.6pp (1 câu) |
| | Âm-tính | 100% | 100% | = |

SAB theo category: `luu_chuyen` retr **66.7%→100%**, `bang_can_doi` **80%→90%** — retrieval tăng rõ
nhờ section-boost giờ hoạt động.

**Kết luận (trung thực)**: fix đạt **đúng mục tiêu kỹ thuật** (SAB có `statement_kind`, section-boost
chạy → **retrieval SAB +7.4pp**), NHƯNG **judge không cải thiện** (SAB/DHG đều −~1 câu). Điều này
**tái khẳng định nút thắt SAB = MODEL đọc số**, không phải retrieval (retrieval tốt hơn nhưng qwen2:7b
vẫn đọc lệch số lớn 15–32 nghìn tỷ). "Hồi quy" DHG (−1 câu, coverage đổi không đáng kể, retrieval y
hệt) **gần như là nhiễu** qwen2 (temp=0 nhưng Ollama không tất định tuyệt đối). Âm-tính an toàn.

**Giữ fix** (retrieval — chỉ số được theo dõi — tăng thật; sửa đúng lỗ hổng metadata; tiền đề để
section-boost/qkind có tác dụng trên tài liệu scan). **Hướng tiếp** (nút thắt thật): (1) prompt/format
đọc đúng dòng+cột cho bảng nhiều dòng số lớn; (2) model mạnh hơn (qwen2.5:7b-instruct); (3) kiểm
SECTION_BOOST on/off trên SAB (giờ đã có kind) xem boost có net dương không.

### `v6.0-adaptive-p1` — Adaptive RAG Phase 1: Query Analyzer + Retrieval Grader + rewrite-loop (2026-09-10)

Nâng cấp kiến trúc: thêm **Adaptive RAG** (opt-in `RETRIEVAL_MODE=adaptive`, mã ở
[app/rag/adaptive.py](../../app/rag/adaptive.py)) nhúng vào chokepoint `retrieve()` nên tool + eval + agent
dùng chung, KHÔNG đổi contract. Luồng: **Query Analyzer** (1 LLM call → JSON
`type/complexity/statement_kind/strategy`; LLM hiểu tiếng Việt → phân loại `statement_kind`
lái section-boost thay keyword `_classify_query`) → `hybrid_search(qkind=...)` → **Retrieval
Grader** (chấm ngữ cảnh ĐỦ trả lời chưa) → nếu chưa đủ thì **rewrite query → retrieve lại**
(bounded `ADAPTIVE_MAX_RETRIES=1`, rerank-merge giữ k=5). Mọi lỗi LLM/parse → fallback
`hybrid_search`. Đây là **Phase 1**; Decomposition + HyDE để Phase 2.

**So sánh HEAD-TO-HEAD** (cùng code hiện tại, `--skip-ingest`): adaptive vs control
`hybrid_rerank` (`v6.0-control-hybrid`). SAB so `v5.4-sab-baseline` (cùng thời rule-11):

| Tài liệu | Chỉ số | hybrid (control) | **adaptive** | Δ |
|---|---|---|---|---|
| **SAB** (scan) | Judge | 55.2% (16/29) | **62.1% (18/29)** | **+6.9pp** |
|  | Retrieval | 74.1% | 74.1% | = |
|  | Âm-tính | 50% | 50% | = |
| **DHG** (scan) | Judge | 89.3% (25/28) | **92.9% (26/28)** | **+3.6pp** |
|  | Numeric | 90.5% | **95.2%** | **+4.7pp** |
|  | Retrieval | 96.2% (25/26) | 88.5% (23/26) | **−7.7pp** |
|  | Âm-tính | 100% | 100% | = |
| **VNM** (native) | Judge | 85.7% (24/28) | 85.7% (24/28) | = |
|  | Numeric | 85.0% | 85.0% | = |
|  | Retrieval | 84.6% (22/26) | 80.8% (21/26) | **−3.8pp** |
|  | Âm-tính | 100% | 100% | = |

**Kết luận (trung thực, trái với giả thuyết ban đầu):**
- ✅ **Âm-tính giữ nguyên MỌI tài liệu** (SAB 50%, DHG/VNM 100%) — rủi ro chính (rewrite kéo
  chunk nhiễu → bịa số) KHÔNG xảy ra nhờ `MAX_RETRIES=1` + rerank-cắt-k=5.
- ✅ **Judge ≥ control mọi tài liệu**: SAB +6.9pp, DHG +3.6pp, VNM =. Numeric: DHG +4.7pp,
  VNM/SAB =. Lợi ích tập trung ở **tài liệu SCAN** (SAB `suy_luan` 0%→50%, `ket_qua`
  62.5%→75%; DHG đọc số tốt hơn) — đến từ **Grader→rewrite**, không phải qkind.
- ❌ **Retrieval (page-recall) TỤT**: DHG −7.7pp (2 câu), VNM −3.8pp (1 câu). Cơ chế: Grader
  đôi khi báo "chưa đủ" cho câu vốn trả lời được → rewrite → rerank-merge đẩy chunk gold ra
  khỏi top-5. Đáng chú ý: **judge/numeric vẫn giữ/tăng dù page-recall giảm** (chunk còn lại đủ
  để trả lời), nhưng đây là chi phí thật.
- ❌ **Analyzer-qkind KHÔNG có đóng góp đo được trên native VNM** (judge/numeric y hệt control).
  Giả thuyết "native hưởng lợi từ qkind LLM" SAI: keyword `_classify_query` vốn đã đủ cho câu
  VNM (phần lớn tiếng Anh); LLM-classify + grader + rewrite chỉ thêm nhiễu, không thêm giá trị.

**Quyết định**: GIỮ mặc định `hybrid_rerank`; adaptive **vẫn opt-in**. Lợi ích thực (judge trên
scan) chưa đủ bù chi phí (retrieval −, 2–3× LLM call/câu) để thay mặc định. `dump_failures`
bắn cho DHG (2 câu, 92.9% > 89.3%).

**Hướng tiếp**: (1) **Sửa Grader quá nhạy** — nếu sau retry vẫn "chưa đủ", trả về docs LẦN
ĐẦU thay vì bộ đã-rewrite (chặn tụt page-recall); hoặc chỉ nhận rewrite khi grade TỐT HƠN.
(2) Giữ qkind cho tài liệu OCR có `statement_kind` (VNM đã có nhưng vô ích; DHG/SAB scan
thiếu → cần gắn `statement_kind` cho OCR chunk trước, xem `ocr-section-boost-inert`).
(3) Phase 2 Decomposition cho `luu_chuyen`/`suy_luan` nhiều-bước.

### `v5.4-sab-baseline` — Tài liệu MỚI: SAB 2025 (Sabeco, SCAN tiếng Việt, KPMG, 74 trang) (2026-09-10)

Chạy pipeline **v5.4 hiện tại** (per-row OCR + section-boost VI + tool-call repair + BM25 `\w+`
+ prompt rule 10/11) trên **tài liệu thứ 5** — Tổng Công ty CP Bia-Rượu-NGK Sài Gòn (Sabeco),
BCTC HỢP NHẤT đã kiểm toán 2025, **PDF scan** 74 trang → 74/74 OCR, **694 table chunk** (per-row),
872 chunk, ingest **669s**. Tập test MỚI `data/eval/qa_ground_truth_sab.json` (29 câu, số liệu
đối chiếu TRỰC TIẾP ảnh PDF gốc; kiểm chéo số học: tổng tài sản = nợ + vốn CSH ✓). Đây là
**baseline đầu tiên** cho SAB.

| Chỉ số | v5.4-sab-baseline |
|---|---|
| Retrieval hit-rate@k | **74.1%** (20/27) |
| Numeric accuracy | **52.4%** (11/21) |
| Judge accuracy | **55.2%** (16/29) |
| Âm-tính | **50%** (1/2) |

Theo category (retr / judge): bang_can_doi **80% / 60%**, ket_qua **87.5% / 62.5%**,
luu_chuyen **66.7% / 0%**, suy_luan **100% / 0%**, thong_tin_cong_ty **25% / 100%**,
am_tinh 50%. `dump_failures` bắn (13 câu sai).

**Đọc hiểu — retrieval ĐÃ ỔN, nút thắt chuyển sang MODEL đọc/suy luận:**
- Retrieval tốt (BM25 `\w+` phát huy: bang_can_doi 80%, ket_qua 87.5%) nhưng **numeric chỉ 52%**:
  model đọc SAI SỐ dù trúng dòng — vd Q6 trả "340.000.000" thay 22 nghìn tỷ; Q23 lấy nhầm dòng
  "LN trước thuế" thay LCT HĐKD. Số SAB rất lớn (15–32 nghìn tỷ, nhiều chữ số) → dễ đọc lệch.
- **suy_luan 0%**: Q26 trả số lợi nhuận gộp thay vì TÍNH % (không suy luận tỷ lệ).
- **luu_chuyen judge 0%**: đọc nhầm dòng/số trong bảng LCTT nhiều dòng.
- **Q28 âm-tính BỊ BỊA**: trả "giá cổ phiếu = 136.265.460.000 VND" → âm-tính tụt 50%.
- `thong_tin` retr 25% nhưng judge 100%: expected_pages có thể lệch (thông tin lặp) — judge vẫn đúng.

**Hướng cải thiện (chung, không chỉ SAB):** điểm nghẽn giờ là **model-side đọc số + suy luận +
chống bịa**, không phải retrieval. Cân nhắc: (1) prompt/format đọc đúng dòng+cột năm cho bảng
nhiều dòng; (2) hướng dẫn tính toán cho câu suy luận %; (3) siết âm-tính; (4) thử model mạnh hơn
(qwen2.5:7b-instruct) — đòn bẩy lớn cho cả đọc số lẫn suy luận.

### `v5.4-dhg-numfmt` — Prompt rule 11: con số + VND, bỏ mã số/note (2026-09-08)

Thay đổi so với `v5.3`: thêm rule 11 vào SYSTEM_PROMPT
([app/agent/financial_agent.py](app/agent/financial_agent.py)) — báo con số giá trị kèm "VND",
KHÔNG chèn mã số (Code)/thuyết minh (Note) vào con số. Fix Q23 (trước: "40 (1.957.374.626.414)"
→ nay: "1.957.374.626.414 VND"). Query-time, không re-ingest.

| Chỉ số | v5.3 | **v5.4** | Δ |
|---|---|---|---|
| Retrieval hit-rate@k | 96.2% (25/26) | 96.2% (25/26) | = |
| Numeric accuracy | 90.5% (19/21) | 90.5% (19/21) | = |
| Judge accuracy | 85.7% (24/28) | **89.3%** (25/28) | +3.6pp |
| Âm-tính | 100% (2/2) | 100% (2/2) | = |

**`luu_chuyen_tien_te` judge 66.7%→100%** (cả 3 câu OK). Còn lại: `bang_can_doi` judge 80%
(vài câu đọc số chưa chuẩn), `thong_tin_cong_ty` judge 66.7% (Q2 kiểm toán đọc nhầm thành
viên HĐQT). Lưu ý nhỏ: Q23 answer bỏ dấu âm (báo số dương) nhưng judge vẫn chấp nhận — có thể
siết Rule 4 sau. **Tiến trình DHG tổng: judge 39.3%(v4.6)→89.3%(v5.4) = +50pp; retrieval
26.9%→96.2%.**

### `v5.3-dhg-bm25-vi` ⭐⭐ — Sửa BUG tokenizer: BM25 mù tiếng Việt (2026-09-08)

Thay đổi so với `v5.2`: **1 dòng** — `_TOK_RE` trong [app/rag/retrieval.py](app/rag/retrieval.py#L60)
từ `re.compile(r"[a-z0-9]+")` → `re.compile(r"\w+")`. Regex cũ chỉ khớp ASCII nên **băm nát
chữ tiếng Việt có dấu** ("tài sản" → ['t','i','s','n']); dòng "TÀI SẢN NGẮN HẠN" tokenize ra
**rỗng** → **BM25 vô dụng với toàn bộ nội dung tiếng Việt**. Query-time, KHÔNG re-ingest.

| Chỉ số | v5.2 | **v5.3** | Δ |
|---|---|---|---|
| Retrieval hit-rate@k | 76.9% (20/26) | **96.2%** (25/26) | **+19.3pp** |
| Numeric accuracy | 76.2% (16/21) | **90.5%** (19/21) | **+14.3pp** |
| Judge accuracy | 75.0% (21/28) | **85.7%** (24/28) | **+10.7pp** |
| Âm-tính | 100% (2/2) | **100%** (2/2) | = |

Theo category: **`bang_can_doi_ke_toan` retr 50%→100%**, judge 60%→80%; ket_qua 100%/100%;
suy_luan 100%; thong_tin_cong_ty judge 33%→66.7%. Chỉ còn `luu_chuyen` judge 66.7% (Q23 lỗi
format số, không phải retrieval).

**Kết luận: đây là đòn bẩy LỚN NHẤT toàn dự án cho DHG** — một bugfix regex. BM25 (nửa hybrid
retrieval) trước giờ chết với tiếng Việt; sửa xong retrieval gần chạm trần. Bài học: kiểm
tokenizer trên ngôn ngữ đích trước khi tối ưu ranking. Áp dụng cho MỌI tài liệu tiếng Việt
(VNM/SAB/GAS...), nên re-eval các tài liệu VN khác cũng nên tăng.

### `v5.2-dhg-candidates` ⭐ — Nới recall pool (DENSE 25→50, BM25 15→30) (2026-09-08)

Thay đổi so với `v5.1`: tăng `DENSE_CANDIDATES` 25→50, `BM25_CANDIDATES` 15→30
([app/config.py](app/config.py)); **query-time, không re-ingest**. Chẩn đoán cho thấy chunk
địa chỉ DHG (trang 13) ở dense rank 38 / bm25 rank 26 — ngoài pool cũ nên section-boost
(v5.1) không cứu được (boost chỉ sắp xếp lại pool, không thêm ứng viên). Nới pool → chunk vào
top-5 → Q1 đúng. Kèm sửa ground-truth `expected_pages` Q1 `[7]→[13]` (địa chỉ thực ở OCR
trang 13). `RAG_TOP_K` giữ 5.

| Chỉ số | v5.1 | **v5.2** | Δ |
|---|---|---|---|
| Retrieval hit-rate@k | 69.2% (18/26) | **76.9%** (20/26) | +7.7pp |
| Numeric accuracy | 71.4% (15/21) | **76.2%** (16/21) | +4.8pp |
| Judge accuracy | 67.9% (19/28) | **75.0%** (21/28) | +7.1pp |
| Âm-tính | 100% (2/2) | **100%** (2/2) | = (pool rộng KHÔNG phá âm-tính) |

Theo category, v5.2: `ket_qua_kinh_doanh` **100%/100%** (retr/judge), `suy_luan` judge
50%→**100%**, `luu_chuyen` retr 100%, `bang_can_doi` retr 40%→50%. `dump_failures` bắn
(7 câu sai, giảm 9→7).

**Q1 ĐÃ FIX** (judge=True, retr HIT@3): *"Công ty Cổ phần Dược Hậu Giang ... 288 Bis Nguyễn
Văn Cừ, Phường Cái Khế, Thành phố Cần Thơ, Việt Nam."* — recall pool là đòn đúng.

**Còn lại là lỗi ĐỌC (không phải retrieval):**
- Q2 (kiểm toán): lấy đúng trang nhưng trả tên thành viên HĐQT thay vì đơn vị kiểm toán
  (Deloitte).
- Q3 (kỳ báo cáo): đọc nhầm năm — trả 31/12/**2024** thay vì 2025 (cột so sánh).
- `bang_can_doi` retr 50% và `luu_chuyen` judge 66.7%: vẫn dư địa.
→ `thong_tin_cong_ty` judge giữ 33% nhưng đổi bản chất: Q1 hết lỗi, Q2/Q3 chuyển thành lỗi đọc.

**Tiến trình DHG tổng:** judge 39.3%(v4.6)→**75.0%**(v5.2) = **+35.7pp**; retrieval
26.9%→**76.9%** = +50pp. Hướng tiếp: lỗi ĐỌC SỐ/NĂM (siết prompt "năm hiện tại vs năm so
sánh", cột value) và đọc đúng thực thể (kiểm toán) — model-side, không phải retrieval.

### `v5.1-dhg-company-profile` — Prompt anti-fabricate + section-boost 'company_profile' (2026-09-08)

Thay đổi so với `v5.0`: (A) thêm rule prompt cấm bịa địa chỉ/quốc gia khi không có dòng
địa chỉ ([financial_agent.py](app/agent/financial_agent.py)); (B) mở rộng section-boost với
loại thứ 4 **`company_profile`** — tag chunk text general-info (`_ocr_text_statement_kind`
theo heading THÔNG TIN KHÁI QUÁT/…) + thêm nhóm keyword profile vào `_QUERY_KIND_KEYWORDS`.
Re-ingest DHG (565 chunk).

| Chỉ số | v5.0 | **v5.1** | Δ |
|---|---|---|---|
| Retrieval hit-rate@k | 69.2% (18/26) | 69.2% (18/26) | = |
| Numeric accuracy | 57.1% (12/21) | **71.4%** (15/21) | +14.3pp |
| Judge accuracy | 64.3% (18/28) | **67.9%** (19/28) | +3.6pp |
| Âm-tính | 100% (2/2) | 100% (2/2) | = |

Theo category, judge tăng ở `luu_chuyen` 33%→**66.7%**; `ket_qua` giữ 87.5%. `dump_failures`
bắn (9 câu sai, giảm 11→9).

**NHƯNG mục tiêu chính (Q1) KHÔNG đạt.** Q1 vẫn judge=BAD, `retr_pages=[3,3,4,3,31]` — trang
13 (tên+địa chỉ) **vẫn không được truy hồi**. Nguyên nhân lặp lại: **section-boost chỉ SẮP
XẾP LẠI pool, không THÊM ứng viên** — chunk trang 13 không lọt pool dense-25 ∪ BM25-15 cho câu
Q1 (câu ngắn "công ty nào + trụ sở" khớp yếu với chunk profile ~2900 ký tự loãng), nên boost
chỉ nhấc được trang 3/4 (cũng tag company_profile). Model đọc trang 3 (báo cáo ban điều hành,
OCR nhiễu) → **bịa tên "Dược Hải Dương"**. Prompt có tác dụng: hết bịa "Thái Lan" (giờ nói
"Việt Nam nhưng không rõ"), nhưng tên vẫn sai. Q2 (kiểm toán) có lấy được trang 13 nhưng đọc
nhầm thành viên HĐQT thành đơn vị kiểm toán (lỗi đọc, không phải truy hồi).

**Kết luận & hướng đúng (v5.2):** nút thắt Q1 là **RECALL ứng viên**, không phải ranking. Cần
đưa trang 13 vào pool trước rerank/boost:
1. Tăng `DENSE_CANDIDATES`/`BM25_CANDIDATES` (query-time, không re-ingest) để chunk profile
   lọt pool → boost mới nhấc được.
2. Hoặc chia nhỏ chunk profile trang 13 (khối ~2900 ký tự đa chủ đề) để câu "trụ sở" khớp
   chunk tinh hơn. Khớp memory [[ocr-section-boost-inert]] (boost ≠ recall).
Lưu ý `expected_pages=[7]` của Q1 thực tế sai (địa chỉ ở OCR trang 13) — retrieval metric của
Q1 sẽ vẫn tính MISS dù có lấy được trang 13; cần sửa ground-truth khi xử lý v5.2.

### `v5.0-dhg-toolcall-repair` — Sửa lỗi tool-call rò ra dạng TEXT (2026-09-07)

Thay đổi so với `v4.9`: thêm middleware **repair** cho agent. `qwen2:7b` (qua Ollama) thỉnh
thoảng phát lệnh gọi tool theo template text `<tool_call>{...}</tool_call>` nhét vào
`message.content` thay vì kênh `tool_calls` → `create_agent` tưởng là câu trả lời cuối → trả
NGUYÊN chuỗi tool-call làm answer (3/28 câu DHG cố định: id 1, 3, 16; `agent_error=None`).
Middleware `wrap_model_call` ([app/agent/tool_call_repair.py](app/agent/tool_call_repair.py))
phát hiện blob khi `tool_calls` rỗng, parse JSON, gán lại `tool_calls` → agent chạy search
thật. No-op khi không rò. Không đổi model/prompt/tool/retrieval; chạy `--skip-ingest`.

| Chỉ số | v4.9 | **v5.0** | Δ |
|---|---|---|---|
| Retrieval hit-rate@k | 69.2% (18/26) | 69.2% (18/26) | = (không đụng truy hồi) |
| Numeric accuracy | 57.1% (12/21) | 57.1% (12/21) | = |
| Judge accuracy | 60.7% (17/28) | **64.3%** (18/28) | +3.6pp |
| Âm-tính | 100% (2/2) | **100%** (2/2) | = |

`thong_tin_cong_ty` judge **0%→33.3%** (3 câu từng bị rò text nay trả lời thật; Q3 đúng).
Q1/Q16 vẫn sai nhưng vì **nội dung** (Q1 trả tên công ty con trong thuyết minh — vi phạm
rule 8; Q16 đọc lệch số), KHÔNG còn là lỗi rò tool-call. Verify: unit test parser trên 3
chuỗi rò thật + smoke test agent (hết `</tool_call>`).

**Tiến trình DHG tổng:** judge 39.3%(v4.6)→**64.3%**(v5.0) = **+25pp**; retrieval 26.9%→69.2%.

### `v4.9-dhg-sectionboost-vi` ⭐ — Bật section-boost cho câu hỏi TIẾNG VIỆT (2026-09-07)

Thay đổi so với `v4.8`: thêm cụm từ khoá **tiếng Việt** vào `_QUERY_KIND_KEYWORDS`
([app/rag/retrieval.py:169](app/rag/retrieval.py#L169)) để `_classify_query` phân loại được
câu hỏi Việt (trước đây chỉ có keyword tiếng Anh → luôn trả `None` → section-boost không bao
giờ bắn). Row chunk OCR đã mang `statement_kind` từ v4.8, nên chỉ cần sửa mắt xích phân loại
câu hỏi là bật cả chuỗi boost. **Không re-ingest** (chỉ đổi query-classifier, chạy
`--skip-ingest` trên dữ liệu v4.8). Đã verify 28/28 câu phân loại đúng (company-profile &
Q27 "giá cổ phiếu" → `None` an toàn; Q18 "lợi nhuận **kế toán** trước thuế" cần thêm keyword
`"trước thuế"`).

| Chỉ số | v4.8-rows | **v4.9-boost-vi** | Δ vs v4.8 |
|---|---|---|---|
| Retrieval hit-rate@k | 61.5% (16/26) | **69.2%** (18/26) | +7.7pp |
| Numeric accuracy | 52.4% (11/21) | **57.1%** (12/21) | +4.7pp |
| Judge accuracy | 57.1% (16/28) | **60.7%** (17/28) | +3.6pp |
| Âm-tính | 100% (2/2) | **100%** (2/2) | = (Q28 không regress) |

Theo category (retr / judge), v4.9:
| Category | n | Retrieval | Judge | (v4.8 judge) |
|---|---|---|---|---|
| bang_can_doi_ke_toan | 10 | 40.0% | 60.0% | (60.0%) |
| ket_qua_kinh_doanh | 8 | 87.5% | **87.5%** | (62.5%) |
| luu_chuyen_tien_te | 3 | 100% | 33.3% | (33.3%) |
| suy_luan_tinh_toan | 2 | **100%** | 50.0% | (100%) |
| thong_tin_cong_ty | 3 | 66.7% | 0.0% | (0.0%) |
| am_tinh_khong_co | 2 | — | 100% | (100%) |

**Kết luận: hiệu quả, an toàn.** KQKD judge +25pp (boost nhấc đúng dòng IS lên top-1),
suy_luan retr 0%→100%. Âm-tính giữ 100% dù Q28 nay phân loại income (prompt agent chặn bịa
số thành công). `dump_failures` bắn (11 câu sai, giảm dần 17→12→11).

**Còn lại (không phải lỗi ranking):**
- `bang_can_doi` retr vẫn 40% — boost chỉ **sắp xếp lại pool**, không thêm ứng viên; các
  dòng bảng cân đối còn thiếu KHÔNG lọt pool dense/BM25 trước rerank → là vấn đề **recall
  ứng viên**, cần tăng `DENSE_CANDIDATES`/BM25 hoặc embedding VI tốt hơn, không phải boost.
- `luu_chuyen` judge 33% dù retr 100% → agent **đọc lệch số** (num MISS Q23/Q24) → hướng
  prompt/model.

**Tiến trình DHG tổng:** retr 26.9%→**69.2%** (+42.3pp), judge 39.3%→**60.7%** (+21.4pp)
qua v4.7→v4.8→v4.9. Đòn bẩy quyết định: **per-row (v4.8)**; boost VI (v4.9) tinh chỉnh thêm.

### `v4.8-dhg-ocr-rows` ⭐ — Tách bảng OCR theo DÒNG + header năm (2026-09-07)

Thay đổi so với `v4.7`: chunk bảng scan (OCR) từ **1 chunk/bảng** → **1 chunk NHỎ mỗi
DÒNG dữ liệu**, dạng câu tự nhiên kèm header năm căn cột (mô phỏng `camelot_rows` của
native). Vd: `A TÀI SẢN NGẮN HẠN 100 (trang 7): Số cuối năm = 3.888.768.378.369; Số đầu
năm = 4.604.003.766.930.` Cờ `OCR_TABLE_MODE=rows` (nay là **mặc định**). Row chunk mang
metadata `statement_kind` (phân loại VN theo section-title) sẵn cho section-boost. Chỉ đụng
`app/ingestion/pdf_ingestion.py` + `app/config.py`. Ingest: bảng chunk **59 → 433**, tổng
chunk 191 → 565.

| Chỉ số | v4.6-baseline | v4.7-section | **v4.8-rows** | Δ vs baseline |
|---|---|---|---|---|
| Retrieval hit-rate@k | 26.9% (7/26) | 30.8% (8/26) | **61.5%** (16/26) | **+34.6pp** |
| Numeric accuracy | 23.8% (5/21) | 28.6% (6/21) | **52.4%** (11/21) | **+28.6pp** |
| Judge accuracy | 39.3% (11/28) | 32.1% (9/28) | **57.1%** (16/28) | **+17.8pp** |
| Âm-tính | 100% | 100% | **100%** (2/2) | = |

Theo category (retr / judge), v4.8:
| Category | n | Retrieval | Judge | (retr baseline) |
|---|---|---|---|---|
| bang_can_doi_ke_toan | 10 | **40.0%** | 60.0% | (0.0%) |
| ket_qua_kinh_doanh | 8 | **87.5%** | 62.5% | (50.0%) |
| luu_chuyen_tien_te | 3 | **100%** | 33.3% | (66.7%) |
| thong_tin_cong_ty | 3 | 66.7% | 0.0% | (33.3%) |
| suy_luan_tinh_toan | 2 | 0.0% | 100% | (0.0%) |
| am_tinh_khong_co | 2 | — | 100% | — |

**Kết luận: đúng đòn bẩy.** Per-row nhấc retrieval +34.6pp, judge +17.8pp; bảng cân đối
lần đầu thoát 0% (→40%), KQKD 50%→87.5%, lưu chuyển →100%. Khớp memory
[[retrieval-granularity-bottleneck]]: nút thắt là **độ mịn chunk**, không phải trích xuất.
`dump_failures` tự bắn (12 câu sai, giảm từ 17). So sánh trực tiếp: prepend section-title
(v4.7) gần như vô ích, chính **per-row** mới quyết định.

**Điểm còn yếu (hướng tiếp theo):**
- `luu_chuyen` judge 33.3% dù retr 100% → agent đọc lệch số (num MISS Q23/Q24), không
  phải lỗi truy hồi. Hướng: siết prompt đọc số / model mạnh hơn.
- `bang_can_doi` retr mới 40% → còn dòng bị thuyết minh đè. Có thể cộng thêm section-boost
  cho câu VN (thêm keyword tiếng Việt vào `_QUERY_KIND_KEYWORDS`) — `statement_kind` đã
  set sẵn ở row chunk, chỉ thiếu vế query-classifier VN.
- `thong_tin_cong_ty` judge 0% (n=3) — dạng câu company-profile, tách riêng.

### `v4.7-dhg-ocr-section` — Prepend section-title vào table chunk OCR (2026-09-06)

Thay đổi so với `v4.6-dhg-baseline`: chèn tiêu đề section + ngày + đơn vị vào đầu
`page_content` của **table chunk OCR** (cờ `OCR_PREPEND_SECTION_TITLE`, mặc định bật).
Docling đặt heading `## BẢNG CÂN ĐỐI KẾ TOÁN` ở text block riêng nên chunk bảng vốn là
bảng pipe trần; nay `_extract_ocr_page` mang heading `##` (lấy `##` cuối, bỏ tên công ty)
+ dòng ngày/đơn vị sang chunk bảng kế tiếp. Chỉ đụng `app/ingestion/pdf_ingestion.py`
(+ 1 cờ config). Cấu trúc chunk không đổi (vẫn 59 bảng / 191 chunk).

| Chỉ số | v4.6-baseline | v4.7-section | Δ |
|---|---|---|---|
| Retrieval hit-rate@k | 26.9% (7/26) | **30.8%** (8/26) | +3.9pp |
| Numeric accuracy | 23.8% (5/21) | **28.6%** (6/21) | +4.8pp |
| Judge accuracy | 39.3% (11/28) | **32.1%** (9/28) | **−7.2pp** |
| Âm-tính | 100% (2/2) | 100% (2/2) | = |

Theo category (retr / judge): bang_can_doi **0.0% / 20.0%** (baseline 0.0% / 30.0%),
ket_qua_kinh_doanh 62.5% / 50.0% (baseline 50.0% / 62.5%), luu_chuyen 66.7% / 33.3% (=).

**Kết luận: thay đổi KHÔNG hiệu quả cho mục tiêu chính, và regress judge.** Retrieval +
numeric nhích nhẹ (trong biên nhiễu), nhưng judge giảm và **bang_can_doi_ke_toan vẫn
retrieval 0%** — đúng chỗ cần cứu.

**Chẩn đoán gốc rễ (mới, quan trọng):** soi retrieved pages của 10 câu bảng cân đối →
chúng truy hồi toàn **trang 22–36 (THUYẾT MINH)**, gần như không chạm trang 7–9 (bảng cân
đối chính). Các chunk thuyết minh giải thích từng khoản mục (cùng con số + nhiều văn xuôi)
**đè** bảng chính cô đọng ở cross-encoder + dense. Đây CHÍNH LÀ hiện tượng section-boost
(v4.6) sinh ra để chống — nhưng section-boost đọc `metadata.statement_kind`, mà tài liệu
OCR **không hề set field này** (chỉ camelot_rows set) nên boost trơ hoàn toàn. Prepend
tiêu đề vào `page_content` không thể bù vì nó không tác động section-boost, và lượng văn
xuôi thuyết minh về tài sản/nợ quá lớn.

**Hướng đi đúng (v4.8):** set `metadata.statement_kind` cho table chunk OCR (phân loại theo
nhãn dòng tổng, mô phỏng `_statement_kind` trong `camelot_extract.py`) để **kích hoạt
section-boost** — nhấc bảng cân đối chính lên trên nhóm thuyết minh cùng số. Đây là biến
đơn lẻ tiếp theo, giữ nguyên prepend (behind flag) hoặc tắt để cô lập. Khớp memory
[[retrieval-granularity-bottleneck]]: nút thắt là truy hồi/độ ưu tiên chunk, không phải
trích xuất.

### `v4.6-dhg-baseline` — Tài liệu MỚI: DHG BCTC 2025 (SCAN tiếng Việt, Deloitte) (2026-09-06)

Chạy pipeline v4.6 (camelot_rows + section-boost) trên **tài liệu thứ 4** — Công ty CP
Dược Hậu Giang, BCTC đã kiểm toán năm 2025, **PDF dạng SCAN** → 40/40 trang qua OCR
(Docling + EasyOCR `vi`), 59 bảng, 191 chunks, ingest **376,5s**. Tập test
`data/eval/qa_ground_truth_dhg.json` (28 câu, số liệu đối chiếu trực tiếp ảnh PDF gốc).
Đây là **baseline đầu tiên** cho DHG (không có mốc trước để so).

| Chỉ số | v4.6-dhg-baseline |
|---|---|
| Retrieval hit-rate@k | **26.9%** (7/26), mean recall 25.0% |
| Numeric accuracy | **23.8%** (5/21) |
| Judge accuracy | **39.3%** (11/28) |
| Âm-tính (không bịa số) | **100%** (2/2) |

Theo category:

| Category | n | Retrieval | Judge |
|---|---|---|---|
| bang_can_doi_ke_toan | 10 | **0.0%** | 30.0% |
| ket_qua_kinh_doanh | 8 | 50.0% | 62.5% |
| luu_chuyen_tien_te | 3 | 66.7% | 33.3% |
| thong_tin_cong_ty | 3 | 33.3% | 0.0% |
| suy_luan_tinh_toan | 2 | 0.0% | 0.0% |
| am_tinh_khong_co | 2 | — | 100% |

**Đọc hiểu:**
- Điểm số thấp hơn hẳn 3 tài liệu native trước (aapl/vnm/tsla) — đúng như kỳ vọng với
  **PDF scan**: OCR làm nhiễu số liệu + phá cấu trúc bảng, kéo cả retrieval lẫn numeric xuống.
- **Bất thường lớn nhất: bang_can_doi_ke_toan retrieval = 0.0% (0/10).** 10/10 câu bảng
  cân đối kế toán đều MISS trang, trong khi ket_qua_kinh_doanh (trang 10) lại HIT nhiều
  câu. Cần kiểm tra: (a) OCR có tách được bảng cân đối thành chunk truy hồi được không,
  và (b) `expected_pages` trong ground-truth (7–9) có khớp thứ tự trang PDF 40 trang không
  — nghi ngờ lệch trang giữa số trang in trên báo cáo và index trang PDF.
- **Âm-tính 100%**: không bịa số cho câu bẫy — tốt.

**Hướng cải thiện đề xuất:**
1. Soi 17 câu sai đã dump tự động: `data/eval/failures/failures_v4.6-dhg-baseline_*.json`.
2. Kiểm tra thủ công `expected_pages` bảng cân đối vs trang OCR thực tế (đọc chunk theo page).
3. Nếu OCR tách bảng cân đối kém → tinh chỉnh `split_markdown_blocks` / TableFormer cho scan.

### `v4.6-aapl` — Tài liệu MỚI: Apple FY2023 condensed statements (2026-09-05)

Chạy pipeline v4.6 (camelot_rows + section-boost) trên **tài liệu thứ 3** — Apple Inc.,
báo cáo tài chính rút gọn *unaudited* năm tài chính kết thúc 30/09/2023 (chỉ **3 trang**:
IS / BS / CF). Tập test mới `data/eval/qa_ground_truth_aapl.json` (33 câu). Đây là
**baseline đầu tiên** cho Apple (không có mốc trước để so).

| Chỉ số | v4.6-aapl |
|---|---|
| Retrieval hit-rate@k | **96.8%** (30/31) |
| Numeric accuracy | **92.0%** (23/25) |
| Judge accuracy | **78.8%** (26/33) |
| Âm-tính | **0%** (0/2) |

Theo nhóm: balance_sheet judge **100%** (8/8), income_statement **83.3%** (10/12),
cash_flow **83.3%** (5/6), company_profile **100%** (2/2), reasoning **33.3%** (1/3),
negative **0%** (0/2).

**7 câu sai — chẩn đoán (retrieval GẦN NHƯ HOÀN HẢO, lỗi ở khâu ĐỌC/LLM):**
- **Q4, Q12** (retr HIT r@1): LLM **đọc lệch dòng/cột** trong bảng IS dày 4 cột của Apple
  (3-tháng ×2 + 12-tháng ×2). Q4 Products net sales → trả 189,282 (nhầm sang *cost of sales*);
  Q12 EPS → trả 15,744,231 (nhầm sang *shares used in computing EPS*). Không phải lỗi truy hồi.
- **Q28** (retr MISS): `_classify_query` gắn câu "acquisition of **property, plant** and equipment"
  vào `balance_sheet` (từ khoá "property, plant") → section-boost nhấc dòng BS *PP&E net 43,715*
  vượt dòng CF đúng *10,959*. **Đây là tác dụng phụ của section-boost** — câu cash_flow nhưng
  chứa từ khoá dùng chung với balance_sheet. Cần thêm ưu tiên CF cho cụm "payments/acquisition of".
- **Q29, Q31** (retr HIT): LLM **tính sai tỉ lệ** (Q31 trả 64.1% thay vì ~22.2%) hoặc né tính
  (Q29) — hạn chế số học của qwen2:7b, đã thấy ở Tesla.
- **Q32, Q33** (âm-tính): pool Apple **chỉ có bảng số** (3 trang toàn số) nên luôn trả về số hấp
  dẫn → qwen2:7b **bịa** (headcount→200,583 = iPhone sales; guidance 2024→383,285 = total net
  sales). Tài liệu không có văn xuôi để model "thấy trống mà abstain". Đây là điểm yếu rõ nhất.

**Kết luận:** trích bảng + truy hồi trên tài liệu native tiếng Anh rất tốt (retr 96.8%, num 92%).
Điểm nghẽn còn lại là **(a)** LLM đọc lệch cột ở bảng nhiều cột, **(b)** section-boost misroute khi
từ khoá dùng chung (Q28), **(c)** âm-tính khi tài liệu toàn bảng số. Câu sai đã dump:
`data/eval/failures/failures_v4.6-aapl-sectionboost_Apple_Inc_20260905_201558.json`.

### `v4.6` ⭐ — Section-aware retrieval boost (2026-09-05)

Gỡ nút thắt RANKING: 9/13 câu sai còn lại của Tesla là retrieval-miss — dòng bảng BCTC chính
SẠCH có trong pool (BM25 hạng 0–4) nhưng **cross-encoder dìm xuống hạng 8–22**, ưu ái văn xuôi
MD&A/thuyết minh cùng số. Vì metadata `statement` rỗng, ta **tự tạo tín hiệu section**:
- **Gắn `statement_kind`** khi ingest ([camelot_extract.py](../../app/ingestion/camelot_extract.py)
  `_statement_kind`): phân loại TRANG theo dòng-tổng đặc trưng (balance_sheet / income_statement /
  cash_flow) qua union nhãn các bảng cùng trang. Tesla p50/51/54 + VNM p7/10-11/12-13 gắn đúng.
- **Boost khi rerank** ([retrieval.py](../../app/rag/retrieval.py) `_classify_query` + `hybrid_search`):
  phân loại câu hỏi (đã kiểm 48/48 statement đúng, 0 harmful trên company/operational), rồi
  `final = minmax(cross) + W·(statement_kind==query_kind)`, **W=1.0** (đã tune: W=0.3 chỉ cứu
  4/10, W=1.0 cứu 7/10). Cờ `SECTION_BOOST`, `SECTION_BOOST_WEIGHT` ([config.py](../../app/config.py)).

| Chỉ số | Tesla v4.5→**v4.6** | VNM v4.5→**v4.6** |
|---|---|---|
| Retrieval hit-rate@k | 22.6% → **61.3%** (+38.7) | 80.8% → **84.6%** (+3.8) |
| Numeric accuracy | 45.8% → **83.3%** (+37.5) | 80.0% → **90.0%** (+10.0) |
| Judge accuracy | 60.6% → **75.8%** (+15.2) | 89.3% → **92.9%** (+3.6) |
| Âm-tính | 50% → **100%** | 100% → **100%** = |

**WIN lớn CẢ HAI tài liệu, KHÔNG hồi quy âm-tính.** Rủi ro Q32/Q33 (Tesla) & Q28 (VNM) đã lo
trước — thực tế **âm-tính giữ/tăng**: Tesla 50%→100% (Q33 giờ đúng, Q32 giữ đúng), VNM giữ 100%
(Q28 KHÔNG regress). Boost không kéo được số vào pool câu bẫy vì tài liệu không có dữ liệu đó.
- Tesla judge flips: **+6 BAD→OK** (Q6,7,11 balance, Q21 R&D, Q26 cash, Q33 negative) −1 (Q14).
- VNM judge flips: **+2 BAD→OK** (Q5, Q24) −1 (Q25 reasoning gross-margin, borderline). Numeric
  80→90, retrieval 80.8→84.6 → section-boost còn **cải thiện** VNM chứ không chỉ giữ.

**Tesla còn 8 câu BAD** (ngoài phạm vi section-boost): Q24 (EPS 4.73/4.30 <5 chữ số bị rớt khi
trích — lỗi extraction), Q3 (LLM né tránh dù context đúng), Q10 (LLM chọn nhầm p75 note gộp
"41,782" thay net PP&E 29,725), Q9/Q14 (đọc lệch số), Q20 (income from operations — dòng cụ thể
vẫn bị vùi trong 18 dòng income cùng trang), Q29/Q31 (reasoning cần nhiều số/không vào pool).

---

### `v4.5` — Prose-guard: bỏ VĂN XUÔI bị camelot nhận nhầm thành "dòng bảng" (2026-09-02)

Vá lỗi phát hiện khi soi chunk: `camelot_rows` đẻ ra chunk "bảng" thực chất là **câu văn bị vỡ**
(vd p49 báo cáo kiểm toán: *"million as of December 31, The Company provides a manufacturer's
warranty … (page 49): current year = 2023.."*). Trên trang chỉ có chữ, camelot trả về **bảng
1 CỘT**, B1 rút số ra khỏi câu thành `values`, để cả câu làm `label`. Đo: **259/864 (29%)** chunk
camelot_rows của Tesla là văn-xuôi-vỡ, trên 57 trang — **tất cả đều đã có native text chunk phủ**.

**Fix ([camelot_extract.py](../../app/ingestion/camelot_extract.py)):** trong **bảng 1 cột**
(`df.shape[1] <= 1` = camelot không tìm được cấu trúc cột), bỏ dòng có **nhãn giống câu văn**
(`_is_prose_label`: ≥8 từ). Bảng **đa cột luôn giữ trọn** → không đụng line-item nhãn dài
("Common stock; $ par value; 6,000 shares…"). Glued-data thật trong bảng 1 cột có nhãn NGẮN
("Cash on hand", "Miraka Holdings Limited") nên được giữ. Đã kiểm read-only: **0 dòng báo cáo
tài chính thật bị bỏ** trên cả Tesla (p50/51/54) lẫn VNM (p7/8/10/12; giữ nguyên p35 cash /
p38 investments).

| Chỉ số | Tesla v4.4 | **Tesla v4.5** | | VNM v4.4 | **VNM v4.5** |
|---|---|---|---|---|---|
| Retrieval hit-rate@k | 22.6% | **22.6%** = | | 80.8% | **80.8%** = |
| Numeric accuracy | 41.7% | **45.8%** ⬆+4.2pp | | 80.0% | **80.0%** = |
| Judge accuracy | 54.5% | **60.6%** ⭐+6.1pp | | 92.9% | **89.3%** (nhiễu, xem dưới) |
| Âm-tính | 50% | **50%** = | | 100% | **100%** = |

**Tesla WIN:** bỏ 185 chunk nhiễu → context sạch hơn → **+2 câu BAD→OK (Q13 retained earnings,
Q14 accounts payable), 0 regress**; numeric +4.2pp, judge +6.1pp.

Ingest: Tesla 864→**679** table chunk (tổng 1419→1234, bỏ 185 văn-xuôi, gồm đúng chunk người
dùng phản ánh — p49 còn 0);
VNM 605→**586** (bỏ 19 prose). **VNM không hồi quy thật:** retrieval/numeric/âm-tính y hệt v4.4;
judge −1 câu (Q1 company_profile) **là NHIỄU LLM, KHÔNG do guard** — đã chứng minh: retrieved
pages Q1 **y hệt** v4.4 ([4,15,38,5,23]) và context vẫn có tên công ty ở **rank 1 (p4 VINAMILK)
+ rank 2 (p15 VIETNAM DAIRY PRODUCTS JOINT STOCK COMPANY)**; qwen2:7b lần này trả lời né tránh
dù đáp án nằm ngay top-2 (Q1 dao động BAD→OK→BAD qua các version = judge/LLM borderline). p38
investments thật vẫn được giữ (rank 3: "Asia Coconut Processing… 24.96%…").

**Kết luận: WIN cho Tesla (judge +6.1pp, numeric +4.2pp) mà không mất dữ liệu thật, không hồi
quy VNM.** Bỏ chunk rác không chỉ dọn ngữ nghĩa mà còn giảm nhiễu retrieval → LLM đọc đúng hơn.
(Fix B — narrative splitter tách theo câu — HOÃN, lượt sau.)

---

### `v4.4` ⭐ — `row_tol` NHỎ phổ quát (6): sửa Tesla KHÔNG hồi quy Vinamilk (2026-09-02)

Một cấu hình DUY NHẤT cho cả hai layout, **bỏ hẳn việc chọn mode thủ công theo tài liệu**.
Điều tra sâu (đo camelot read-only) đã đảo lại chẩn đoán ở `v4.3-vnm-adaptive`: hồi quy
Vinamilk **KHÔNG do adaptive tách nhãn wrap**, mà do **camelot SẬP phát hiện CỘT ở `row_tol`
lớn** (VNM pitch ~28pt → adaptive tol~17 → bảng về **1 cột**, số dính vào nhãn; ngay cả
fixed=15 cũng làm p43 sập). **`row_tol` nhỏ (6) giữ cột ổn định trên CẢ hai tài liệu**, mỗi
dòng vật lý = 1 grid-row (Tesla hết gộp cặp), và **nhãn-wrap của VAS được logic POST-hoc sẵn
có nối lại** (`_is_label_only` + `_looks_like_continuation`). Thay đổi tối thiểu:
`CAMELOT_ROW_TOL` 15→**6** ([config.py](../../app/config.py)); không đụng luồng trích.

| Chỉ số | Tesla v4.2 (15) | **Tesla v4.4 (6)** | | VNM v4.2 (15) | **VNM v4.4 (6)** |
|---|---|---|---|---|---|
| Retrieval hit-rate@k | 25.8% | **22.6%** (7/31) | | 80.8% | **80.8%** = |
| Numeric accuracy | 20.8% | **41.7%** (10/24) ⭐×2 | | 80.0% | **80.0%** = |
| Judge accuracy | 42.4% | **54.5%** (18/33) ⭐+12pp | | 89.3% | **92.9%** ⭐ +3.6pp |
| Âm-tính | 50% | **50%** | | 100% | **100%** = |

**Vinamilk: KHÔNG hồi quy — còn NHỈNH hơn.** judge +1 (Q1 BAD→OK), 0 regress; retrieval/
numeric/âm-tính giữ nguyên. Q10/Q11 (bị adaptive làm rớt trang 8) nay **HIT r@1, num=OK**.
Ingest 605 bảng/758 chunks (v4.2: 589/742). **Tesla: row_tol=6 = 864 bảng/1419 chunks TRÙNG
KHÍT bản adaptive (tol=8) → điểm y hệt `v4.3-tsla-adaptive`** (numeric 41.7%, judge 54.5%,
retrieval 22.6%) — đã xác nhận, temperature=0. Nút thắt còn lại của Tesla là RANKING (total
assets/current assets lặp ở MD&A lấn át trang BCTC), tách hẳn khỏi extraction, để dành
section-aware boost.

**Kết luận: hằng nhỏ phổ quát thắng adaptive lẫn fixed=15.** Không cần env chọn mode; sửa
được cả lỗi gộp cặp của Tesla lẫn lỗi sập cột của Vinamilk (kể cả p43 mà fixed=15 làm hỏng).
Phần B (gia cố heuristic nối nhãn cho nhãn viết-HOA/prefix-wrap ở 7 tài liệu khác) **chưa
cần** — 2 tài liệu eval không có ca nào. `CAMELOT_ROW_TOL_MODE` adaptive giữ lại làm tuỳ chọn.

---

### `v4.3-vnm-adaptive` — Kiểm tra HỒI QUY: adaptive `row_tol` trên Vinamilk (2026-09-02)

Chạy adaptive `row_tol` (đổi mới ở `v4.3-tsla-adaptive`) trên Vinamilk để chắc chắn không
phá config thắng v4.2. **Kết quả: HỒI QUY rõ — adaptive KHÔNG được làm mặc định toàn cục.**

| Chỉ số | v4.2 (fixed row_tol=15) | **v4.3-vnm-adaptive** | Δ |
|---|---|---|---|
| Retrieval hit-rate@k | 80.8% (21/26) | **69.2%** (18/26) | **−11.6pp** ↓ |
| Numeric accuracy | 80.0% (16/20) | **70.0%** (14/20) | **−10.0pp** ↓ |
| Judge accuracy | 89.3% (25/28) | **78.6%** (22/28) | **−10.7pp** ↓ |
| Âm-tính | 100% | **100%** | = |

Judge flips: **0 BAD→OK, −3 OK→BAD** (Q10, Q11 balance_sheet; Q25 reasoning). Ingest 604
bảng/757 chunks (v4.2: 589/742 — chênh nhỏ nhưng đủ phá).

> ⚠️ **HIỆU CHỈNH CHẨN ĐOÁN (xem `v4.4`):** giả thuyết "adaptive tách nhãn wrap" bên dưới
> **SAI**. Điều tra sâu cho thấy nguyên nhân thật là **camelot SẬP phát hiện CỘT ở tol lớn**
> (VNM pitch ~28pt → adaptive ~17 → bảng về 1 cột, số dính nhãn). `row_tol` NHỎ (6) mới là
> lời giải: giữ cột ổn định + logic nối-nhãn sẵn có xử lý wrap → `v4.4` không hồi quy.

**Nguyên nhân (giả thuyết BAN ĐẦU — đã bị `v4.4` bác bỏ):** ngờ rằng VAS có **nhãn xuống dòng**
mà `row_tol=15` cố ý gộp, còn adaptive hạ tol làm tách. Thực tế đo lại: ở tol nhỏ nhãn-wrap
vẫn được nối POST-hoc, và Q10/Q11 rớt là do **cột sập** (số nhập vào nhãn) chứ không phải
nhãn bị tách. Xem `v4.4` để có chẩn đoán đúng.

**Quyết định: `CAMELOT_ROW_TOL_MODE` giữ mặc định `fixed`; `adaptive` chỉ bật theo tài liệu**
(SEC 10-K tiếng Anh dòng khít → adaptive; VAS wrapped-label → fixed=15). Đây là lựa chọn
**per-document**, không phải per-repo. Hướng tổng quát hoá thật sự: heuristic nhận diện
wrapped-label (nhãn viết thường/không có số ở dòng kế) để merge có chọn lọc thay vì chỉ dựa
line pitch — khi đó một chế độ mới có thể đúng cho cả hai layout.

---

### `v4.3-tsla-adaptive` — Adaptive `row_tol` = 0.6 × line-pitch cho camelot (2026-09-02)

Sửa **lỗi extraction gốc** mà export Tesla lộ ra (xem `v4.2-tsla`): camelot stream với
`row_tol=15` cố định **gộp mỗi 2 dòng liền nhau thành 1** trên bảng BCTC khít của SEC 10-K
(line pitch ~13pt < 15 < 2×13). Giải pháp ([camelot_extract.py](../../app/ingestion/camelot_extract.py)
`_page_row_tols`/`_read_tables`): dò **line pitch từng trang** (median khoảng cách dòng qua
pdfplumber), đặt `row_tol = round(0.6 × pitch)` clamp [2,20], gom trang theo bucket tol để
chỉ gọi camelot vài lần. Bật bằng `CAMELOT_ROW_TOL_MODE=adaptive` (mặc định repo vẫn `fixed`
để giữ nguyên config thắng của Vinamilk). Tesla → tol=8 cho p50/51/54; ingest 864 bảng /
**1419 chunks** (fixed: 667 / 1222) — dòng tách đúng thay vì dính.

| Chỉ số | v4.2-tsla (fixed row_tol=15) | **v4.3-tsla-adaptive** | Δ |
|---|---|---|---|
| Retrieval hit-rate@k | 25.8% (8/31) | **22.6%** (7/31) | −3.2pp (nhiễu page-hit) |
| Numeric accuracy | 20.8% (5/24) | **41.7%** (10/24) | **+20.9pp** ⭐ (×2) |
| Judge accuracy | 42.4% (14/33) | **54.5%** (18/33) | **+12.1pp** ⭐ |
| Âm-tính | 50.0% | **50.0%** | = |

**Kết luận: đúng như chẩn đoán — sửa extraction cứu thẳng khâu ĐỌC SỐ, không đụng retrieval.**
Numeric **gấp đôi**; **income_statement judge 20% → 70%** (LLM giờ đọc được đúng dòng doanh
thu/chi phí thay vì chuỗi 2 số dính nhau). Judge flips: **+7 BAD→OK** (Q15,16,17,19,22
income_statement + Q30 reasoning + Q32 âm-tính) −3 OK→BAD (Q29,31 reasoning + Q33 âm-tính,
n nhỏ = nhiễu) = **net +4**.

**Nút thắt còn lại đã tách bạch rõ = RANKING, không phải extraction.** Q6 (total assets
106,618) và Q7 (total current assets 49,616) **vẫn MISS** vì trang 50 không lọt top-5:
cùng con số xuất hiện ở MD&A (p34, p39–46) lấn át bảng BCTC chính. `balance_sheet` retrieval
vẫn 22%. Đây là bệnh trùng-lặp-số của 10-K dài, cần section-aware boost / khử trùng lặp —
tách hẳn khỏi lỗi extraction vừa sửa. (Retrieval −3.2pp chỉ là 1 câu, trong nhiễu page-hit.)

**Đề xuất:** ~~cân nhắc bật adaptive mặc định~~ → **ĐÃ kiểm tra, xem `v4.3-vnm-adaptive`:
adaptive HỒI QUY Vinamilk −10pp mọi mặt** (phá nhãn wrap của VAS). Kết luận: giữ mặc định
`fixed`, bật `adaptive` **theo tài liệu** (10-K → adaptive; VAS → fixed). Bước tiếp cho
Tesla: section-aware retrieval boost (ưu tiên "Consolidated Statements of…").

---

### `v4.2-tsla` — Chạy config tốt nhất (v4.2) trên tài liệu MỚI: Tesla 2023 10-K (2026-09-02)

Đánh giá **khả năng khái quát hoá** của config tốt nhất (v4.2: `camelot_rows` +
`hybrid_rerank` + `qwen2:7b`, k=5) sang **tài liệu chưa từng tune**: Tesla, Inc. Form 10-K
FY2023 — SEC filing tiếng Anh, native text-layer, **130 trang** (gấp đôi Vinamilk 66 trang).
Ingest: 130 trang native (0 OCR), 667 bảng, **1222 chunks**, ~27s. Bộ test: 33 câu
([qa_ground_truth_tsla.json](qa_ground_truth_tsla.json)). Report thô:
[results/eval_report_v4.2-tsla_20260901_215536.md](results/eval_report_v4.2-tsla_20260901_215536.md).

| Chỉ số | v4.2 (VNM) | **v4.2-tsla (Tesla)** | Δ |
|---|---|---|---|
| Retrieval hit-rate@k | 80.8% | **25.8%** (8/31) | **−55.0pp** ↓↓ |
| Numeric accuracy | 80.0% | **20.8%** (5/24) | **−59.2pp** ↓↓ |
| Judge accuracy | 89.3% | **42.4%** (14/33) | **−46.9pp** ↓↓ |
| Âm-tính | 100% | **50.0%** (1/2) | −50pp ↓ |

**Kết luận: config tune trên Vinamilk KHÔNG khái quát sang 10-K lớn — sập toàn diện.**
Nút thắt là **XẾP HẠNG retrieval trên tài liệu dài**, KHÔNG phải trích xuất.

Theo category (retr / judge, n):
`income_statement 0% / 20% (n=10)` · `balance_sheet 33% / 22% (n=9)` ·
`cash_flow 25% / 75% (n=4)` · `company_profile 67% / 67% (n=3)` ·
`reasoning 0% / 67% (n=3)` · `operational 100% / 100% (n=2)` · `negative 50% (n=2)`.

**Chẩn đoán gốc (đã kiểm chứng bằng scroll Qdrant):** các trang báo cáo tài chính hợp nhất
**CÓ chunk** trong store (p50=18, p51=18, p54=38 chunks) → **extraction OK**. Nhưng retriever
xếp hạng nhầm: **p51 (income statement) KHÔNG BAO GIỜ vào top-5** cho cả 10 câu
income_statement; thay vào đó kéo các trang **MD&A "Results of Operations" (p34, p39–46)** và
**thuyết minh (p52, p63, p72–90)** — nơi CÙNG con số doanh thu/lợi nhuận được lặp lại dưới
dạng bảng/narrative khác. Đây là **bệnh đặc thù 10-K**: mỗi chỉ tiêu tài chính xuất hiện
2–3 nơi (MD&A tóm tắt + BCTC chính + notes), 1222 chunk đồng dạng làm cross-encoder không
phân biệt được "bảng BCTC chính" với "bảng thảo luận". Vinamilk (66 trang, ít trùng lặp)
không phơi bày lỗi này.

**Điểm sáng (giữ được):** `operational` 100% (sản lượng xe, headcount — text đặc thù,
không trùng), `cash_flow` judge 75% (dù retr thấp, số liệu đủ đặc trưng để LLM trả đúng).
**Âm-tính rớt còn 50%:** Q32 bịa `$136.9 triệu` cho câu bẫy (10-K dài → nhiều số tangential
lọt top-5, tái hiện đúng bệnh k lớn ở v3.1/v3.2).

**Hướng cải thiện (Tesla/10-K):** (a) **section-aware boost** — ưu tiên chunk thuộc
"Consolidated Statements of ..." hơn "Management's Discussion"/"Notes" khi câu hỏi nhắm BCTC
chính; (b) khử trùng lặp con số theo section trước khi rerank; (c) query rewriting gắn tên
statement ("consolidated balance sheet"); (d) `expected_pages` của bộ test hiện chỉ chấp
nhận trang BCTC chính — cân nhắc thêm trang MD&A tương đương để metric page-hit công bằng
hơn (một phần lỗi là artifact page-hit, nhưng numeric 20.8% cho thấy lỗi đọc số là THẬT).

---

### `v4.2` ⭐ TỐT NHẤT — siết prompt (ưu tiên dòng Code) + làm sạch chunk dị dạng (2026-09-01)

Nhắm khâu LLM ĐỌC (4/6 câu BAD của v4.1 có gold ở rank 1-2 nhưng LLM đọc sai). Hai phần:
- **A. Siết prompt** ([financial_agent.py](../../app/agent/financial_agent.py) luật 6-9):
  ưu tiên dòng nhãn khớp + có `[Code]`, bỏ dòng "Segment"/note; không abstain khi có dòng
  khớp; thực thể báo cáo = công ty ở tiêu đề (không lấy tên công ty con); một con số, không
  breakdown.
- **B. Làm sạch chunk camelot** ([camelot_extract.py](../../app/ingestion/camelot_extract.py)):
  B1 `_split_label_numbers` tách số bị nhét vào label (p37 "Short-term investments ...1,193...
  in shares" → nhãn sạch); B2 `_compose_segment_header` gắn nhãn nhóm + tag **"total"** cho
  bảng segment header lặp (p65 → "total 2023 VND = 24,544,731,615,410").

| Chỉ số | v3.3 | v4.1 | v4.2a (prompt) | **v4.2 (prompt+chunk)** |
|---|---|---|---|---|
| Retrieval hit-rate@k | 65.4% | 80.8% | 80.8% | **80.8%** |
| Numeric accuracy | 70.0% | 75.0% | 80.0% | **80.0%** |
| Judge accuracy | 85.7% | 78.6% | 85.7% | **89.3%** ⭐ |
| Âm-tính | 100% | 100% | 100% | **100%** |

**WIN TOÀN DIỆN — tốt nhất mọi chỉ số.** Tách biến rõ: **prompt (A)** cứu Q17 (bỏ số
segment), Q25 (hết bịa quy trình tính) → judge 78.6→85.7, numeric 75→80. **Chunk cleaning
(B)** cứu thêm Q9 (hết ghép breakdown từ chunk p37) → judge 85.7→**89.3**. So v3.3:
**judge +3.6pp, numeric +10pp, retrieval +15.4pp**, âm-tính giữ 100%.

**Còn 3 câu BAD:** Q1 (LLM né tránh, không trích tên cty dù ở rank 1), Q5 (LLM nói "not
provided" dù dòng TOTAL ASSETS ở rank 2) — **giới hạn năng lực qwen2:7b**, chunk/retrieval
đã đúng; Q24 (retrieval/rerank miss, paid vs received). Hướng tiếp: thử model mạnh hơn
(qwen2.5:7b-instruct / qwen3:8b) cho Q1/Q5; hybrid keyword riêng cho Q24.

---

### `v4.1` — Hybrid làm mặc định + BM25 lọc stopword (2026-09-01)

Hai thay đổi so với v4: (1) **`RETRIEVAL_MODE=hybrid_rerank` thành MẶC ĐỊNH**; (2) **BM25
lọc stopword** ([retrieval.py](../../app/rag/retrieval.py) `_tok_filtered`): bỏ stopword
tiếng Anh + boilerplate lặp (document-frequency >25%: vinamilk/consolidated/vnd/2023…) +
số lẻ. Query "total assets as at 31 December 2023" rút còn `total assets`.

| Chỉ số | v3.3 | v4 | **v4.1** |
|---|---|---|---|
| Retrieval hit-rate@k | 65.4% | 80.8% | **80.8%** |
| Numeric accuracy | 70.0% | 75.0% | **75.0%** |
| Judge accuracy | **85.7%** | 78.6% | **78.6%** |
| Âm-tính | 100% | 100% | **100%** |

**Stopword fix cứu retrieval Q5:** "TOTAL ASSETS [Code 270]" từ BM25 rank 32→1, vào final
top-5 (rank 2). balance_sheet retrieval 88.9→**100%**. NHƯNG **0 judge flip** — LLM vẫn
trả "not provided" dù dòng đúng ở rank 2.

**PHÁT HIỆN CHỐT — nút thắt đã dịch hẳn sang khâu LLM đọc, không còn ở retrieval.**
Trong 6 câu BAD, **4 câu gold đã ở top-1/2** nhưng LLM đọc sai:
- Q1 (gold p4 rank 1) → LLM chọn tên công ty CON "Viet Nam Sugar JSC".
- Q5 (gold p7 rank 2) → LLM nói "không có trong tài liệu".
- Q9 (gold p6 Code 120 rank 2) → LLM ghép breakdown thừa từ chunk p37.
- Q17 (gold p10 Code 20 rank 1) → LLM lấy số segment 20.8T từ chunk p65 dị dạng.

Chỉ còn 2 câu là retrieval-miss thật: Q24 (cross-encoder dìm dòng cash-flow terse dưới
note "dividends received"), Q25 (reasoning, thiếu p10).

**Đánh đổi vs v3.3:** retrieval +15pp, numeric +5pp, judge −7pp. judge của v3.3 cao hơn
một phần do 2 ca chấm lỏng (Q5/Q25 abstain judged OK) — v4.1 phơi bày đúng lỗ hổng thật.
**hybrid là NỀN TỐT HƠN để xây tiếp**: dense không bao giờ đưa Q5 tới LLM (chunk ngoài
top-60), hybrid đưa tới rank 2 → mọi fix prompt/model tiếp theo mới có tác dụng.

**Hướng v4.2 (nút thắt LLM-read):** (a) siết prompt buộc trích dòng có nhãn KHỚP CHÍNH XÁC
+ có Code, bỏ dòng segment/note; (b) làm sạch chunk camelot dị dạng (p65 "Segment gross
profit" 6 số, p37 "Short-term investments in shares"); (c) thử model mạnh hơn (qwen2.5:7b).
Config: `RETRIEVAL_MODE=hybrid_rerank` (mặc định), `DENSE_CANDIDATES=25`, `BM25_CANDIDATES=15`.

---

### `v4` — Hybrid (dense + BM25) + cross-encoder rerank trước top-k (2026-09-01)

Thêm pipeline truy hồi mới ([app/rag/retrieval.py](../../app/rag/retrieval.py)): hợp
**dense top-25 ∪ BM25 top-15** rồi **rerank bằng `BAAI/bge-reranker-v2-m3`** → top-k=5.
BM25 in-memory (scroll toàn bộ chunk/doc, cache) — không re-ingest. Bật bằng
`RETRIEVAL_MODE=hybrid_rerank` (mặc định repo vẫn `dense`). Dùng chung cho cả tool lẫn eval.

| Chỉ số | v3.3 | **v4** | Δ vs v3.3 |
|---|---|---|---|
| Retrieval hit-rate@k | 65.4% | **80.8%** | **+15.4pp** |
| Numeric accuracy | 70.0% | **75.0%** | **+5.0pp** |
| Judge accuracy | 85.7% | **78.6%** | **−7.1pp** |
| Âm-tính | 100% | **100%** | = |

**Kết luận: đánh đổi, CHƯA lật mặc định.** Retrieval & numeric tăng mạnh (đúng mục tiêu
truy hồi mịn hơn) nhưng judge giảm. Mổ 6 câu flip judge:
- **Fix thật (BAD→OK): Q2** (auditor KPMG p5 lên rank 1 nhờ rerank), **Q6** (current assets lên rank 1).
- **Regress thật (OK→BAD): Q1** — rerank đa dạng hoá top-5, kéo tên công ty CON "Viet Nam
  Sugar JSC" vào ngữ cảnh → LLM chọn nhầm công ty. **Q17** — rerank đẩy income statement
  (p10) lên rank 1 thay vì note p65 sạch → LLM đọc nhầm dòng gross profit (20.8T thay 24.5T).
- **KHÔNG phải regress (artifact judge): Q5, Q25** — retrieval MISS ở CẢ hai phiên bản.
  v3.3 model "không tìm thấy" và judge chấm lỏng = OK (dù GT có số); v4 model thử đoán → sai.
  Không do rerank; chỉ là judge lệch chuẩn trên ca cả hai đều miss.

Trừ 2 artifact: **regress thật (Q1,Q17) = fix thật (Q2,Q6)** → judge cân bằng, trong khi
retrieval +15pp, numeric +5pp. Q9 gold lên rank 2 (top-5) nhưng LLM vẫn chọn nhầm; Q24 gold
p13 dense rank>50, BM25 rank 30, cross-encoder xếp rank 9 → vẫn miss (va chạm "paid" vs
"received/distributed", `statement` metadata p13 rỗng nên không enrich được).

**Còn lại để v4.x xử lý:** (a) Q1/Q17 — rerank chọn trang "đúng hơn" nhưng LLM đọc nhầm
entity/dòng → cần chống nhiễu ở khâu đọc (rerank giữ chunk tín hiệu mạnh, hoặc fuse điểm
dense) ; (b) Q24 — cross-encoder không phân biệt paid/received.

Config: `RETRIEVAL_MODE=hybrid_rerank`, `RERANK_MODEL=BAAI/bge-reranker-v2-m3`,
`DENSE_CANDIDATES=25`, `BM25_CANDIDATES=15`. Mặc định repo giữ `dense` cho tới khi judge ≥ v3.3.

---

### `v3.3` ⭐ TỐT NHẤT (judge) — đặt tên dòng tổng + bỏ boilerplate + prompt (2026-09-01)

Tinh chỉnh v3 dựa trên phân tích 6 ca sai (v3_failures.json / v3_gold.json):
1. **Đặt tên dòng TỔNG**: Camelot xuống dòng tách tên subtotal ("Current assets")
   khỏi phần số ("(100=...)"). Gộp lại → nhãn `Current assets (100 = 110 + ...)`
   ([camelot_extract.py](../../app/ingestion/camelot_extract.py) pre-continuation merge).
2. **Bỏ lặp boilerplate**: statement-title + section rời khỏi text nhúng (đưa vào
   metadata) → chunk chi tiết gọn, vector phân biệt mạnh hơn (hết "nhoè").
3. **Siết prompt nhẹ**: trả đúng số trong context, giữ dấu ngoặc = âm, không summarize.

| Chỉ số | v0 | v3 | **v3.3** | Δ vs v0 |
|---|---|---|---|---|
| Retrieval hit-rate@k | 65.4% | 53.8% | **65.4%** | = |
| Numeric accuracy | 50.0% | 60.0% | **70.0%** | **+20pp** |
| Judge accuracy | 78.6% | 78.6% | **85.7%** | **+7.1pp** |
| Âm-tính | 100% | 100% | **100%** | = |

**Kết luận: WIN toàn diện.** Numeric & judge cao nhất mọi phiên bản, retrieval bằng
v0, âm-tính giữ 100%. Judge flips v0→v3.3: **+5 (Q10, Q12, Q13, Q17, Q21)** −3 (Q2, Q6,
Q9) = net +2. Còn 4 ca BAD: Q2 (auditor-narrative), Q6/Q9 (dòng bị note đè), Q24 (dividends).

**Bài học quan trọng — k:** thử `k=12` (v3.1/v3.2) để cứu Q8/Q10/Q12 nhưng **phản tác
dụng**: câu âm-tính (share price / kế hoạch 2024) bị 12 chunk số liệu tangential làm
model **bịa số** → âm-tính 0%, judge 67.9%. **k=5 là tối ưu** (judge 85.7%, âm-tính
100%). Numeric 70% đến từ CHUNK tốt hơn, KHÔNG phải k. → giữ `RAG_TOP_K=5`.

Config thắng cuộc (đã set mặc định): `TABLE_EXTRACTOR=camelot_rows`, `RAG_TOP_K=5`,
prompt siết nhẹ. (`TABLE_EXTRACTOR` mặc định repo vẫn `pdfplumber`; camelot_rows opt-in.)

---

### `v3-camelot-rows` — MỘT chunk nhỏ mỗi dòng + header năm (2026-08-31)

Trích bảng bằng Camelot (stream tuned) nhưng emit **một `Document` NHỎ cho mỗi DÒNG
dữ liệu**, mỗi chunk là câu tự nhiên mang sẵn header năm căn cột:
`Gross profit [Code 20] — consolidated income statement (page 10): 2023 VND = 24,544,731,615,410; 2022 VND = 23,897,231,506,707.`
Gate `TABLE_EXTRACTOR=camelot_rows` ([camelot_extract.py](../../app/ingestion/camelot_extract.py) `extract_all_row_chunks`). 589 row-chunks; coverage precheck 19/19 primaries đúng trang.

| Chỉ số | v0 | v1 | v2 | **v3** |
|---|---|---|---|---|
| Retrieval hit-rate@k | **65.4%** | 50.0% | 15.4% | 53.8% |
| Numeric accuracy | 50.0% | 55.0% | 35.0% | **60.0%** ⬆ |
| Judge accuracy | 78.6% | 78.6% | 67.9% | **78.6%** |
| Âm-tính | 100% | 100% | 100% | 100% |

**Kết luận: tốt nhất về chất lượng trả lời** — numeric **60% (cao nhất)**, judge
78.6% (ngang cao nhất), vượt xa v1/v2. Judge flips v0→v3: **+4 (Q13, Q17, Q21, Q24 —
đúng các mục tiêu đọc-lệch-cột + loose-row)**, −4 (Q6, Q8, Q20, Q23 — vài dòng Camelot
trích số chưa sạch) → net judge 0.

**Còn hạn chế:** retrieval hit-rate (page-hit) 53.8% **chưa về mức v0 (65.4%)** dù
chunk đã nhỏ (589 chunk). Nghi vấn: mỗi câu NL lặp tiền tố statement-title + section
giống nhau trên mọi dòng cùng báo cáo → embedding bớt phân biệt giữa các dòng; nhiều
chunk "cash/inventory" ở trang thuyết minh cạnh tranh top-5. Hướng tinh chỉnh v3.x:
rút gọn tiền tố, đưa nhãn+số lên đầu mạnh hơn; sửa trích số cho Q6/Q8/Q20/Q23.
`TABLE_EXTRACTOR` mặc định vẫn `pdfplumber`; camelot_rows là tuỳ chọn.

---

### `v2-camelot` — Native tables via Camelot stream (2026-08-31)

Bảng trang native lấy từ **Camelot stream** (`edge_tol=500, row_tol=15` — đã tinh
chỉnh: cột Code/Note tách riêng, header năm căn cột, full recall, accuracy tự báo
98.9%), narrative vẫn từ pdfplumber, scan→OCR. Gate bằng `TABLE_EXTRACTOR=camelot`
trong [config](../../app/config.py) ([camelot_extract.py](../../app/ingestion/camelot_extract.py)).

| Chỉ số | v1 | **v2-camelot** | Δ vs v1 |
|---|---|---|---|
| Retrieval hit-rate@k | 50.0% | **15.4%** | **−34.6pp** ↓↓ |
| Numeric accuracy | 55.0% | **35.0%** | −20.0pp ↓ |
| Judge accuracy | 78.6% | **67.9%** | −10.7pp ↓ |
| Âm-tính | 100% | 100% | 0 |

**Kết luận: TỆ HƠN rõ rệt — loại bỏ.** Judge flips v1→v2: +1 (Q15) nhưng −4 (Q2,
Q10, Q22, Q23). balance_sheet retrieval **0%**.

**Nguyên nhân (bài học quan trọng):** Camelot cho **chất lượng BẢNG tốt nhất** (đọc
dễ) nhưng gộp mỗi trang thành **MỘT bảng lớn ~30 dòng** → embedding "nhoè" nặng →
truy vấn theo dòng lẻ (hàng tồn kho, tiền, cổ tức...) không match đúng trang →
retrieval sập còn 15%. Đây là **cùng bệnh v1 nhưng nặng hơn**: chất lượng trích
bảng cao KHÔNG cứu được end-to-end khi chunk quá thô.

**Điểm nghẽn thật = ĐỘ MỊN RETRIEVAL, không phải chất lượng trích bảng.** Hướng
đúng tiếp theo: chunk bảng theo **từng dòng/chỉ tiêu** (giữ retrieval mịn) + gắn
header năm vào mỗi chunk nhỏ — thay vì cải thiện trình trích bảng. (Mặc định
`TABLE_EXTRACTOR=pdfplumber` giữ nguyên; camelot chỉ là tuỳ chọn.)

---

### `v1` — Merge table fragments + recover year-header (2026-08-31)

Thay đổi ở [`_extract_text_page`](../../app/ingestion/pdf_ingestion.py): giữ
`table_settings` mặc định (cột sạch), **gộp các mảnh bảng cùng schema trên một
trang** thành 1 bảng logic, **khôi phục dải header năm** (`31/12/2023 1/1/2023 /
VND VND`) ngay trên bảng, drop "blob" mis-extraction, và **giữ loose rows**
(Goodwill/Share capital/Dividends) ở narrative để không mất recall.

| Chỉ số | v0 | **v1** | Δ |
|---|---|---|---|
| Retrieval hit-rate@k | 65.4% | **50.0%** | **−15.4pp** ↓ |
| Numeric accuracy | 50.0% | **55.0%** | +5.0pp ↑ |
| Judge accuracy | 78.6% | **78.6%** | **0** |
| Âm-tính | 100% | 100% | 0 |

**Kết luận: lateral move, chưa phải win.** Judge flips triệt tiêu nhau:
- ✅ Sửa (BAD→OK): **Q10, Q13, Q17** (3 mục tiêu đọc-lệch-cột) + Q24 = **+4**
- ❌ Regress (OK→BAD): **Q8, Q9, Q11, Q15** = **−4**

**Nguyên nhân:** merge gộp mỗi trang thành 1 bảng lớn → embedding "nhoè" → các
dòng lẻ (hàng tồn kho, đầu tư ngắn hạn, vốn chủ sở hữu) không còn retrieve trúng
trang (balance_sheet retr 77.8%→22.2%). Header năm sửa được lỗi đọc cột, nhưng
việc merge hi sinh độ mịn retrieval — hai hiệu ứng bù trừ.

**Bài học → v2:** cái sửa lỗi là **header năm**, không phải **merge**. Giữ chunk
nhỏ (mịn) + prepend header năm vào từng fragment (không gộp bảng lớn) để lấy
best-of-both.

---

### `v0` — Baseline (2026-08-31)

Cấu hình gốc: `LLM_MODEL=qwen2:7b`, `EMBEDDING_MODEL=nomic-embed-text`, k=5,
chunk 1000/overlap 200, 1 tool RAG, không rerank. Chưa có thay đổi cải thiện nào.

**Tài liệu:** Vinamilk FY2023 (native/pdfplumber) — `vinamilk_2023_consolidated_financial_statements.pdf`
Ingest: 66 trang (native 65, ocr 1), 143 bảng, 151 chunks, ~18s.

| Chỉ số | Kết quả |
|---|---|
| Retrieval hit-rate@k | **65.4%** (17/26), mean recall 63.5% |
| Numeric accuracy | **50.0%** (10/20) |
| LLM-as-judge accuracy | **78.6%** (22/28) |
| Âm-tính (không bịa số) | **100%** (2/2) |

Theo category:

| Category | n | Retrieval | Judge |
|---|---|---|---|
| balance_sheet | 9 | 77.8% | 66.7% |
| income_statement | 8 | 75.0% | 75.0% |
| cash_flow | 3 | 66.7% | 66.7% |
| company_profile | 4 | 25.0% | 100% |
| reasoning | 2 | 50.0% | 100% |
| negative_not_found | 2 | — | 100% |

**Các câu sai / cần cải thiện (v0):**

| Q | Category | Retr | Num | Judge | Nhận định |
|---|---|---|---|---|---|
| Q10 | balance_sheet | HIT | MISS | BAD | Tổng nợ phải trả — truy hồi trúng trang nhưng đọc sai/lệch số |
| Q12 | balance_sheet | MISS | MISS | BAD | Vốn góp chủ sở hữu — không truy hồi được đúng bảng |
| Q13 | balance_sheet | HIT | MISS | BAD | Goodwill — trúng trang, sai số |
| Q17 | income_statement | HIT | MISS | BAD | Lợi nhuận gộp — trúng trang, sai số |
| Q21 | income_statement | MISS | MISS | BAD | Chi phí bán hàng — miss cả truy hồi lẫn số |
| Q24 | cash_flow | MISS | MISS | BAD | Cổ tức đã chi — miss cả truy hồi lẫn số |
| Q5, Q8, Q9, Q15 | table | mixed | MISS | OK | Đọc lệch con số dù judge vẫn chấp nhận |
| Q1, Q3, Q4 | company_profile | MISS | n/a | OK | Page-hit miss nhưng agent vẫn trả đúng (thông tin lặp nhiều trang) |

**Đọc hiểu:**
- Điểm nghẽn chính là **numeric 50%**: agent thường xuyên đọc lệch con số ở các bảng
  cân đối kế toán / KQKD, kể cả khi retrieval đã trúng trang (Q10, Q13, Q17). → hướng
  cải thiện nằm ở prompt/định dạng bảng Markdown và khả năng đọc bảng của LLM, không
  chỉ ở retrieval.
- **company_profile retrieval 25%** là hạn chế của metric page-hit (thông tin lặp ở
  nhiều trang) chứ không phải lỗi trả lời — judge 100%.
- **Âm-tính 100%**: hệ thống không bịa số cho câu bẫy — tốt.

**Hướng cải thiện đề xuất cho v1+:**
1. Tăng chất lượng đọc số từ bảng: cải thiện prompt tool/agent yêu cầu trích đúng
   dòng + cột năm; hoặc thử model mạnh hơn (`qwen2.5:7b-instruct`, `qwen3:8b`).
2. Retrieval: thử tăng k, thêm rerank, hoặc query rewriting cho câu bảng.
3. Đánh giá lại numeric matcher với dung sai làm tròn nếu muốn đo "gần đúng".

---

## Mẫu mục cho lần cải thiện tiếp theo

```
### `vX` — <tên thay đổi> (YYYY-MM-DD)

Thay đổi so với phiên bản trước: <mô tả>

| Chỉ số | vX | Δ so với trước |
|---|---|---|
| Retrieval hit-rate@k | ... | ... |
| Numeric accuracy | ... | ... |
| Judge accuracy | ... | ... |
| Âm-tính | ... | ... |
```
