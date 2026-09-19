"""Eval runner chấm tự động cho bộ QA ground-truth (VNM native + DHG scan).

Luồng cho MỖI file ground-truth:
  1. Ingest source_document vào Qdrant với 1 document_id tất định (deterministic)
     để có thể cô lập tài liệu khi truy hồi và khi hỏi agent.
  2. Với mỗi câu hỏi:
       - Retrieval hit: similarity_search(k) có filter theo document_id, đối chiếu
         trang truy hồi với `expected_pages` (page recall@k).
       - Đối chiếu số: rút con số chính từ `ground_truth` và kiểm tra câu trả lời
         của agent có chứa đúng con số đó không (chuẩn hoá dấu . , khoảng trắng).
       - LLM-as-judge: nhờ LLM chấm đúng/sai so với ground_truth (xử lý được câu
         suy luận % và câu âm-tính "không tìm thấy").
  3. Xuất báo cáo điểm: JSON + Markdown + bảng tóm tắt ra console.

Chạy từ thư mục gốc:
    python -m app.eval.run_eval
    python -m app.eval.run_eval --groundtruth data/eval/qa_ground_truth_vnm.json
    python -m app.eval.run_eval --skip-ingest --limit 5
    python -m app.eval.run_eval --no-judge          # chỉ retrieval + đối chiếu số
"""

import argparse
import glob
import hashlib
import json
import os
import re
import sys
import time
import traceback
from datetime import datetime

from app.config import JUDGE_LLM_MODEL, LLM_MODEL, UPLOAD_DIR
from app.ingestion.pdf_ingestion import ingest_pdf
from app.rag.retrieval import retrieve
from app.rag.vector_store import (
    build_document_filter,
    get_client,
    get_vector_store,
)

# Danh mục câu không có con số "chính" để đối chiếu -> chỉ dựa vào judge.
_NON_NUMERIC_CATEGORIES = {"reasoning", "suy_luan_tinh_toan"}
_NEGATIVE_CATEGORIES = {"negative_not_found", "am_tinh_khong_co"}


# --------------------------------------------------------------------------- #
# Tiện ích số học
# --------------------------------------------------------------------------- #

# Một chuỗi số: chữ số đầu, rồi các nhóm chữ số ngăn bởi 1 dấu . hoặc , .
_NUM_RE = re.compile(r"\d(?:[.,]?\d)*")


def _normalize_number(token: str) -> str:
    """Bỏ mọi dấu phân tách nghìn/thập phân -> chuỗi chữ số thuần."""
    return re.sub(r"[.,\s]", "", token)


def extract_numbers(text: str) -> list[str]:
    """Trả về danh sách chuỗi-chữ-số đã chuẩn hoá xuất hiện trong text."""
    return [_normalize_number(m) for m in _NUM_RE.findall(text or "")]


def expected_numbers_from_truth(ground_truth: str) -> list[str]:
    """Rút các con số 'đáng kể' (>= 4 chữ số) từ ground_truth, giữ nguyên thứ tự.

    Con số CHÍNH (đáp án năm hiện tại) là phần tử đầu tiên: ground_truth luôn
    viết số năm nay trước, số năm trước trong ngoặc.

    Bỏ các token là NĂM 4 chữ số (19xx/20xx) để không nhầm mốc thời gian trong
    câu văn bản (vd "năm 2023") thành con số tài chính cần đối chiếu.
    """
    nums = [
        n for n in extract_numbers(ground_truth)
        if len(n) >= 4 and not re.fullmatch(r"(19|20)\d\d", n)
    ]
    return nums


# --------------------------------------------------------------------------- #
# document_id tất định theo tên file (để --skip-ingest tái dùng được)
# --------------------------------------------------------------------------- #

def deterministic_document_id(filename: str) -> str:
    digest = hashlib.sha1(filename.encode("utf-8")).hexdigest()[:12]
    return f"eval-{digest}"


def resolve_pdf_path(source_document: str, uploads_dir: str) -> str | None:
    """Tìm file PDF trong uploads_dir (khớp basename)."""
    direct = os.path.join(uploads_dir, source_document)
    if os.path.isfile(direct):
        return direct
    if os.path.isfile(source_document):
        return source_document
    # thử khớp basename lỏng
    base = os.path.basename(source_document)
    for cand in glob.glob(os.path.join(uploads_dir, "*.pdf")):
        if os.path.basename(cand) == base:
            return cand
    return None


# --------------------------------------------------------------------------- #
# Ingest
# --------------------------------------------------------------------------- #

def ingest_document(pdf_path: str, document_id: str) -> dict:
    """Xoá điểm cũ của document_id rồi ingest lại (tránh chunk trùng khi re-run)."""
    try:
        client = get_client()
        client.delete(
            collection_name=os.getenv("QDRANT_COLLECTION", "financial_reports"),
            points_selector=build_document_filter(document_id),
        )
    except Exception:
        # Collection có thể chưa tồn tại -> bỏ qua, ingest sẽ tạo.
        pass
    return ingest_pdf(pdf_path, document_id=document_id)


# --------------------------------------------------------------------------- #
# Retrieval
# --------------------------------------------------------------------------- #

def evaluate_retrieval(vector_store, question: str, document_id: str, k: int) -> dict:
    # Dùng cùng pipeline truy hồi mà tool/production chạy (dense | hybrid_rerank).
    docs = retrieve(question, document_id, k)
    retrieved_pages = []
    retrieved = []
    for rank, d in enumerate(docs, start=1):
        page = d.metadata.get("page")
        retrieved_pages.append(page)
        retrieved.append(
            {
                "rank": rank,
                "page": page,
                "content_type": d.metadata.get("content_type"),
                "source": d.metadata.get("source"),
            }
        )
    return {"retrieved_pages": retrieved_pages, "retrieved": retrieved}


def score_retrieval(retrieved_pages: list, expected_pages: list) -> dict:
    """page recall@k. expected rỗng (câu âm-tính) -> không tính (None)."""
    if not expected_pages:
        return {"applicable": False, "hit": None, "recall": None, "first_hit_rank": None}
    retset = set(p for p in retrieved_pages if p is not None)
    matched = [p for p in expected_pages if p in retset]
    first_hit_rank = None
    for rank, p in enumerate(retrieved_pages, start=1):
        if p in set(expected_pages):
            first_hit_rank = rank
            break
    return {
        "applicable": True,
        "hit": len(matched) > 0,
        "recall": len(matched) / len(expected_pages),
        "matched_pages": matched,
        "first_hit_rank": first_hit_rank,
    }


# --------------------------------------------------------------------------- #
# Đối chiếu số
# --------------------------------------------------------------------------- #

def score_numeric(question_obj: dict, answer: str) -> dict:
    category = question_obj.get("category", "")
    if category in _NON_NUMERIC_CATEGORIES or category in _NEGATIVE_CATEGORIES:
        return {"applicable": False, "primary": None, "primary_found": None}

    expected = expected_numbers_from_truth(question_obj.get("ground_truth", ""))
    if not expected:
        return {"applicable": False, "primary": None, "primary_found": None}

    answer_nums = set(extract_numbers(answer))
    primary = expected[0]
    any_found = [n for n in expected if n in answer_nums]
    return {
        "applicable": True,
        "primary": primary,
        "primary_found": primary in answer_nums,
        "any_expected_found": len(any_found) > 0,
        "expected_all": expected,
    }


# --------------------------------------------------------------------------- #
# LLM-as-judge
# --------------------------------------------------------------------------- #

_JUDGE_SYSTEM = """Bạn là giám khảo chấm điểm cho hệ thống hỏi-đáp báo cáo tài chính.
So sánh CÂU TRẢ LỜI của hệ thống với ĐÁP ÁN CHUẨN cho CÂU HỎI.

Nguyên tắc chấm:
- Đúng (correct=true) nếu câu trả lời truyền đạt CÙNG dữ kiện/con số cốt lõi như đáp án chuẩn.
  Chấp nhận khác biệt về định dạng số (dấu . , khoảng trắng, đơn vị VND) và cách diễn đạt.
- Với câu suy luận tỉ lệ %, chấp nhận sai số nhỏ (±1 điểm phần trăm) so với đáp án chuẩn.
- Với câu ÂM TÍNH (đáp án là "không tìm thấy"/"not found"): ĐÚNG chỉ khi hệ thống nói
  rõ thông tin không có/không được công bố và KHÔNG bịa ra con số. Nếu hệ thống bịa số -> Sai.
- Sai (correct=false) nếu con số/dữ kiện cốt lõi khác, thiếu, hoặc bịa đặt.

CHỈ trả về JSON một dòng đúng cú pháp, không thêm chữ nào khác:
{"correct": true hoặc false, "reason": "giải thích ngắn"}"""


def build_judge():
    from langchain_ollama import ChatOllama

    # JUDGE_LLM_MODEL (mặc định = LLM_MODEL) cho phép GHIM judge cố định khi so sánh
    # nhiều answer-model — xem app/config.py.
    return ChatOllama(model=JUDGE_LLM_MODEL, temperature=0)


def parse_judge_json(text: str) -> dict:
    # Lấy khối JSON đầu tiên trong output (LLM có thể lỡ thêm chữ).
    m = re.search(r"\{.*\}", text or "", re.DOTALL)
    raw = m.group(0) if m else (text or "")
    try:
        data = json.loads(raw)
        return {
            "correct": bool(data.get("correct")),
            "reason": str(data.get("reason", ""))[:500],
        }
    except Exception:
        low = (text or "").lower()
        guess = ("true" in low and "false" not in low) or "đúng" in low
        return {"correct": bool(guess), "reason": "parse_failed: " + (text or "")[:200]}


def score_judge(judge, question_obj: dict, answer: str) -> dict:
    prompt = (
        f"CÂU HỎI:\n{question_obj.get('question','')}\n\n"
        f"ĐÁP ÁN CHUẨN:\n{question_obj.get('ground_truth','')}\n\n"
        f"CÂU TRẢ LỜI CỦA HỆ THỐNG:\n{answer}\n"
    )
    try:
        resp = judge.invoke(
            [
                {"role": "system", "content": _JUDGE_SYSTEM},
                {"role": "user", "content": prompt},
            ]
        )
        verdict = parse_judge_json(resp.content)
        verdict["applicable"] = True
        return verdict
    except Exception as exc:
        return {"applicable": True, "correct": None, "reason": f"judge_error: {exc}"}


# --------------------------------------------------------------------------- #
# Agent
# --------------------------------------------------------------------------- #

def ask_agent(agent, question: str, document_id: str) -> dict:
    try:
        result = agent.invoke(
            {"messages": [{"role": "user", "content": question}]},
            config={"configurable": {"document_id": document_id}},
        )
        return {"answer": result["messages"][-1].content, "error": None}
    except Exception as exc:
        return {"answer": "", "error": f"{exc}"}


# --------------------------------------------------------------------------- #
# Chạy eval cho 1 file ground-truth
# --------------------------------------------------------------------------- #

def run_groundtruth_file(
    gt_path: str,
    uploads_dir: str,
    k: int,
    do_ingest: bool,
    do_agent: bool,
    judge,
    limit: int | None,
) -> dict:
    with open(gt_path, "r", encoding="utf-8") as f:
        gt = json.load(f)

    source_document = gt["source_document"]
    questions = gt["questions"]
    if limit:
        questions = questions[:limit]

    document_id = deterministic_document_id(source_document)
    pdf_path = resolve_pdf_path(source_document, uploads_dir)

    print(f"\n{'='*78}")
    print(f"GROUND-TRUTH: {os.path.basename(gt_path)}")
    print(f"  company        : {gt.get('company','')}")
    print(f"  source_document: {source_document}")
    print(f"  document_id    : {document_id}")
    print(f"  #questions     : {len(questions)}")
    print(f"{'='*78}")

    ingest_info = None
    if do_ingest:
        if not pdf_path:
            print(f"  [LỖI] Không tìm thấy PDF '{source_document}' trong {uploads_dir}. Bỏ qua ingest.")
        else:
            print(f"  Đang ingest {pdf_path} ... (PDF scan có thể mất vài phút do OCR)")
            t0 = time.time()
            ingest_info = ingest_document(pdf_path, document_id)
            ingest_info["seconds"] = round(time.time() - t0, 1)
            print(f"  Ingest xong trong {ingest_info['seconds']}s: {ingest_info}")
    else:
        print("  --skip-ingest: dùng lại dữ liệu đã có trong Qdrant.")

    vector_store = get_vector_store()

    agent = None
    if do_agent:
        from app.agent.financial_agent import create_financial_agent

        agent = create_financial_agent()

    results = []
    for i, q in enumerate(questions, start=1):
        expected_pages = q.get("expected_pages", []) or []

        retr = evaluate_retrieval(vector_store, q["question"], document_id, k)
        retr_score = score_retrieval(retr["retrieved_pages"], expected_pages)

        agent_out = {"answer": "", "error": "agent disabled"}
        num_score = {"applicable": False}
        judge_score = {"applicable": False}

        if do_agent:
            agent_out = ask_agent(agent, q["question"], document_id)
            num_score = score_numeric(q, agent_out["answer"])
            if judge is not None:
                judge_score = score_judge(judge, q, agent_out["answer"])

        row = {
            "id": q.get("id"),
            "category": q.get("category"),
            "content_type": q.get("content_type"),
            "question": q["question"],
            "ground_truth": q.get("ground_truth"),
            "expected_pages": expected_pages,
            "retrieved": retr["retrieved"],
            "retrieval": retr_score,
            "answer": agent_out["answer"],
            "agent_error": agent_out["error"],
            "numeric": num_score,
            "judge": judge_score,
        }
        results.append(row)
        _print_question_line(i, row)

    summary = summarize(results)
    return {
        "groundtruth_file": gt_path,
        "company": gt.get("company", ""),
        "source_document": source_document,
        "document_id": document_id,
        "ingest": ingest_info,
        "k": k,
        "summary": summary,
        "results": results,
    }


def _print_question_line(i: int, row: dict) -> None:
    r = row["retrieval"]
    if r["applicable"]:
        rflag = "HIT " if r["hit"] else "MISS"
        rtxt = f"retr={rflag}(r@{r['first_hit_rank']})" if r["hit"] else f"retr={rflag}"
    else:
        rtxt = "retr= n/a"

    n = row["numeric"]
    if n.get("applicable"):
        ntxt = "num=OK  " if n.get("primary_found") else "num=MISS"
    else:
        ntxt = "num= n/a"

    j = row["judge"]
    if j.get("applicable"):
        jc = j.get("correct")
        jtxt = "judge=OK  " if jc else ("judge=?? " if jc is None else "judge=BAD ")
    else:
        jtxt = "judge=n/a"

    print(
        f"  Q{row['id']:>2} [{(row['category'] or '')[:18]:<18}] "
        f"{rtxt:<16} {ntxt:<9} {jtxt}"
    )


# --------------------------------------------------------------------------- #
# Tổng hợp điểm
# --------------------------------------------------------------------------- #

def _rate(num: int, den: int):
    return None if den == 0 else round(num / den, 4)


def summarize(results: list[dict]) -> dict:
    retr_appl = [r for r in results if r["retrieval"]["applicable"]]
    retr_hits = sum(1 for r in retr_appl if r["retrieval"]["hit"])
    retr_recall_vals = [r["retrieval"]["recall"] for r in retr_appl]

    num_appl = [r for r in results if r["numeric"].get("applicable")]
    num_hits = sum(1 for r in num_appl if r["numeric"].get("primary_found"))

    judge_appl = [r for r in results if r["judge"].get("applicable") and r["judge"].get("correct") is not None]
    judge_hits = sum(1 for r in judge_appl if r["judge"].get("correct"))

    # Câu âm-tính: judge quyết định có đúng cách "từ chối" không.
    neg = [r for r in results if r["category"] in _NEGATIVE_CATEGORIES]
    neg_ok = sum(1 for r in neg if r["judge"].get("correct"))

    # Breakdown theo category (dùng judge làm thước đo đúng/sai).
    by_cat: dict = {}
    for r in results:
        c = r["category"] or "unknown"
        b = by_cat.setdefault(c, {"total": 0, "judge_ok": 0, "judge_scored": 0,
                                  "retr_appl": 0, "retr_hit": 0})
        b["total"] += 1
        if r["retrieval"]["applicable"]:
            b["retr_appl"] += 1
            b["retr_hit"] += 1 if r["retrieval"]["hit"] else 0
        if r["judge"].get("applicable") and r["judge"].get("correct") is not None:
            b["judge_scored"] += 1
            b["judge_ok"] += 1 if r["judge"].get("correct") else 0
    for c, b in by_cat.items():
        b["retrieval_hit_rate"] = _rate(b["retr_hit"], b["retr_appl"])
        b["judge_accuracy"] = _rate(b["judge_ok"], b["judge_scored"])

    return {
        "total_questions": len(results),
        "retrieval": {
            "applicable": len(retr_appl),
            "hits": retr_hits,
            "hit_rate": _rate(retr_hits, len(retr_appl)),
            "mean_recall": (round(sum(retr_recall_vals) / len(retr_recall_vals), 4)
                            if retr_recall_vals else None),
        },
        "numeric": {
            "applicable": len(num_appl),
            "hits": num_hits,
            "accuracy": _rate(num_hits, len(num_appl)),
        },
        "judge": {
            "scored": len(judge_appl),
            "correct": judge_hits,
            "accuracy": _rate(judge_hits, len(judge_appl)),
        },
        "negative_tests": {
            "total": len(neg),
            "handled_correctly": neg_ok,
            "accuracy": _rate(neg_ok, len(neg)),
        },
        "by_category": by_cat,
    }


def aggregate(all_docs: list[dict]) -> dict:
    flat = [r for d in all_docs for r in d["results"]]
    return summarize(flat)


# --------------------------------------------------------------------------- #
# Xuất báo cáo
# --------------------------------------------------------------------------- #

def _fmt_rate(x):
    return "  n/a" if x is None else f"{x*100:5.1f}%"


def print_summary_block(title: str, s: dict) -> None:
    print(f"\n{'-'*78}")
    print(f"TÓM TẮT: {title}")
    print(f"{'-'*78}")
    print(f"  Tổng số câu hỏi            : {s['total_questions']}")
    print(f"  Retrieval hit-rate (@k)    : {_fmt_rate(s['retrieval']['hit_rate'])}"
          f"  ({s['retrieval']['hits']}/{s['retrieval']['applicable']}), "
          f"mean recall {_fmt_rate(s['retrieval']['mean_recall'])}")
    print(f"  Đối chiếu số (numeric)     : {_fmt_rate(s['numeric']['accuracy'])}"
          f"  ({s['numeric']['hits']}/{s['numeric']['applicable']})")
    print(f"  LLM-as-judge accuracy      : {_fmt_rate(s['judge']['accuracy'])}"
          f"  ({s['judge']['correct']}/{s['judge']['scored']})")
    print(f"  Câu âm-tính (không bịa số) : {_fmt_rate(s['negative_tests']['accuracy'])}"
          f"  ({s['negative_tests']['handled_correctly']}/{s['negative_tests']['total']})")
    print("  Theo category:")
    for c, b in sorted(s["by_category"].items()):
        print(f"    - {c:<22} retr {_fmt_rate(b['retrieval_hit_rate'])}"
              f"  judge {_fmt_rate(b['judge_accuracy'])}  (n={b['total']})")


def write_markdown(report: dict, path: str) -> None:
    lines = []
    lines.append(f"# Báo cáo eval RAG tài chính — `{report.get('version','v0')}`")
    lines.append("")
    lines.append(f"- Phiên bản: **{report.get('version','v0')}**"
                 + (f" — {report['note']}" if report.get("note") else ""))
    lines.append(f"- Thời điểm: `{report['generated_at']}`")
    lines.append(f"- LLM model: `{report['llm_model']}`"
                 f"  |  Embedding: `{report.get('embedding_model','')}`"
                 f"  |  k = {report['k']}")
    lines.append("")
    ov = report["overall"]
    lines.append("## Tổng thể (gộp tất cả tài liệu)")
    lines.append("")
    lines.append("| Chỉ số | Giá trị | Chi tiết |")
    lines.append("|---|---|---|")
    lines.append(f"| Retrieval hit-rate@k | {_fmt_rate(ov['retrieval']['hit_rate']).strip()} | {ov['retrieval']['hits']}/{ov['retrieval']['applicable']} |")
    lines.append(f"| Numeric accuracy | {_fmt_rate(ov['numeric']['accuracy']).strip()} | {ov['numeric']['hits']}/{ov['numeric']['applicable']} |")
    lines.append(f"| Judge accuracy | {_fmt_rate(ov['judge']['accuracy']).strip()} | {ov['judge']['correct']}/{ov['judge']['scored']} |")
    lines.append(f"| Âm-tính đúng | {_fmt_rate(ov['negative_tests']['accuracy']).strip()} | {ov['negative_tests']['handled_correctly']}/{ov['negative_tests']['total']} |")
    lines.append("")

    for doc in report["documents"]:
        s = doc["summary"]
        lines.append(f"## {doc['company'] or doc['source_document']}")
        lines.append("")
        lines.append(f"- File ground-truth: `{os.path.basename(doc['groundtruth_file'])}`")
        lines.append(f"- Source document: `{doc['source_document']}`  |  document_id: `{doc['document_id']}`")
        if doc.get("ingest"):
            ing = doc["ingest"]
            lines.append(f"- Ingest: {ing.get('num_pages')} trang "
                         f"(native {ing.get('num_text_pages')}, ocr {ing.get('num_ocr_pages')}), "
                         f"{ing.get('num_chunks')} chunks, {ing.get('num_tables')} bảng.")
        lines.append("")
        lines.append(f"- Retrieval hit-rate@k: **{_fmt_rate(s['retrieval']['hit_rate']).strip()}** "
                     f"({s['retrieval']['hits']}/{s['retrieval']['applicable']})")
        lines.append(f"- Numeric accuracy: **{_fmt_rate(s['numeric']['accuracy']).strip()}** "
                     f"({s['numeric']['hits']}/{s['numeric']['applicable']})")
        lines.append(f"- Judge accuracy: **{_fmt_rate(s['judge']['accuracy']).strip()}** "
                     f"({s['judge']['correct']}/{s['judge']['scored']})")
        lines.append("")
        lines.append("| Q | Category | Exp.pages | Retr | Num | Judge | Ghi chú judge |")
        lines.append("|---|---|---|---|---|---|---|")
        for r in doc["results"]:
            rr = r["retrieval"]
            retr = "n/a" if not rr["applicable"] else ("HIT" if rr["hit"] else "MISS")
            nn = r["numeric"]
            num = "n/a" if not nn.get("applicable") else ("OK" if nn.get("primary_found") else "MISS")
            jj = r["judge"]
            if not jj.get("applicable") or jj.get("correct") is None:
                jud = "n/a"
            else:
                jud = "OK" if jj.get("correct") else "BAD"
            reason = (jj.get("reason", "") or "").replace("|", "/").replace("\n", " ")[:80]
            lines.append(f"| {r['id']} | {r['category']} | {r['expected_pages']} | {retr} | {num} | {jud} | {reason} |")
        lines.append("")

    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #

def main(argv=None):
    parser = argparse.ArgumentParser(description="Eval runner chấm tự động cho bộ QA ground-truth.")
    parser.add_argument(
        "--groundtruth", "-g", nargs="+",
        default=None,
        help="Danh sách file ground-truth JSON (mặc định: tất cả data/eval/qa_ground_truth_*.json).",
    )
    parser.add_argument("--uploads-dir", default=UPLOAD_DIR, help="Thư mục chứa PDF nguồn.")
    parser.add_argument("--k", type=int, default=5, help="Số chunk truy hồi (recall@k).")
    parser.add_argument("--skip-ingest", action="store_true",
                        help="Không ingest lại (tái dùng dữ liệu đã có trong Qdrant).")
    parser.add_argument("--no-agent", action="store_true",
                        help="Chỉ đánh giá retrieval, không hỏi agent (bỏ numeric + judge).")
    parser.add_argument("--no-judge", action="store_true",
                        help="Bỏ LLM-as-judge (giữ retrieval + đối chiếu số).")
    parser.add_argument("--limit", type=int, default=None,
                        help="Chỉ chấm N câu đầu mỗi tài liệu (smoke test).")
    parser.add_argument("--out-dir", default="data/eval/results", help="Thư mục xuất báo cáo.")
    parser.add_argument("--version", "-v", default="v0",
                        help="Nhãn phiên bản cho lần cải thiện này (vd v1, v2-rerank). "
                             "Ghi vào report để so sánh các lần chạy.")
    parser.add_argument("--note", default="",
                        help="Ghi chú ngắn về thay đổi của phiên bản này.")
    args = parser.parse_args(argv)

    if args.groundtruth:
        gt_files = args.groundtruth
    else:
        gt_files = sorted(glob.glob("data/eval/qa_ground_truth_*.json"))

    if not gt_files:
        print("Không tìm thấy file ground-truth nào.", file=sys.stderr)
        return 2

    do_ingest = not args.skip_ingest
    do_agent = not args.no_agent
    judge = None
    if do_agent and not args.no_judge:
        judge = build_judge()

    documents = []
    for gt_path in gt_files:
        try:
            doc_report = run_groundtruth_file(
                gt_path=gt_path,
                uploads_dir=args.uploads_dir,
                k=args.k,
                do_ingest=do_ingest,
                do_agent=do_agent,
                judge=judge,
                limit=args.limit,
            )
            documents.append(doc_report)
            print_summary_block(doc_report["company"] or os.path.basename(gt_path),
                                doc_report["summary"])
        except Exception:
            print(f"\n[LỖI] khi xử lý {gt_path}:\n{traceback.format_exc()}", file=sys.stderr)

    if not documents:
        print("Không có tài liệu nào được chấm.", file=sys.stderr)
        return 1

    overall = aggregate(documents)
    print_summary_block("TỔNG THỂ (gộp tất cả tài liệu)", overall)

    from app.config import EMBEDDING_MODEL

    generated_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report = {
        "version": args.version,
        "note": args.note,
        "generated_at": generated_at,
        "llm_model": LLM_MODEL,
        "embedding_model": EMBEDDING_MODEL,
        "k": args.k,
        "overall": overall,
        "documents": documents,
    }

    print(f"\nPhiên bản: {args.version}"
          + (f"  |  ghi chú: {args.note}" if args.note else ""))

    os.makedirs(args.out_dir, exist_ok=True)
    json_path = os.path.join(args.out_dir, f"eval_report_{args.version}_{stamp}.json")
    md_path = os.path.join(args.out_dir, f"eval_report_{args.version}_{stamp}.md")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    write_markdown(report, md_path)

    print(f"\nĐã ghi báo cáo:\n  - {json_path}\n  - {md_path}")

    # TỰ ĐỘNG khi CẢI THIỆN: nếu judge-accuracy của tài liệu nào vượt mức tốt nhất
    # trước đó, tự sinh JSON liệt kê các câu còn sai (GT/answer/retrieval/gold context)
    # vào data/eval/failures/. Bọc try/except để không bao giờ làm hỏng lần eval.
    try:
        from app.eval.dump_failures import dump_if_improved

        outcomes = dump_if_improved(report, results_dir=args.out_dir, report_path=json_path)
        for o in outcomes:
            if o.get("dumped"):
                print(f"  [CẢI THIỆN] {o['company']}: judge {o['judge_accuracy']} > "
                      f"prior {o['prior_best']} -> đã dump {o['num_failures']} câu sai: {o['path']}")
            elif o.get("error"):
                print(f"  [dump lỗi] {o['company']}: {o['error']}")
    except Exception as exc:
        print(f"  [dump_failures bỏ qua do lỗi: {exc}]")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
