"""Liệt kê CÁC CÂU CÒN SAI của một lần eval ra JSON để mổ lỗi.

Với mỗi câu bị LLM-judge chấm SAI (judge.correct == False) trong một eval report,
ghi lại: câu hỏi, ground_truth, câu trả lời của hệ thống, **retrieval context**
(chạy lại đúng pipeline `retrieve()` để lấy text 5 chunk agent thấy) và **gold
context** (mọi chunk trên `expected_pages`, lấy từ Qdrant). Xuất JSON vào
`data/eval/failures/`.

TỰ ĐỘNG khi CẢI THIỆN: `dump_if_improved()` so judge-accuracy của lần này với
mức TỐT NHẤT trước đó (cùng `source_document`); chỉ dump khi tăng (hoặc lần đầu).
`run_eval.py` gọi hàm này ở cuối mỗi lần chạy nên báo cáo lỗi tự sinh khi điểm lên.

Chạy tay:
    python -m app.eval.dump_failures                     # report mới nhất, chỉ khi cải thiện
    python -m app.eval.dump_failures --report <path>     # chỉ định report
    python -m app.eval.dump_failures --all               # dump bất kể có cải thiện hay không
"""

import argparse
import glob
import json
import os
from datetime import datetime

DEFAULT_RESULTS_DIR = "data/eval/results"
DEFAULT_OUT_DIR = "data/eval/failures"


# --------------------------------------------------------------------------- #
# Phát hiện "cải thiện": so judge accuracy với mức tốt nhất TRƯỚC ĐÓ
# --------------------------------------------------------------------------- #

def _judge_acc(doc: dict):
    s = (doc.get("summary") or {}).get("judge") or {}
    return s.get("accuracy")


def best_prior_judge(source_document: str, before_ts: str, results_dir: str,
                     exclude_path: str) -> float | None:
    """Judge-accuracy TỐT NHẤT của các report TRƯỚC (cùng source_document, generated_at nhỏ hơn).

    Trả None nếu chưa có report nào trước đó (=> lần đầu, coi như cải thiện).
    """
    best = None
    for path in glob.glob(os.path.join(results_dir, "eval_report_*.json")):
        if os.path.abspath(path) == os.path.abspath(exclude_path):
            continue
        try:
            with open(path, encoding="utf-8") as f:
                rep = json.load(f)
        except Exception:
            continue
        if rep.get("generated_at", "") >= before_ts:
            continue  # chỉ tính các lần TRƯỚC lần hiện tại
        for doc in rep.get("documents", []):
            if doc.get("source_document") != source_document:
                continue
            acc = _judge_acc(doc)
            if acc is not None and (best is None or acc > best):
                best = acc
    return best


# --------------------------------------------------------------------------- #
# Lấy retrieval context + gold context cho một câu
# --------------------------------------------------------------------------- #

def _gold_context(client, collection, document_id, pages):
    if not pages:
        return []
    from app.rag.vector_store import build_document_filter
    pages = set(pages)
    out, offset = [], None
    while True:
        batch, offset = client.scroll(
            collection_name=collection,
            scroll_filter=build_document_filter(document_id),
            limit=500, offset=offset, with_payload=True, with_vectors=False,
        )
        for p in batch:
            md = p.payload.get("metadata", {})
            if md.get("page") in pages:
                out.append({
                    "page": md.get("page"),
                    "content_type": md.get("content_type"),
                    "source": md.get("source"),
                    "text": p.payload.get("page_content", ""),
                })
        if offset is None:
            break
    out.sort(key=lambda x: (x["page"] if x["page"] is not None else 1e9,
                            x["content_type"] or ""))
    return out


def _retrieval_context(question, document_id, k):
    from app.rag.retrieval import retrieve
    try:
        docs = retrieve(question, document_id, k)
    except Exception as exc:
        return [{"error": f"retrieve_failed: {exc}"}]
    return [{
        "rank": i + 1,
        "page": d.metadata.get("page"),
        "content_type": d.metadata.get("content_type"),
        "source": d.metadata.get("source"),
        "text": d.page_content,
    } for i, d in enumerate(docs)]


def _is_failure(row: dict) -> bool:
    j = row.get("judge") or {}
    return bool(j.get("applicable")) and j.get("correct") is False


# --------------------------------------------------------------------------- #
# Dump cho MỘT tài liệu trong report
# --------------------------------------------------------------------------- #

def _safe(s: str) -> str:
    return "".join(c if c.isalnum() else "_" for c in (s or "")).strip("_")[:40]


def dump_document_failures(report: dict, doc: dict, out_dir: str, k: int) -> dict:
    from app.rag.vector_store import get_client
    from app.config import QDRANT_COLLECTION

    document_id = doc["document_id"]
    client = get_client()

    failures = []
    for row in doc.get("results", []):
        if not _is_failure(row):
            continue
        expected_pages = row.get("expected_pages") or []
        failures.append({
            "id": row.get("id"),
            "category": row.get("category"),
            "content_type": row.get("content_type"),
            "question": row.get("question"),
            "ground_truth": row.get("ground_truth"),
            "answer": row.get("answer"),
            "agent_error": row.get("agent_error"),
            "expected_pages": expected_pages,
            "numeric": row.get("numeric"),
            "judge_reason": (row.get("judge") or {}).get("reason", ""),
            "retrieval_context": _retrieval_context(row.get("question", ""), document_id, k),
            "gold_context": _gold_context(client, QDRANT_COLLECTION, document_id, expected_pages),
        })

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    version = report.get("version", "v0")
    company = _safe(doc.get("company") or doc.get("source_document") or "doc")
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"failures_{version}_{company}_{stamp}.json")

    payload = {
        "version": version,
        "generated_at": report.get("generated_at"),
        "dumped_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "company": doc.get("company"),
        "source_document": doc.get("source_document"),
        "document_id": document_id,
        "k": k,
        "judge_accuracy": _judge_acc(doc),
        "num_failures": len(failures),
        "failure_ids": [f["id"] for f in failures],
        "failures": failures,
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    return {"path": path, "num_failures": len(failures)}


# --------------------------------------------------------------------------- #
# Cấp cao: dump khi cải thiện (dùng bởi run_eval + CLI)
# --------------------------------------------------------------------------- #

def dump_if_improved(report: dict, results_dir: str = DEFAULT_RESULTS_DIR,
                     out_dir: str = DEFAULT_OUT_DIR, report_path: str = "",
                     force: bool = False) -> list[dict]:
    """Với mỗi tài liệu trong report: nếu judge-accuracy > mức tốt nhất trước đó
    (hoặc force / lần đầu) -> dump JSON các câu còn sai. Trả danh sách kết quả.

    An toàn: mọi lỗi được nuốt và báo trong kết quả, không làm hỏng caller.
    """
    k = report.get("k", 5)
    before_ts = report.get("generated_at", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    outcomes = []
    for doc in report.get("documents", []):
        cur = _judge_acc(doc)
        baseline = best_prior_judge(doc.get("source_document", ""), before_ts,
                                    results_dir, report_path)
        improved = force or baseline is None or (cur is not None and cur > baseline)
        entry = {"company": doc.get("company"), "source_document": doc.get("source_document"),
                 "judge_accuracy": cur, "prior_best": baseline, "improved": improved,
                 "dumped": False}
        if improved:
            try:
                res = dump_document_failures(report, doc, out_dir, k)
                entry.update({"dumped": True, **res})
            except Exception as exc:
                entry["error"] = f"dump_failed: {exc}"
        outcomes.append(entry)
    return outcomes


def _load_latest_report(results_dir: str) -> tuple[dict, str]:
    paths = sorted(glob.glob(os.path.join(results_dir, "eval_report_*.json")),
                   key=os.path.getmtime)
    if not paths:
        raise SystemExit(f"Không tìm thấy report nào trong {results_dir}.")
    with open(paths[-1], encoding="utf-8") as f:
        return json.load(f), paths[-1]


def main(argv=None):
    ap = argparse.ArgumentParser(description="Dump câu còn sai của một eval report ra JSON.")
    ap.add_argument("--report", default=None, help="Đường dẫn eval_report_*.json (mặc định: mới nhất).")
    ap.add_argument("--results-dir", default=DEFAULT_RESULTS_DIR)
    ap.add_argument("--out-dir", default=DEFAULT_OUT_DIR)
    ap.add_argument("--all", action="store_true",
                    help="Dump bất kể có cải thiện hay không (bỏ so sánh baseline).")
    args = ap.parse_args(argv)

    if args.report:
        with open(args.report, encoding="utf-8") as f:
            report = json.load(f)
        report_path = args.report
    else:
        report, report_path = _load_latest_report(args.results_dir)

    outcomes = dump_if_improved(report, args.results_dir, args.out_dir,
                                report_path=report_path, force=args.all)
    for o in outcomes:
        if o.get("dumped"):
            print(f"[DUMP] {o['company']}: {o['num_failures']} câu sai "
                  f"(judge {o['judge_accuracy']} > prior {o['prior_best']}) -> {o['path']}")
        elif o.get("error"):
            print(f"[LỖI] {o['company']}: {o['error']}")
        else:
            print(f"[BỎ QUA] {o['company']}: judge {o['judge_accuracy']} "
                  f"KHÔNG vượt prior {o['prior_best']} (dùng --all để ép dump).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
