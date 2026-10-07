"""End-to-end evaluation of the hallucination guards with the real local LLM.

Runs every question (answerable and unanswerable) through the guarded pipeline on
the full corpus and records whether it answered or abstained, which guard fired,
whether the answer cited the expected page, and latency.

Usage:
    python -m scripts.eval_guards            # single-pass pipeline -> storage/eval/guards.json
    python -m scripts.eval_guards --agent    # LangGraph agent      -> storage/eval/agent.json
"""
import argparse
import json
import statistics
import time
from collections import Counter

from backend.agent.graph import DocumentAgent
from backend.config import settings
from backend.rag.models import get_llm, get_reranker
from backend.rag.pipeline import answer_question
from backend.rag.retrieval import Retriever
from scripts.eval import EVAL_DIR, build_store, load_questions


def summarise(rows: list[dict]) -> dict:
    answerable = [r for r in rows if r["answerable"]]
    unanswerable = [r for r in rows if not r["answerable"]]
    answered = [r for r in answerable if not r["abstained"]]
    return {
        "threshold": settings.relevance_threshold,
        "llm_model": settings.llm_model,
        "n_answerable": len(answerable),
        "n_unanswerable": len(unanswerable),
        "abstention_rate_unanswerable": sum(r["abstained"] for r in unanswerable) / len(unanswerable),
        "false_abstention_rate_answerable": sum(r["abstained"] for r in answerable) / len(answerable),
        "answered_with_valid_citation": sum(bool(r["citations"]) for r in answered) / max(len(answered), 1),
        "answered_citing_expected_page": sum(r["cited_expected"] for r in answered) / max(len(answered), 1),
        "invalid_citations_dropped": sum(r["invalid_citations"] for r in rows),
        "abstain_reasons": dict(Counter(r["reason"] for r in rows if r["abstained"])),
        "latency_s_mean": statistics.mean(r["latency_s"] for r in rows),
        "latency_s_median": statistics.median(r["latency_s"] for r in rows),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--agent", action="store_true", help="evaluate the LangGraph agent instead")
    args = parser.parse_args()

    answerable, unanswerable = load_questions()
    store = build_store("full", sorted(settings.data_dir.glob("*.pdf")))
    retriever = Retriever(store, get_reranker())
    llm = get_llm()
    agent = DocumentAgent(store, retriever, llm) if args.agent else None

    rows = []
    for i, q in enumerate(answerable + unanswerable, start=1):
        start = time.perf_counter()
        out = agent.run(q["question"]) if agent else answer_question(retriever, llm, q["question"])
        latency = time.perf_counter() - start
        cited_expected = q["answerable"] and any(
            c["filename"] == q["doc"] and c["page"] in q["pages"] for c in out["citations"]
        )
        rows.append({
            "id": q["id"], "answerable": q["answerable"], "abstained": out["abstained"], "reason": out["reason"],
            "citations": out["citations"], "invalid_citations": len(out["invalid_citations"]),
            "cited_expected": cited_expected, "latency_s": latency, "answer": out["answer"],
        })
        print(f"[{i}/{len(answerable) + len(unanswerable)}] {q['id']}: {out['reason']} ({latency:.1f}s)", flush=True)

    summary = summarise(rows)
    name = "agent.json" if args.agent else "guards.json"
    (EVAL_DIR / name).write_text(json.dumps({"summary": summary, "rows": rows}, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
