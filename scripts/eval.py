"""Retrieval evaluation: baseline vs hybrid + rerank, on a 3-PDF corpus and the full corpus.

A question is a hit when a returned chunk comes from the expected file AND one of
its expected pages. MRR uses the rank of the first such chunk (0 if none).

Usage:
    python -m scripts.eval retrieval    # -> storage/eval/retrieval.json
    python -m scripts.eval report       # rewrite docs/EVAL.md from the saved JSON files
"""
import argparse
import json
import statistics
import time
from pathlib import Path

from backend.config import ROOT_DIR, settings
from backend.rag.models import get_embedder, get_reranker
from backend.rag.retrieval import Retriever, baseline_retrieve, reciprocal_rank_fusion
from backend.rag.store import VectorStore

QUESTIONS_FILE = ROOT_DIR / "eval" / "questions.json"
EVAL_DIR = settings.storage_dir / "eval"
SMALL_CORPUS = ["Attention Is All You Need.pdf", "BERT.pdf", "rag.pdf"]


def load_questions() -> tuple[list[dict], list[dict]]:
    """Return (answerable, unanswerable) questions."""
    questions = json.loads(QUESTIONS_FILE.read_text(encoding="utf-8"))["questions"]
    return [q for q in questions if q["answerable"]], [q for q in questions if not q["answerable"]]


def build_store(name: str, files: list[Path]) -> VectorStore:
    """Build (or reuse) a separate index for an evaluation corpus, so the app's index is untouched."""
    store = VectorStore(EVAL_DIR / name, get_embedder())
    for f in files:
        store.add_document(f)
    return store


def is_hit(chunk: dict, q: dict) -> bool:
    return chunk["filename"] == q["doc"] and chunk["page"] in q["pages"]


def first_hit_rank(results: list[dict], q: dict) -> int:
    return next((rank for rank, r in enumerate(results, start=1) if is_hit(r, q)), 0)


def evaluate(method, questions: list[dict]) -> dict:
    """Run one retrieval method over the questions; compute hit rate, MRR and latency."""
    method(questions[0]["question"])  # warm-up so one-off model loading is not timed
    ranks, doc_hits, latencies = [], [], []
    for q in questions:
        start = time.perf_counter()
        results = method(q["question"])
        latencies.append((time.perf_counter() - start) * 1000)
        ranks.append(first_hit_rank(results, q))
        doc_hits.append(any(r["filename"] == q["doc"] for r in results))
    latencies.sort()
    return {
        "hit": sum(r > 0 for r in ranks) / len(ranks),
        "doc_hit": sum(doc_hits) / len(doc_hits),
        "mrr": sum(1 / r for r in ranks if r) / len(ranks),
        "latency_ms_mean": statistics.mean(latencies),
        "latency_ms_p95": latencies[round(0.95 * (len(latencies) - 1))],
        "n": len(questions),
    }


def hybrid_no_rerank(retriever: Retriever, query: str, k: int = 5) -> list[dict]:
    """Ablation: the fused BM25 + dense list without the cross-encoder."""
    dense = retriever.store.dense_search(query, settings.candidate_k)
    sparse = retriever.bm25_search(query, settings.candidate_k)
    fused = reciprocal_rank_fusion([[c for c, _ in dense], [c for c, _ in sparse]])
    return [retriever.store.chunks[cid] for cid, _ in fused[:k]]


def run_corpus(name: str, files: list[Path], questions: list[dict]) -> dict:
    store = build_store(name, files)
    retriever = Retriever(store, get_reranker())
    methods = {
        "Baseline: dense top-3 (original main.py)": lambda q: baseline_retrieve(store, q, k=3),
        "Dense top-5": lambda q: baseline_retrieve(store, q, k=5),
        "Hybrid BM25 + dense (RRF) top-5, no rerank": lambda q: hybrid_no_rerank(retriever, q),
        "Hybrid + cross-encoder rerank top-5": lambda q: retriever.retrieve(q),
    }
    print(f"\n== {name}: {len(files)} PDFs, {store.index.ntotal} chunks, {len(questions)} questions ==")
    results = {"docs": len(files), "chunks": store.index.ntotal, "methods": {}}
    for label, method in methods.items():
        m = evaluate(method, questions)
        results["methods"][label] = m
        print(f"{label:45s} hit={m['hit']:.3f} doc_hit={m['doc_hit']:.3f} mrr={m['mrr']:.3f} "
              f"latency={m['latency_ms_mean']:.0f}ms (p95 {m['latency_ms_p95']:.0f}ms)")
    return results


def top_rerank_scores(questions: list[dict]) -> list[float]:
    """Best reranker score per question on the full corpus (used to calibrate the relevance gate)."""
    store = build_store("full", sorted(settings.data_dir.glob("*.pdf")))
    retriever = Retriever(store, get_reranker())
    return [retriever.retrieve(q["question"])[0]["rerank_score"] for q in questions]


def run_retrieval() -> None:
    answerable, unanswerable = load_questions()
    pdfs = sorted(settings.data_dir.glob("*.pdf"))
    small = [p for p in pdfs if p.name in SMALL_CORPUS]
    results = {
        "small": run_corpus("small", small, [q for q in answerable if q["doc"] in SMALL_CORPUS]),
        "full": run_corpus("full", pdfs, answerable),
        "rerank_scores": {
            "answerable": top_rerank_scores(answerable),
            "unanswerable": top_rerank_scores(unanswerable),
        },
    }
    EVAL_DIR.mkdir(parents=True, exist_ok=True)
    (EVAL_DIR / "retrieval.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"\nSaved {EVAL_DIR / 'retrieval.json'}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=["retrieval", "report"])
    args = parser.parse_args()
    if args.command == "retrieval":
        run_retrieval()
    else:
        from scripts.eval_report import write_report

        write_report()


if __name__ == "__main__":
    main()
