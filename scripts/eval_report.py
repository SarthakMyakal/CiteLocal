"""Render the saved evaluation JSON files as Markdown tables (pasted into docs/EVAL.md).

Usage: python -m scripts.eval report
"""
import json

from scripts.eval import EVAL_DIR


def retrieval_table(corpus: dict) -> str:
    lines = [
        "| Method | Hit@k (page) | Doc hit | MRR | Mean latency | p95 latency |",
        "|---|---|---|---|---|---|",
    ]
    for label, m in corpus["methods"].items():
        lines.append(f"| {label} | {m['hit']:.3f} | {m['doc_hit']:.3f} | {m['mrr']:.3f} | "
                     f"{m['latency_ms_mean']:.0f} ms | {m['latency_ms_p95']:.0f} ms |")
    return "\n".join(lines)


def guards_table(summary: dict) -> str:
    rows = [
        ("Abstention rate on unanswerable questions", f"{summary['abstention_rate_unanswerable']:.1%}"),
        ("False abstention rate on answerable questions", f"{summary['false_abstention_rate_answerable']:.1%}"),
        ("Answered questions with at least one valid citation", f"{summary['answered_with_valid_citation']:.1%}"),
        ("Answered questions citing the expected page", f"{summary['answered_citing_expected_page']:.1%}"),
        ("Invalid citations dropped (total)", str(summary["invalid_citations_dropped"])),
        ("Abstain reasons", ", ".join(f"{k}: {v}" for k, v in summary["abstain_reasons"].items()) or "none"),
        ("Latency per question (mean / median)",
         f"{summary['latency_s_mean']:.1f} s / {summary['latency_s_median']:.1f} s"),
    ]
    return "\n".join(["| Metric | Value |", "|---|---|"] + [f"| {k} | {v} |" for k, v in rows])


def write_report() -> None:
    retrieval = json.loads((EVAL_DIR / "retrieval.json").read_text(encoding="utf-8"))
    parts = []
    for key in ("small", "full"):
        c = retrieval[key]
        n = next(iter(c["methods"].values()))["n"]
        parts.append(f"### {key.title()} corpus ({c['docs']} PDFs, {c['chunks']} chunks, {n} questions)\n\n"
                     + retrieval_table(c))
    for name, title in (("guards.json", "Guarded single-pass pipeline"), ("agent.json", "LangGraph agent")):
        path = EVAL_DIR / name
        if path.exists():
            summary = json.loads(path.read_text(encoding="utf-8"))["summary"]
            parts.append(f"### {title} (threshold {summary['threshold']:.2f}, {summary['llm_model']}, "
                         f"{summary['n_answerable']} answerable + {summary['n_unanswerable']} unanswerable)\n\n"
                         + guards_table(summary))
    print("\n\n".join(parts))


if __name__ == "__main__":
    write_report()
