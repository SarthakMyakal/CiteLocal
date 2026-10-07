"""Pick the relevance-gate threshold from the reranker scores saved by `scripts.eval retrieval`.

For every candidate threshold t, questions whose top reranker score is >= t pass the gate.
We choose the t with the best balanced accuracy:
    (share of answerable questions that pass + share of unanswerable questions blocked) / 2
Ties go to the lower threshold (prefer answering over abstaining). Every value between
the best score and the next lower observed score gives the same accuracy, so we return
the midpoint of that gap rather than sitting exactly on one question's score. The
groundedness check is the second line of defence for anything the gate lets through.

Usage: python -m scripts.calibrate
"""
import json

from backend.config import settings

EVAL_DIR = settings.storage_dir / "eval"


def calibrate(answerable: list[float], unanswerable: list[float]) -> dict:
    candidates = sorted(set(answerable + unanswerable))
    best = None
    for i, t in enumerate(candidates):
        kept = sum(s >= t for s in answerable) / len(answerable)
        blocked = sum(s < t for s in unanswerable) / len(unanswerable)
        score = (kept + blocked) / 2
        if best is None or score > best["balanced_accuracy"]:
            midpoint = (candidates[i - 1] + t) / 2 if i > 0 else t
            best = {"threshold": round(midpoint, 2), "answerable_kept": kept, "unanswerable_blocked": blocked,
                    "balanced_accuracy": score}
    return best


def main() -> None:
    scores = json.loads((EVAL_DIR / "retrieval.json").read_text(encoding="utf-8"))["rerank_scores"]
    result = calibrate(scores["answerable"], scores["unanswerable"])
    (EVAL_DIR / "threshold.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
    print("Set relevance_threshold in backend/config.py (or PDA_RELEVANCE_THRESHOLD) to this value.")


if __name__ == "__main__":
    main()
