"""Single-pass guarded RAG: retrieve -> relevance gate -> generate -> citations -> groundedness.

The LangGraph agent (backend/agent) adds query rewriting and retries on top of
these same steps; this module is the simple version used by the evaluation.
"""
import time

from backend.config import settings
from backend.rag import guards


def answer_question(retriever, llm, question: str, doc_ids=None, collection=None,
                    threshold: float | None = None) -> dict:
    threshold = settings.relevance_threshold if threshold is None else threshold
    timings = {}

    start = time.perf_counter()
    chunks = retriever.retrieve(question, doc_ids=doc_ids, collection=collection)
    timings["retrieve_ms"] = (time.perf_counter() - start) * 1000

    def result(answer, abstained, reason, citations=(), invalid=()):
        return {
            "answer": answer,
            "abstained": abstained,
            "reason": reason,
            "citations": list(citations),
            "invalid_citations": list(invalid),
            "chunks": chunks,
            "timings": timings,
        }

    # 1. Relevance gate: skip the LLM entirely if nothing relevant was found.
    if not guards.passes_relevance_gate(chunks, threshold):
        return result(guards.NOT_FOUND, True, "relevance_gate")

    # 2. Generate an answer with citations.
    start = time.perf_counter()
    raw = llm.invoke(guards.answer_messages(question, chunks)).content
    timings["generate_ms"] = (time.perf_counter() - start) * 1000
    if guards.is_not_found(raw):
        return result(guards.NOT_FOUND, True, "llm_not_found")

    # 3. Validate citations against what was actually retrieved.
    answer, valid, invalid = guards.validate_citations(raw, chunks)

    # 4. Groundedness check.
    start = time.perf_counter()
    grounded = guards.is_grounded(llm, question, answer, chunks)
    timings["groundedness_ms"] = (time.perf_counter() - start) * 1000
    if not grounded:
        return result(guards.NOT_FOUND, True, "not_grounded", invalid=invalid)

    return result(answer, False, "answered", valid, invalid)
