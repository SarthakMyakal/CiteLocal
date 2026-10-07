"""Hallucination guards: relevance gate, citation validation, groundedness check.

Citations: each retrieved passage is shown to the LLM with a short id ([S1], [S2], ...).
Small models copy short ids far more reliably than long "[filename, p. N]" labels, and
"S" ids cannot be confused with a paper's own reference markers like [12]. After
generation, every id is mapped back to "[filename, p. N]" in code, so the final citation
always points at a passage that was actually retrieved. Unknown ids are dropped.
"""
import re

from langchain_core.messages import HumanMessage, SystemMessage

NOT_FOUND = "Not found in the provided documents."

PASSAGE_ID_RE = re.compile(r"\[\s*S(\d+)\s*\]", re.IGNORECASE)
# Direct citations are accepted too: [attention.pdf, p. 3], [attention.pdf, page 3], [attention.pdf, 3]
FILE_CITATION_RE = re.compile(r"\[\s*([^\[\]]+?\.pdf)\s*,\s*(?:p\.|pp\.|page)?\s*(\d+)\s*\]", re.IGNORECASE)

# Both LLM calls share this system message and then the same context block, so
# Ollama can reuse the already-processed prompt prefix for the second call.
CONTEXT_SYSTEM = (
    "You are a careful assistant for a private document collection. "
    "You only use the context passages the user provides; you never use outside knowledge."
)

ANSWER_RULES = (
    "Answer the question using only the passages above.\n"
    "- Each passage starts with an id such as [S1].\n"
    "- End each sentence with the id of the passage it came from, for example: "
    "The model has 12 layers [S2].\n"
    f'- If the passages do not contain the answer, reply exactly: "{NOT_FOUND}"\n'
    "- Be concise."
)

GROUNDED_TASK = (
    "Question: {question}\n"
    "Proposed answer: {answer}\n\n"
    "Is the proposed answer supported by the passages above? Look for the passage that states it. "
    "Reply YES or NO only."
)


def passes_relevance_gate(chunks: list[dict], threshold: float) -> bool:
    """Abstain early when even the best chunk is a poor match for the question."""
    return bool(chunks) and chunks[0].get("rerank_score", float("-inf")) >= threshold


def format_context(chunks: list[dict]) -> str:
    return "\n\n".join(
        f"[S{i}] ({c['filename']}, page {c['page']})\n{c['text']}" for i, c in enumerate(chunks, start=1)
    )


def _messages(chunks: list[dict], task: str) -> list:
    """System message + context first (shared prefix), task-specific instructions last."""
    return [
        SystemMessage(CONTEXT_SYSTEM),
        HumanMessage(f"Context passages:\n\n{format_context(chunks)}\n\n{task}"),
    ]


def answer_messages(question: str, chunks: list[dict]) -> list:
    return _messages(chunks, f"{ANSWER_RULES}\n\nQuestion: {question}")


def is_not_found(answer: str) -> bool:
    return "not found in the provided documents" in answer.lower()


def validate_citations(answer: str, chunks: list[dict]) -> tuple[str, list[dict], list[dict]]:
    """Turn passage ids into [filename, p. N], keep citations of retrieved pages, drop the rest.

    Returns (cleaned answer, valid citations, invalid citations).
    """
    allowed = {(c["filename"].lower(), c["page"]): c for c in chunks}
    valid, invalid, seen = [], [], set()

    def keep(chunk: dict) -> str:
        key = (chunk["filename"].lower(), chunk["page"])
        if key not in seen:
            seen.add(key)
            valid.append({"filename": chunk["filename"], "page": chunk["page"]})
        return f"[{chunk['filename']}, p. {chunk['page']}]"

    def from_id(match: re.Match) -> str:
        n = int(match.group(1))
        if 1 <= n <= len(chunks):
            return keep(chunks[n - 1])
        invalid.append({"id": f"S{n}"})
        return ""

    def from_file(match: re.Match) -> str:
        filename, page = match.group(1).strip(), int(match.group(2))
        chunk = allowed.get((filename.lower(), page))
        if chunk:
            return keep(chunk)
        invalid.append({"filename": filename, "page": page})
        return ""

    cleaned = FILE_CITATION_RE.sub(from_file, answer)
    cleaned = PASSAGE_ID_RE.sub(from_id, cleaned)
    cleaned = re.sub(r"[ \t]+([.,;:])", r"\1", cleaned).strip()
    return cleaned, valid, invalid


def is_grounded(llm, question: str, answer: str, chunks: list[dict]) -> bool:
    """Second LLM pass: do the retrieved passages actually support the answer?

    Citation labels are removed first so the check judges the claims, not the labels
    (the small model tends to answer NO when a label points at the wrong passage).
    """
    claims = FILE_CITATION_RE.sub("", answer).strip()
    reply = llm.invoke(_messages(chunks, GROUNDED_TASK.format(question=question, answer=claims))).content
    return reply.strip().upper().startswith("YES")
