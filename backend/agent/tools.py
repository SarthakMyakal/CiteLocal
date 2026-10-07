"""The three tools the agent can call.

Built inside a factory so each tool closes over the store, retriever and LLM
(LangChain tools take only JSON-serialisable arguments from the model).
"""
import re

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.tools import tool

SUMMARY_CHAR_LIMIT = 6000  # keep the prompt small enough for a 3B model on CPU
FILLER_WORDS = {"the", "paper", "document", "doc", "pdf", "file", "a", "of"}


def _words(text: str) -> set[str]:
    return set(re.findall(r"[a-z0-9]+", text.lower())) - FILLER_WORDS


def find_document(store, name: str) -> dict | None:
    """Match a document by id or filename, tolerating case, '.pdf', '_' and words like 'paper'.

    "LoRA_paper", "lora", "the BERT pdf" all find the right file.
    """
    wanted = _words(name.removesuffix(".pdf"))
    if not wanted:
        return None
    docs = store.list_documents()
    for d in docs:  # exact id or filename first
        if name.strip().lower() in (d["doc_id"], d["filename"].lower()):
            return d
    for d in docs:  # every meaningful word of the request appears in the filename
        if wanted <= _words(d["filename"].removesuffix(".pdf")):
            return d
    return None


def make_tools(store, retriever, llm):
    @tool
    def search_documents(query: str) -> str:
        """Default tool. Search the documents for any question about facts, numbers, methods or other content."""
        chunks = retriever.retrieve(query)
        return "\n\n".join(f"[{c['filename']}, p. {c['page']}] {c['text']}" for c in chunks)

    @tool
    def list_documents() -> str:
        """Use ONLY when the user asks which documents or files are available. Never for content questions."""
        docs = store.list_documents()
        if not docs:
            return "No documents have been uploaded yet."
        return "\n".join(f"- {d['filename']} ({d['num_pages']} pages, collection: {d['collection']})" for d in docs)

    @tool
    def summarise_document(filename: str) -> str:
        """Use ONLY when the user asks to summarise or give an overview of one named document."""
        doc = find_document(store, filename)
        if doc is None:
            return f"No document named '{filename}' was found."
        chunks = sorted((c for c in store.chunks.values() if c["doc_id"] == doc["doc_id"]), key=lambda c: c["chunk_id"])
        text, pages = "", set()
        for c in chunks:  # the start of a document (abstract, intro) is the most informative part
            if len(text) + len(c["text"]) > SUMMARY_CHAR_LIMIT:
                break
            text += c["text"] + "\n"
            pages.add(c["page"])
        summary = llm.invoke([
            SystemMessage("Summarise the document excerpt in 4-6 sentences. Use only the excerpt."),
            HumanMessage(text),
        ]).content
        page_range = f"pp. {min(pages)}-{max(pages)}" if pages else "no text"
        return f"{summary}\n\n(Summary of {doc['filename']}, based on {page_range}.)"

    return [search_documents, list_documents, summarise_document]
