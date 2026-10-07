"""LangGraph agent.

    route ──(list / summarise tool)──────────────────────────────> END
      │
      └─(search)─> rewrite -> retrieve -> grade ──(weak, retries left)──> rewrite
                                            │
                                            ├─(weak, no retries left)──> abstain -> END
                                            └─(relevant)─> generate -> check ─> END (answer or abstain)

Every node appends an entry to `trace`, which is returned with the answer.
"""
import operator
import time
from typing import Annotated, TypedDict

from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import END, StateGraph

from backend.agent.tools import make_tools
from backend.config import settings
from backend.rag import guards

ROUTER_SYSTEM = (
    "You are a document assistant. Call exactly one tool.\n"
    "- search_documents: the default. Use it for every question about what the documents say "
    "(facts, numbers, methods, results), and whenever you are unsure.\n"
    "- list_documents: only if the user asks which documents or files exist.\n"
    "- summarise_document: only if the user asks to summarise one document; pass its exact filename.\n"
    "Available documents: {filenames}"
)

REWRITE_SYSTEM = (
    "Rewrite the user's question as a short search query for a document search engine. "
    "Keep important names, numbers and technical terms. Output only the query."
)

RETRY_SYSTEM = (
    "A search with the query below found nothing relevant. Write a different search query for the same "
    "question: use synonyms or more general terms. Output only the query."
)


class AgentState(TypedDict, total=False):
    question: str
    doc_ids: list[str] | None
    collection: str | None
    query: str
    rewrites: int  # retries used so far
    chunks: list[dict]
    relevant: list[dict]
    answer: str
    citations: list[dict]
    invalid_citations: list[dict]
    abstained: bool
    reason: str
    trace: Annotated[list[dict], operator.add]  # each node appends; LangGraph concatenates


def step(name: str, start: float, **detail) -> dict:
    return {"trace": [{"step": name, "ms": round((time.perf_counter() - start) * 1000), **detail}]}


class DocumentAgent:
    def __init__(self, store, retriever, llm, threshold: float | None = None):
        self.store = store
        self.retriever = retriever
        self.llm = llm
        self.threshold = settings.relevance_threshold if threshold is None else threshold
        self.tools = {t.name: t for t in make_tools(store, retriever, llm)}
        self.graph = self._build()

    # ---------- nodes ----------
    def route(self, state: AgentState) -> dict:
        start = time.perf_counter()
        # When the user has filtered to specific documents, it is always a content search.
        if state.get("doc_ids") or state.get("collection"):
            return {"reason": "search", **step("route", start, tool="search_documents", why="filter set")}
        filenames = ", ".join(d["filename"] for d in self.store.list_documents()) or "none"
        reply = self.llm.bind_tools(list(self.tools.values())).invoke(
            [SystemMessage(ROUTER_SYSTEM.format(filenames=filenames)), HumanMessage(state["question"])]
        )
        call = reply.tool_calls[0] if getattr(reply, "tool_calls", None) else None
        if call and call["name"] in ("list_documents", "summarise_document"):
            output = self.tools[call["name"]].invoke(call.get("args") or {})
            return {
                "answer": output, "abstained": False, "reason": call["name"], "citations": [],
                **step("route", start, tool=call["name"], args=call.get("args") or {}),
            }
        return {"reason": "search", **step("route", start, tool="search_documents")}

    def rewrite(self, state: AgentState) -> dict:
        start = time.perf_counter()
        retry = "query" in state
        prompt = RETRY_SYSTEM if retry else REWRITE_SYSTEM
        text = f"Question: {state['question']}" + (f"\nPrevious query: {state['query']}" if retry else "")
        query = self.llm.invoke([SystemMessage(prompt), HumanMessage(text)]).content.strip().strip('"')
        query = query or state["question"]
        rewrites = state.get("rewrites", 0) + (1 if retry else 0)
        return {"query": query, "rewrites": rewrites, **step("rewrite", start, query=query, retry=retry)}

    def retrieve(self, state: AgentState) -> dict:
        start = time.perf_counter()
        chunks = self.retriever.retrieve(
            state["query"], doc_ids=state.get("doc_ids"), collection=state.get("collection"),
            rerank_query=state["question"],
        )
        hits = [f"{c['filename']} p.{c['page']} ({c['rerank_score']:.2f})" for c in chunks]
        return {"chunks": chunks, **step("retrieve", start, results=hits)}

    def grade(self, state: AgentState) -> dict:
        """Keep chunks whose reranker score clears the calibrated relevance threshold."""
        start = time.perf_counter()
        relevant = [c for c in state["chunks"] if c["rerank_score"] >= self.threshold]
        return {"relevant": relevant, **step("grade", start, relevant=len(relevant), of=len(state["chunks"]))}

    def generate(self, state: AgentState) -> dict:
        start = time.perf_counter()
        raw = self.llm.invoke(guards.answer_messages(state["question"], state["relevant"])).content
        if guards.is_not_found(raw):
            return {"answer": guards.NOT_FOUND, "abstained": True, "reason": "llm_not_found",
                    "citations": [], **step("generate", start, result="model said not found")}
        answer, valid, invalid = guards.validate_citations(raw, state["relevant"])
        return {"answer": answer, "citations": valid, "invalid_citations": invalid,
                **step("generate", start, citations=len(valid), invalid_citations=len(invalid))}

    def check(self, state: AgentState) -> dict:
        start = time.perf_counter()
        if state.get("abstained"):
            return step("groundedness_check", start, skipped=True)
        if guards.is_grounded(self.llm, state["question"], state["answer"], state["relevant"]):
            return {"abstained": False, "reason": "answered", **step("groundedness_check", start, grounded=True)}
        return {"answer": guards.NOT_FOUND, "abstained": True, "reason": "not_grounded", "citations": [],
                **step("groundedness_check", start, grounded=False)}

    def abstain(self, state: AgentState) -> dict:
        return {"answer": guards.NOT_FOUND, "abstained": True, "reason": "relevance_gate", "citations": [],
                **step("abstain", time.perf_counter(), why="no chunk above relevance threshold")}

    # ---------- edges ----------
    def after_route(self, state: AgentState) -> str:
        return "rewrite" if state["reason"] == "search" else END

    def after_grade(self, state: AgentState) -> str:
        if state["relevant"]:
            return "generate"
        if state.get("rewrites", 0) < settings.max_rewrites:
            return "rewrite"
        return "abstain"

    def _build(self):
        g = StateGraph(AgentState)
        for name in ("route", "rewrite", "retrieve", "grade", "generate", "check", "abstain"):
            g.add_node(name, getattr(self, name))
        g.set_entry_point("route")
        g.add_conditional_edges("route", self.after_route, ["rewrite", END])
        g.add_edge("rewrite", "retrieve")
        g.add_edge("retrieve", "grade")
        g.add_conditional_edges("grade", self.after_grade, ["generate", "rewrite", "abstain"])
        g.add_edge("generate", "check")
        g.add_edge("check", END)
        g.add_edge("abstain", END)
        return g.compile()

    def run(self, question: str, doc_ids: list[str] | None = None, collection: str | None = None) -> dict:
        state = self.graph.invoke({"question": question, "doc_ids": doc_ids, "collection": collection, "trace": []})
        return {
            "answer": state["answer"],
            "abstained": state.get("abstained", False),
            "reason": state.get("reason", ""),
            "citations": state.get("citations", []),
            "invalid_citations": state.get("invalid_citations", []),
            "chunks": state.get("relevant") or state.get("chunks", []),
            "trace": state["trace"],
        }
