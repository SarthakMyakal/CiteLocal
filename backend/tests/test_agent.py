"""Agent graph paths, retry loop, tools, and trace."""
from langchain_core.messages import AIMessage

from backend.agent.graph import DocumentAgent
from backend.agent.tools import find_document
from backend.rag.guards import NOT_FOUND
from backend.rag.models import get_reranker
from backend.rag.retrieval import Retriever
from backend.tests.fakes import FakeLLM


def tool_call(name: str, **args) -> AIMessage:
    return AIMessage(content="", tool_calls=[{"name": name, "args": args, "id": "call_1"}])


def make_agent(store, replies, threshold=0.0):
    llm = FakeLLM(replies)
    return DocumentAgent(store, Retriever(store, get_reranker()), llm, threshold=threshold), llm


def steps(result):
    return [s["step"] for s in result["trace"]]


def test_search_path_answers_with_trace(store):
    agent, _ = make_agent(store, [
        tool_call("search_documents", query="inverter"),
        "inverter direct current alternating current",
        "The inverter converts direct current into alternating current [solar.pdf, p. 2].",
        "YES",
    ])
    result = agent.run("What does the inverter do?")
    assert not result["abstained"]
    assert result["citations"] == [{"filename": "solar.pdf", "page": 2}]
    assert steps(result) == ["route", "rewrite", "retrieve", "grade", "generate", "groundedness_check"]


def test_weak_retrieval_retries_twice_then_abstains(store):
    agent, llm = make_agent(store, [tool_call("search_documents", query="x"), "q1", "q2", "q3"], threshold=100)
    result = agent.run("Who painted the Mona Lisa?")
    assert result["abstained"] and result["answer"] == NOT_FOUND
    assert steps(result).count("rewrite") == 3  # first rewrite + 2 retries
    assert steps(result)[-1] == "abstain"
    assert llm.calls == 4  # router + 3 rewrites; no generation call


def test_ungrounded_answer_is_replaced(store):
    agent, _ = make_agent(store, [
        tool_call("search_documents", query="x"), "solar panels", "Solar panels were invented in 1954 [solar.pdf, p. 1].", "NO",
    ], threshold=-100)  # let every chunk through so the groundedness check is what fails
    result = agent.run("When were solar panels invented?")
    assert result["abstained"] and result["reason"] == "not_grounded"


def test_list_documents_tool(store):
    agent, _ = make_agent(store, [tool_call("list_documents")])
    result = agent.run("Which documents do I have?")
    assert "baking.pdf" in result["answer"] and "football.pdf" in result["answer"]
    assert steps(result) == ["route"]


def test_summarise_document_tool(store):
    agent, _ = make_agent(store, [tool_call("summarise_document", filename="baking"), "A guide to sourdough."])
    result = agent.run("Summarise the baking document")
    assert "A guide to sourdough." in result["answer"] and "baking.pdf" in result["answer"]


def test_filters_skip_the_router(store):
    doc_id = find_document(store, "football.pdf")["doc_id"]
    agent, _ = make_agent(store, ["offside", "Attackers cannot wait behind the last defender [football.pdf, p. 2].", "YES"])
    result = agent.run("What is offside?", doc_ids=[doc_id])
    assert result["trace"][0]["why"] == "filter set"
    assert all(c["doc_id"] == doc_id for c in result["chunks"])


def test_find_document_is_forgiving(store):
    assert find_document(store, "BAKING_paper")["filename"] == "baking.pdf"
    assert find_document(store, "the football pdf")["filename"] == "football.pdf"
    assert find_document(store, "tennis") is None
    assert find_document(store, "paper") is None
