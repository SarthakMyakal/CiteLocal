"""Every API endpoint, using a temporary index/database and a fake LLM."""
import pytest
from fastapi.testclient import TestClient
from langchain_core.messages import AIMessage

from backend.agent.graph import DocumentAgent
from backend.app.main import Services, create_app
from backend.db.database import make_session_factory
from backend.rag.models import get_embedder, get_reranker
from backend.rag.retrieval import Retriever
from backend.rag.store import VectorStore
from backend.tests.fakes import FakeLLM


@pytest.fixture
def llm():
    return FakeLLM([])


@pytest.fixture
def client(tmp_path, llm):
    store = VectorStore(tmp_path / "index", get_embedder())
    agent = DocumentAgent(store, Retriever(store, get_reranker()), llm, threshold=0.0)
    services = Services(store, agent, make_session_factory(f"sqlite:///{(tmp_path / 'test.db').as_posix()}"),
                        tmp_path / "uploads")
    with TestClient(create_app(services)) as c:
        yield c


def upload(client, path, collection="default"):
    with open(path, "rb") as f:
        return client.post("/documents", files={"file": (path.name, f, "application/pdf")},
                           data={"collection": collection})


def test_health(client):
    body = client.get("/health").json()
    assert body["status"] == "ok" and body["documents"] == 0


def test_upload_list_and_duplicate(client, pdf_dir):
    r = upload(client, pdf_dir / "solar.pdf", "energy")
    assert r.status_code == 201 and r.json()["duplicate"] is False
    assert r.json()["document"]["num_pages"] == 2
    again = upload(client, pdf_dir / "solar.pdf")
    assert again.json()["duplicate"] is True
    docs = client.get("/documents").json()
    assert [d["filename"] for d in docs] == ["solar.pdf"] and docs[0]["collection"] == "energy"


def test_upload_rejects_non_pdf(client, tmp_path):
    bad = tmp_path / "notes.txt"
    bad.write_text("hello")
    assert upload(client, bad).status_code == 400


def test_delete_document(client, pdf_dir):
    doc_id = upload(client, pdf_dir / "baking.pdf").json()["document"]["id"]
    assert client.delete(f"/documents/{doc_id}").status_code == 204
    assert client.get("/documents").json() == []
    assert client.get("/health").json()["chunks"] == 0
    assert client.delete(f"/documents/{doc_id}").status_code == 404


def test_chat_creates_session_and_logs(client, pdf_dir, llm):
    upload(client, pdf_dir / "solar.pdf")
    llm.replies = [
        AIMessage(content="", tool_calls=[{"name": "search_documents", "args": {"query": "inverter"}, "id": "1"}]),
        "inverter alternating current",
        "It converts direct current to alternating current [solar.pdf, p. 2].",
        "YES",
    ]
    r = client.post("/chat", json={"question": "What does the inverter do?"})
    assert r.status_code == 200
    body = r.json()
    assert body["citations"] == [{"filename": "solar.pdf", "page": 2}]
    assert body["sources"] and body["trace"][0]["step"] == "route"

    sessions = client.get("/sessions").json()
    assert sessions[0]["id"] == body["session_id"]
    messages = client.get(f"/sessions/{body['session_id']}/messages").json()
    assert [m["role"] for m in messages] == ["user", "assistant"]
    assert messages[1]["citations"][0]["page"] == 2

    with client.app.state.services.session_factory() as db:
        from backend.db.models import QueryLog

        log = db.query(QueryLog).one()
        assert log.abstained is False and log.chunk_ids and log.latency_ms > 0


def test_chat_continues_existing_session(client, pdf_dir, llm):
    upload(client, pdf_dir / "football.pdf")
    first = client.post("/chat", json={"question": "Which documents are there?"}).json()
    second = client.post("/chat", json={"question": "Anything else?", "session_id": first["session_id"]}).json()
    assert second["session_id"] == first["session_id"]
    assert len(client.get(f"/sessions/{first['session_id']}/messages").json()) == 4


def test_chat_unknown_session_and_validation(client):
    assert client.post("/chat", json={"question": "hi", "session_id": 999}).status_code == 404
    assert client.post("/chat", json={"question": ""}).status_code == 422
    assert client.get("/sessions/999/messages").status_code == 404
