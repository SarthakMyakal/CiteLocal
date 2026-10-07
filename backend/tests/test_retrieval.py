"""Baseline, BM25, fusion, reranking and filtering."""
import pytest

from backend.rag.models import get_reranker
from backend.rag.retrieval import Retriever, baseline_retrieve, reciprocal_rank_fusion, tokenize


@pytest.fixture
def retriever(store):
    return Retriever(store, get_reranker())


def test_tokenize():
    assert tokenize("BM25, Dense & RRF!") == ["bm25", "dense", "rrf"]


def test_rrf_rewards_agreement():
    fused = reciprocal_rank_fusion([[1, 2, 3], [2, 3, 1]])
    assert fused[0][0] == 2  # ranked high in both lists
    assert {cid for cid, _ in fused} == {1, 2, 3}


def test_baseline_returns_top_k(store):
    results = baseline_retrieve(store, "how does bread rise", k=3)
    assert len(results) == 3
    assert results[0]["filename"] == "baking.pdf"


def test_bm25_finds_exact_terms(retriever):
    results = retriever.bm25_search("offside rule", k=3)
    top = retriever.store.chunks[results[0][0]]
    assert top["filename"] == "football.pdf" and top["page"] == 2


def test_hybrid_rerank_finds_right_page(retriever):
    results = retriever.retrieve("What converts direct current to alternating current?")
    assert results[0]["filename"] == "solar.pdf" and results[0]["page"] == 2
    scores = [r["rerank_score"] for r in results]
    assert scores == sorted(scores, reverse=True)
    assert len(results) <= 5


def test_filter_by_document(retriever):
    doc_id = next(d["doc_id"] for d in retriever.store.list_documents() if d["filename"] == "football.pdf")
    results = retriever.retrieve("electricity from sunlight", doc_ids=[doc_id])
    assert results and all(r["doc_id"] == doc_id for r in results)


def test_filter_by_collection(retriever):
    results = retriever.retrieve("bread", collection="sport")
    assert results and all(r["collection"] == "sport" for r in results)


def test_bm25_follows_deletes(retriever):
    doc_id = next(d["doc_id"] for d in retriever.store.list_documents() if d["filename"] == "football.pdf")
    retriever.store.delete_document(doc_id)
    assert retriever.bm25_search("offside", k=3) == []
