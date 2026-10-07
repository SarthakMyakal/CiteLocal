"""Retrieval: the original dense baseline, and hybrid BM25 + dense with reranking.

Pipeline for `Retriever.retrieve`:
    dense top-20  ─┐
                   ├─ reciprocal rank fusion ─> top-20 candidates ─> cross-encoder ─> top-5
    BM25  top-20  ─┘
"""
import re

import numpy as np
from rank_bm25 import BM25Okapi

from backend.config import settings
from backend.rag.store import VectorStore

TOKEN_RE = re.compile(r"\w+")


def tokenize(text: str) -> list[str]:
    return TOKEN_RE.findall(text.lower())


def _result(store: VectorStore, cid: int, **scores) -> dict:
    """Copy of a chunk with the scores that produced it."""
    return {**store.chunks[cid], **scores}


def baseline_retrieve(store: VectorStore, query: str, k: int = settings.baseline_top_k) -> list[dict]:
    """The original main.py behaviour: plain top-k embedding similarity search."""
    return [_result(store, cid, dense_score=s) for cid, s in store.dense_search(query, k)]


def reciprocal_rank_fusion(rankings: list[list[int]], k: int = settings.rrf_k) -> list[tuple[int, float]]:
    """Merge ranked id lists: score(id) = sum over lists of 1 / (k + rank).

    Uses only ranks, so BM25 scores and cosine similarities (different scales)
    never need to be normalised against each other.
    """
    scores: dict[int, float] = {}
    for ranking in rankings:
        for rank, cid in enumerate(ranking, start=1):
            scores[cid] = scores.get(cid, 0.0) + 1.0 / (k + rank)
    return sorted(scores.items(), key=lambda item: item[1], reverse=True)


class Retriever:
    """Hybrid retriever. Keeps an in-memory BM25 index in sync with the vector store."""

    def __init__(self, store: VectorStore, reranker):
        self.store = store
        self.reranker = reranker
        self._bm25 = None
        self._bm25_ids: list[int] = []
        self._bm25_version = -1

    def _ensure_bm25(self) -> None:
        """Rebuild BM25 when documents were added or deleted (cheap at this scale)."""
        if self._bm25_version == self.store.version and self._bm25 is not None:
            return
        self._bm25_ids = list(self.store.chunks.keys())
        corpus = [tokenize(self.store.chunks[cid]["text"]) for cid in self._bm25_ids]
        self._bm25 = BM25Okapi(corpus) if corpus else None
        self._bm25_version = self.store.version

    def bm25_search(self, query: str, k: int, allowed: set[int] | None = None) -> list[tuple[int, float]]:
        self._ensure_bm25()
        if self._bm25 is None:
            return []
        scores = self._bm25.get_scores(tokenize(query))
        results = []
        for i in np.argsort(-scores):
            cid = self._bm25_ids[i]
            if scores[i] <= 0:
                break  # no query term appears in the remaining chunks
            if allowed is None or cid in allowed:
                results.append((cid, float(scores[i])))
                if len(results) == k:
                    break
        return results

    def rerank(self, query: str, candidates: list[dict], top_k: int) -> list[dict]:
        """Score each (query, chunk) pair with the cross-encoder and keep the best top_k."""
        if not candidates:
            return []
        scores = self.reranker.predict([(query, c["text"]) for c in candidates], show_progress_bar=False)
        for c, s in zip(candidates, scores):
            c["rerank_score"] = float(s)
        return sorted(candidates, key=lambda c: c["rerank_score"], reverse=True)[:top_k]

    def retrieve(
        self,
        query: str,
        doc_ids: list[str] | None = None,
        collection: str | None = None,
        top_k: int = settings.final_top_k,
        candidate_k: int = settings.candidate_k,
        rerank_query: str | None = None,
    ) -> list[dict]:
        """Hybrid search + rerank, optionally restricted to some documents or a collection.

        `rerank_query` lets the agent search with a rewritten query but score chunks
        against the user's original question, so scores stay comparable to the threshold.
        """
        allowed = self.store.matching_ids(doc_ids, collection)
        dense = self.store.dense_search(query, candidate_k, allowed)
        sparse = self.bm25_search(query, candidate_k, allowed)
        fused = reciprocal_rank_fusion([[cid for cid, _ in dense], [cid for cid, _ in sparse]])
        candidates = [_result(self.store, cid, rrf_score=score) for cid, score in fused[:candidate_k]]
        return self.rerank(rerank_query or query, candidates, top_k)
