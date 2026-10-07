"""Vector store: a FAISS index plus the chunk text and metadata it points to.

Each chunk gets an integer id. FAISS holds id -> vector; `self.chunks` holds
id -> {text, doc_id, filename, page, collection}. `IndexIDMap2` lets us add
and remove vectors by id, so new documents are added incrementally and a
deleted document's chunks are removed without rebuilding the index.
"""
import json
import threading
from pathlib import Path

import faiss
import numpy as np

from backend.rag.ingest import chunk_pages, file_sha256, read_pages


class VectorStore:
    def __init__(self, index_dir: Path, embedder):
        self.index_dir = Path(index_dir)
        self.embedder = embedder
        self.lock = threading.Lock()
        self.version = 0  # bumped on every change so the BM25 index knows to rebuild
        self.index = faiss.IndexIDMap2(faiss.IndexFlatIP(embedder.dim))
        self.chunks: dict[int, dict] = {}
        self.docs: dict[str, dict] = {}
        self.next_id = 0
        self._load()

    # ---------- persistence ----------
    def _load(self) -> None:
        index_file = self.index_dir / "index.faiss"
        meta_file = self.index_dir / "meta.json"
        if not (index_file.exists() and meta_file.exists()):
            return
        self.index = faiss.read_index(str(index_file))
        meta = json.loads(meta_file.read_text(encoding="utf-8"))
        self.chunks = {int(k): v for k, v in meta["chunks"].items()}
        self.docs = meta["docs"]
        self.next_id = meta["next_id"]

    def save(self) -> None:
        self.index_dir.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.index, str(self.index_dir / "index.faiss"))
        meta = {"chunks": self.chunks, "docs": self.docs, "next_id": self.next_id}
        (self.index_dir / "meta.json").write_text(json.dumps(meta), encoding="utf-8")

    # ---------- documents ----------
    def add_document(self, path: Path, collection: str = "default", filename: str | None = None) -> tuple[dict, bool]:
        """Ingest one PDF. Returns (document info, True if newly added / False if duplicate)."""
        path = Path(path)
        sha = file_sha256(path)
        doc_id = sha[:16]
        if doc_id in self.docs:
            return self.docs[doc_id], False

        filename = filename or path.name
        pages = read_pages(path)
        pieces = chunk_pages(pages)
        vectors = self.embedder.embed([p.text for p in pieces]) if pieces else None

        with self.lock:
            ids = np.arange(self.next_id, self.next_id + len(pieces), dtype="int64")
            if pieces:
                self.index.add_with_ids(vectors, ids)
            for cid, piece in zip(ids.tolist(), pieces):
                self.chunks[cid] = {
                    "chunk_id": cid,
                    "doc_id": doc_id,
                    "filename": filename,
                    "page": piece.page,
                    "collection": collection,
                    "text": piece.text,
                }
            self.next_id += len(pieces)
            info = {
                "doc_id": doc_id,
                "filename": filename,
                "collection": collection,
                "sha256": sha,
                "num_pages": len(pages),
                "num_chunks": len(pieces),
            }
            self.docs[doc_id] = info
            self.version += 1
            self.save()
        return info, True

    def delete_document(self, doc_id: str) -> bool:
        """Remove a document and all of its chunks from the index."""
        with self.lock:
            if doc_id not in self.docs:
                return False
            ids = [cid for cid, c in self.chunks.items() if c["doc_id"] == doc_id]
            if ids:
                self.index.remove_ids(np.array(ids, dtype="int64"))
            for cid in ids:
                del self.chunks[cid]
            del self.docs[doc_id]
            self.version += 1
            self.save()
        return True

    def list_documents(self) -> list[dict]:
        return sorted(self.docs.values(), key=lambda d: d["filename"].lower())

    # ---------- search ----------
    def matching_ids(self, doc_ids: list[str] | None = None, collection: str | None = None) -> set[int] | None:
        """Chunk ids allowed by a filter, or None when there is no filter."""
        if not doc_ids and not collection:
            return None
        return {
            cid
            for cid, c in self.chunks.items()
            if (not doc_ids or c["doc_id"] in doc_ids) and (not collection or c["collection"] == collection)
        }

    def dense_search(self, query: str, k: int, allowed: set[int] | None = None) -> list[tuple[int, float]]:
        """Top-k (chunk_id, cosine similarity). Vectors are normalised, so inner product = cosine."""
        if self.index.ntotal == 0:
            return []
        q = self.embedder.embed([query])
        # With a filter, search everything and filter afterwards. Exact and simple;
        # fine at this scale (flat index over a few thousand chunks).
        fetch = self.index.ntotal if allowed is not None else min(k, self.index.ntotal)
        scores, ids = self.index.search(q, fetch)
        results = []
        for cid, score in zip(ids[0].tolist(), scores[0].tolist()):
            if cid == -1 or (allowed is not None and cid not in allowed):
                continue
            results.append((cid, float(score)))
            if len(results) == k:
                break
        return results
