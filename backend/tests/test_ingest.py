"""Multi-document ingestion, dedup, incremental add, delete, persistence."""
import shutil

from backend.rag.ingest import chunk_pages, read_pages
from backend.rag.models import get_embedder
from backend.rag.store import VectorStore


def test_pdf_pages_are_read_and_chunked(pdf_dir):
    pages = read_pages(pdf_dir / "solar.pdf")
    assert len(pages) == 2 and "photovoltaic" in pages[0]
    chunks = chunk_pages(pages)
    assert [c.page for c in chunks] == [1, 2]


def test_chunks_carry_metadata(store):
    for chunk in store.chunks.values():
        assert {"doc_id", "filename", "page", "collection", "text"} <= chunk.keys()
    assert len(store.docs) == 3
    assert store.index.ntotal == len(store.chunks) == 6


def test_duplicate_is_skipped_even_with_new_name(store, pdf_dir, tmp_path):
    copy = tmp_path / "renamed.pdf"
    shutil.copy(pdf_dir / "solar.pdf", copy)
    _, added = store.add_document(copy)
    assert added is False
    assert len(store.docs) == 3


def test_incremental_add_keeps_existing_vectors(store, tmp_path):
    from backend.tests.conftest import make_pdf

    before = store.index.ntotal
    info, added = store.add_document(make_pdf(tmp_path / "new.pdf", ["Tea is brewed from leaves."]))
    assert added and store.index.ntotal == before + 1
    assert info["num_chunks"] == 1


def test_delete_removes_document_and_chunks(store):
    doc_id = next(d["doc_id"] for d in store.list_documents() if d["filename"] == "baking.pdf")
    assert store.delete_document(doc_id)
    assert store.index.ntotal == 4
    assert all(c["doc_id"] != doc_id for c in store.chunks.values())
    assert store.delete_document(doc_id) is False


def test_index_persists_to_disk(store):
    reloaded = VectorStore(store.index_dir, get_embedder())
    assert reloaded.index.ntotal == store.index.ntotal
    assert reloaded.docs.keys() == store.docs.keys()
    assert reloaded.dense_search("sourdough yeast", k=1)[0][0] == store.dense_search("sourdough yeast", k=1)[0][0]
