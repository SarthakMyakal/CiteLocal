"""Ingest every PDF in a folder into the vector store.

Usage:
    python -m scripts.ingest                 # ingest data/*.pdf into collection "default"
    python -m scripts.ingest path/to/pdfs --collection research
"""
import argparse
import time
from pathlib import Path

from backend.config import settings
from backend.rag.models import get_embedder
from backend.rag.store import VectorStore


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("folder", nargs="?", default=str(settings.data_dir))
    parser.add_argument("--collection", default="default")
    args = parser.parse_args()

    store = VectorStore(settings.index_dir, get_embedder())
    for pdf in sorted(Path(args.folder).glob("*.pdf")):
        start = time.perf_counter()
        info, added = store.add_document(pdf, collection=args.collection)
        status = "added" if added else "skipped (duplicate)"
        print(f"{pdf.name}: {status}, {info['num_chunks']} chunks, {time.perf_counter() - start:.1f}s")
    print(f"Index now has {len(store.docs)} documents and {store.index.ntotal} chunks.")


if __name__ == "__main__":
    main()
