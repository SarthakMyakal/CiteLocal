"""Turns a PDF file into page-aware text chunks."""
import hashlib
import logging
from dataclasses import dataclass
from pathlib import Path

from langchain_text_splitters import RecursiveCharacterTextSplitter
from pypdf import PdfReader

from backend.config import settings

# pypdf logs noisy font-encoding warnings for many academic PDFs; text extraction still works.
logging.getLogger("pypdf").setLevel(logging.ERROR)


@dataclass
class ChunkText:
    page: int  # 1-based page number, as a person would cite it
    text: str


def file_sha256(path: Path) -> str:
    """Content hash used to detect duplicate uploads, even under a different filename."""
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def read_pages(path: Path) -> list[str]:
    """Extract the text of every page (empty string for image-only pages)."""
    reader = PdfReader(str(path))
    return [(page.extract_text() or "") for page in reader.pages]


def chunk_pages(pages: list[str]) -> list[ChunkText]:
    """Split each page separately so every chunk maps to exactly one page number."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
    )
    chunks = []
    for page_no, text in enumerate(pages, start=1):
        for piece in splitter.split_text(text):
            if piece.strip():
                chunks.append(ChunkText(page=page_no, text=piece))
    return chunks
