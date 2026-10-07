"""Shared test fixtures: tiny generated PDFs and a store backed by the real embedder."""
from pathlib import Path

import pytest

from backend.rag.models import get_embedder
from backend.rag.store import VectorStore


def make_pdf(path: Path, pages: list[str]) -> Path:
    """Write a minimal valid PDF with one text line per page (no extra libraries needed)."""
    objects = ["<< /Type /Catalog /Pages 2 0 R >>", None, "<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>"]
    kids = []
    for text in pages:
        safe = text.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")
        stream = f"BT /F1 11 Tf 40 750 Td ({safe}) Tj ET"
        objects.append(f"<< /Length {len(stream)} >>\nstream\n{stream}\nendstream")
        content_no = len(objects)
        objects.append(
            f"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
            f"/Resources << /Font << /F1 3 0 R >> >> /Contents {content_no} 0 R >>"
        )
        kids.append(f"{len(objects)} 0 R")
    objects[1] = f"<< /Type /Pages /Kids [{' '.join(kids)}] /Count {len(kids)} >>"

    out = b"%PDF-1.4\n"
    offsets = []
    for i, obj in enumerate(objects, start=1):
        offsets.append(len(out))
        out += f"{i} 0 obj\n{obj}\nendobj\n".encode("latin-1")
    xref = len(out)
    out += f"xref\n0 {len(objects) + 1}\n0000000000 65535 f \n".encode()
    out += "".join(f"{o:010d} 00000 n \n" for o in offsets).encode()
    out += f"trailer\n<< /Size {len(objects) + 1} /Root 1 0 R >>\nstartxref\n{xref}\n%%EOF\n".encode()
    path.write_bytes(out)
    return path


@pytest.fixture
def pdf_dir(tmp_path) -> Path:
    """Three small PDFs on clearly different topics."""
    d = tmp_path / "pdfs"
    d.mkdir()
    make_pdf(d / "solar.pdf", [
        "Solar panels convert sunlight into electricity using photovoltaic cells.",
        "The inverter changes direct current from the panels into alternating current.",
    ])
    make_pdf(d / "baking.pdf", [
        "Sourdough bread rises because wild yeast ferments the dough overnight.",
        "Bake the loaf at 230 degrees Celsius for forty minutes with steam.",
    ])
    make_pdf(d / "football.pdf", [
        "A football match lasts ninety minutes split into two halves.",
        "The offside rule stops attackers waiting behind the last defender.",
    ])
    return d


@pytest.fixture
def store(tmp_path, pdf_dir) -> VectorStore:
    s = VectorStore(tmp_path / "index", get_embedder())
    s.add_document(pdf_dir / "solar.pdf", collection="energy")
    s.add_document(pdf_dir / "baking.pdf", collection="food")
    s.add_document(pdf_dir / "football.pdf", collection="sport")
    return s
