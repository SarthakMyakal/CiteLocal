"""FastAPI application.

Run from the repository root:
    uvicorn backend.app.main:app --port 8000
"""
import shutil
import tempfile
import time
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path

import httpx
from fastapi import Depends, FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy import func, select
from sqlalchemy.orm import Session as DbSession

from backend.agent.graph import DocumentAgent
from backend.app import schemas
from backend.config import settings
from backend.db import models
from backend.db.database import make_session_factory
from backend.rag.retrieval import Retriever
from backend.rag.store import VectorStore

MAX_UPLOAD_BYTES = 50 * 1024 * 1024


@dataclass
class Services:
    """Everything loaded once at startup and shared by all requests."""

    store: VectorStore
    agent: DocumentAgent
    session_factory: object
    upload_dir: Path


def build_services() -> Services:
    from backend.rag.models import get_embedder, get_llm, get_reranker

    store = VectorStore(settings.index_dir, get_embedder())
    llm = get_llm()
    agent = DocumentAgent(store, Retriever(store, get_reranker()), llm)
    return Services(store, agent, make_session_factory(settings.database_url), settings.upload_dir)


def sync_documents(db: DbSession, store: VectorStore) -> None:
    """Make the documents table match the index (the index is the source of truth for content)."""
    rows = {d.id: d for d in db.scalars(select(models.Document))}
    for doc_id, info in store.docs.items():
        if doc_id not in rows:
            db.add(document_row(info))
    for doc_id, row in rows.items():
        if doc_id not in store.docs:
            db.delete(row)
    db.commit()


def document_row(info: dict) -> models.Document:
    return models.Document(id=info["doc_id"], filename=info["filename"], collection=info["collection"],
                           sha256=info["sha256"], num_pages=info["num_pages"], num_chunks=info["num_chunks"])


def create_app(services: Services | None = None) -> FastAPI:
    """App factory. Tests pass in services with a fake LLM; production builds the real ones."""

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.services = services or build_services()
        app.state.services.upload_dir.mkdir(parents=True, exist_ok=True)
        with app.state.services.session_factory() as db:
            sync_documents(db, app.state.services.store)
        yield

    app = FastAPI(title="CiteLocal", lifespan=lifespan)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    def get_services(request: Request) -> Services:
        return request.app.state.services

    def get_db(svc: Services = Depends(get_services)):
        with svc.session_factory() as db:
            yield db

    # ---------- documents ----------
    @app.post("/documents", response_model=schemas.UploadResult, status_code=201)
    def upload_document(
        file: UploadFile = File(...),
        collection: str = Form("default"),
        svc: Services = Depends(get_services),
        db: DbSession = Depends(get_db),
    ):
        if not (file.filename or "").lower().endswith(".pdf"):
            raise HTTPException(400, "Only PDF files are supported.")
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf", dir=svc.upload_dir) as tmp:
            shutil.copyfileobj(file.file, tmp)
        tmp_path = Path(tmp.name)
        if tmp_path.stat().st_size > MAX_UPLOAD_BYTES:
            tmp_path.unlink()
            raise HTTPException(413, "File is larger than 50 MB.")
        try:
            info, added = svc.store.add_document(tmp_path, collection=collection, filename=Path(file.filename).name)
        except Exception as exc:  # unreadable / corrupt PDF
            tmp_path.unlink(missing_ok=True)
            raise HTTPException(400, f"Could not read PDF: {exc}") from exc

        final_path = svc.upload_dir / f"{info['doc_id']}.pdf"
        if added and not final_path.exists():
            tmp_path.replace(final_path)
        else:
            tmp_path.unlink(missing_ok=True)

        row = db.get(models.Document, info["doc_id"])
        if row is None:
            row = document_row(info)
            db.add(row)
            db.commit()
        return {"document": row, "duplicate": not added}

    @app.get("/documents", response_model=list[schemas.DocumentOut])
    def list_documents(db: DbSession = Depends(get_db)):
        return db.scalars(select(models.Document).order_by(func.lower(models.Document.filename))).all()

    @app.delete("/documents/{doc_id}", status_code=204)
    def delete_document(doc_id: str, svc: Services = Depends(get_services), db: DbSession = Depends(get_db)):
        row = db.get(models.Document, doc_id)
        if row is None and doc_id not in svc.store.docs:
            raise HTTPException(404, "Document not found.")
        svc.store.delete_document(doc_id)
        (svc.upload_dir / f"{doc_id}.pdf").unlink(missing_ok=True)
        if row is not None:
            db.delete(row)
            db.commit()

    # ---------- chat ----------
    @app.post("/chat", response_model=schemas.ChatResponse)
    def chat(req: schemas.ChatRequest, svc: Services = Depends(get_services), db: DbSession = Depends(get_db)):
        if req.session_id is not None:
            session = db.get(models.Session, req.session_id)
            if session is None:
                raise HTTPException(404, "Session not found.")
        else:
            session = models.Session(title=req.question[:60])
            db.add(session)
            db.flush()

        db.add(models.Message(session_id=session.id, role="user", content=req.question))
        start = time.perf_counter()
        result = svc.agent.run(req.question, doc_ids=req.doc_ids, collection=req.collection)
        latency_ms = (time.perf_counter() - start) * 1000

        db.add(models.Message(session_id=session.id, role="assistant", content=result["answer"],
                              citations=result["citations"], trace=result["trace"]))
        db.add(models.QueryLog(session_id=session.id, question=req.question, latency_ms=latency_ms,
                               chunk_ids=[c["chunk_id"] for c in result["chunks"]],
                               abstained=result["abstained"], reason=result["reason"]))
        db.commit()

        sources = [] if result["abstained"] else [
            {"chunk_id": c["chunk_id"], "filename": c["filename"], "page": c["page"],
             "text": c["text"], "score": c.get("rerank_score")}
            for c in result["chunks"]
        ]
        return {"session_id": session.id, "answer": result["answer"], "abstained": result["abstained"],
                "reason": result["reason"], "citations": result["citations"], "sources": sources,
                "trace": result["trace"], "latency_ms": latency_ms}

    # ---------- sessions ----------
    @app.get("/sessions", response_model=list[schemas.SessionOut])
    def list_sessions(db: DbSession = Depends(get_db)):
        return db.scalars(select(models.Session).order_by(models.Session.id.desc())).all()

    @app.get("/sessions/{session_id}/messages", response_model=list[schemas.MessageOut])
    def session_messages(session_id: int, db: DbSession = Depends(get_db)):
        session = db.get(models.Session, session_id)
        if session is None:
            raise HTTPException(404, "Session not found.")
        return session.messages

    # ---------- health ----------
    @app.get("/health", response_model=schemas.Health)
    def health(svc: Services = Depends(get_services)):
        try:
            ollama_ok = httpx.get(f"{settings.ollama_base_url}/api/tags", timeout=2).status_code == 200
        except httpx.HTTPError:
            ollama_ok = False
        return {"status": "ok", "documents": len(svc.store.docs), "chunks": svc.store.index.ntotal,
                "llm_model": settings.llm_model, "ollama": ollama_ok}

    return app


app = create_app()
