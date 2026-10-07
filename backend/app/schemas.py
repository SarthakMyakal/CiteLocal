"""Pydantic request and response models for the API."""
from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field


class DocumentOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: str
    filename: str
    collection: str
    num_pages: int
    num_chunks: int
    created_at: datetime


class UploadResult(BaseModel):
    document: DocumentOut
    duplicate: bool


class Citation(BaseModel):
    filename: str
    page: int


class SourceChunk(BaseModel):
    chunk_id: int
    filename: str
    page: int
    text: str
    score: float | None = None


class ChatRequest(BaseModel):
    question: str = Field(min_length=1, max_length=2000)
    session_id: int | None = None
    doc_ids: list[str] | None = None
    collection: str | None = None


class ChatResponse(BaseModel):
    session_id: int
    answer: str
    abstained: bool
    reason: str
    citations: list[Citation]
    sources: list[SourceChunk]
    trace: list[dict]
    latency_ms: float


class SessionOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    title: str
    created_at: datetime


class MessageOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    role: str
    content: str
    citations: list[Citation]
    trace: list[dict]
    created_at: datetime


class Health(BaseModel):
    status: str
    documents: int
    chunks: int
    llm_model: str
    ollama: bool
