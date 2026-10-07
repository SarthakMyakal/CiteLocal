"""Loads the three models once per process and reuses them.

`lru_cache` turns each loader into a singleton: the first call loads the model,
later calls return the same object. Loading MiniLM takes a few seconds, so doing
it once at startup (not per request, as the original script did) matters.
"""
import os
from functools import lru_cache

import numpy as np

from backend.config import settings

# Privacy: stop huggingface_hub from checking the internet for model updates.
# Models must already be in the local cache (python -m scripts.download_models).
if settings.hf_offline:
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")


class Embedder:
    """Wraps a sentence-transformers model and returns L2-normalised vectors."""

    def __init__(self, model_name: str):
        from sentence_transformers import SentenceTransformer

        self.model = SentenceTransformer(model_name, device="cpu")
        self.dim = self.model.get_embedding_dimension()

    def embed(self, texts: list[str]) -> np.ndarray:
        vectors = self.model.encode(texts, batch_size=32, normalize_embeddings=True, show_progress_bar=False)
        return np.asarray(vectors, dtype="float32")


@lru_cache(maxsize=1)
def get_embedder() -> Embedder:
    return Embedder(settings.embedding_model)


@lru_cache(maxsize=1)
def get_reranker():
    """Cross-encoder that scores (query, chunk) pairs jointly. Small enough for CPU."""
    from sentence_transformers import CrossEncoder

    return CrossEncoder(settings.reranker_model, device="cpu")


@lru_cache(maxsize=1)
def get_llm():
    """Chat model served by the local Ollama daemon. Temperature 0 for repeatable answers."""
    from langchain_ollama import ChatOllama

    return ChatOllama(
        model=settings.llm_model,
        temperature=settings.llm_temperature,
        base_url=settings.ollama_base_url,
        keep_alive=settings.llm_keep_alive,
    )
