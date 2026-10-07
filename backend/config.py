"""All settings in one place.

Every value can be overridden with an environment variable prefixed with PDA_
(for example PDA_LLM_MODEL=qwen2.5:7b) or a local .env file.
"""
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

ROOT_DIR = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="PDA_", env_file=ROOT_DIR / ".env", extra="ignore")

    # Models (all run locally)
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    llm_model: str = "qwen2.5:3b"
    llm_temperature: float = 0.0
    ollama_base_url: str = "http://localhost:11434"
    llm_keep_alive: str = "30m"  # keep the model in RAM between questions (Ollama default is 5m)
    hf_offline: bool = True  # never contact the Hugging Face Hub at runtime (run scripts.download_models once)

    # Chunking
    chunk_size: int = 1000
    chunk_overlap: int = 200

    # Retrieval
    baseline_top_k: int = 3  # the original main.py similarity search
    candidate_k: int = 20  # candidates fetched from each retriever before reranking
    final_top_k: int = 5  # chunks kept after reranking
    rrf_k: int = 60  # reciprocal rank fusion constant

    # Hallucination guards
    relevance_threshold: float = -0.11  # min top reranker score; calibrated by scripts/calibrate.py (docs/EVAL.md)
    max_rewrites: int = 2

    # API
    cors_origins: list[str] = ["http://localhost:5173", "http://127.0.0.1:5173", "http://localhost:4173"]

    # Paths
    data_dir: Path = ROOT_DIR / "data"
    storage_dir: Path = ROOT_DIR / "storage"

    @property
    def index_dir(self) -> Path:
        return self.storage_dir / "index"

    @property
    def upload_dir(self) -> Path:
        return self.storage_dir / "uploads"

    @property
    def database_url(self) -> str:
        return f"sqlite:///{(self.storage_dir / 'app.db').as_posix()}"


settings = Settings()
