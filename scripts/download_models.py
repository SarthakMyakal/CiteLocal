"""One-time setup: download the embedding and reranker models into the local Hugging Face cache.

After this, the app runs fully offline (HF_HUB_OFFLINE=1 is set by backend/rag/models.py).
The LLM is pulled separately with: ollama pull qwen2.5:3b
Usage: python -m scripts.download_models
"""
import os

os.environ["HF_HUB_OFFLINE"] = "0"  # this script is the one place allowed to download
os.environ["TRANSFORMERS_OFFLINE"] = "0"

from backend.rag.models import get_embedder, get_reranker  # noqa: E402


def main() -> None:
    print("Embedding dim:", get_embedder().dim)
    get_reranker()
    print("Models cached. The app will now load them offline.")


if __name__ == "__main__":
    main()
