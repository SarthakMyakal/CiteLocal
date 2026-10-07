"""Download the open-access arXiv papers used by the evaluation (eval/questions.json).

One-off setup step; the app itself never makes network calls.
Usage: python -m scripts.download_corpus

Note: arXiv serves the latest version of each paper. If a newer version changes the
page layout, the expected pages in eval/questions.json may need re-checking.
"""
import time
import urllib.request

from backend.config import settings

PAPERS = {
    "1706.03762": "Attention Is All You Need.pdf",
    "1810.04805": "BERT.pdf",
    "1907.11692": "RoBERTa.pdf",
    "2106.09685": "LoRA.pdf",
    "2004.12832": "colbert.pdf",
    "2004.04906": "dpr.pdf",
    "2005.11401": "rag.pdf",
    "1908.10084": "sentence-BERT.pdf",
    "1910.10683": "t5.pdf",
    "2010.11929": "vit.pdf",
}


def main() -> None:
    settings.data_dir.mkdir(parents=True, exist_ok=True)
    for arxiv_id, name in PAPERS.items():
        target = settings.data_dir / name
        if target.exists():
            continue
        url = f"https://arxiv.org/pdf/{arxiv_id}"
        print(f"Downloading {url} -> {name}")
        request = urllib.request.Request(url, headers={"User-Agent": "citelocal-eval/1.0"})
        with urllib.request.urlopen(request, timeout=120) as response:
            target.write_bytes(response.read())
        time.sleep(3)  # be polite to arXiv
    print(f"data/ has {len(list(settings.data_dir.glob('*.pdf')))} PDFs.")


if __name__ == "__main__":
    main()
