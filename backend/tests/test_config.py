"""Sanity checks on the settings file."""
from backend.config import Settings


def test_defaults_are_sane():
    s = Settings()
    assert s.chunk_overlap < s.chunk_size
    assert s.final_top_k <= s.candidate_k
    assert s.llm_temperature == 0.0


def test_env_override(monkeypatch):
    monkeypatch.setenv("PDA_FINAL_TOP_K", "7")
    assert Settings().final_top_k == 7
