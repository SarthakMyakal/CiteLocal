"""Relevance gate, citation validation, groundedness and the guarded pipeline."""
from backend.rag import guards
from backend.rag.pipeline import answer_question
from backend.tests.fakes import FakeLLM, FakeRetriever

CHUNKS = [
    {"chunk_id": 1, "filename": "solar.pdf", "page": 2, "text": "The inverter converts DC to AC.", "rerank_score": 5.0},
    {"chunk_id": 2, "filename": "solar.pdf", "page": 1, "text": "Panels use photovoltaic cells.", "rerank_score": 1.0},
]


def test_relevance_gate():
    assert guards.passes_relevance_gate(CHUNKS, threshold=2.0)
    assert not guards.passes_relevance_gate(CHUNKS, threshold=6.0)
    assert not guards.passes_relevance_gate([], threshold=-100)


def test_valid_citations_kept_invalid_dropped():
    answer = "The inverter converts DC to AC [solar.pdf, p. 2]. It also cures colds [solar.pdf, p. 9]."
    cleaned, valid, invalid = guards.validate_citations(answer, CHUNKS)
    assert valid == [{"filename": "solar.pdf", "page": 2}]
    assert invalid == [{"filename": "solar.pdf", "page": 9}]
    assert "p. 9" not in cleaned and "[solar.pdf, p. 2]" in cleaned


def test_passage_ids_map_to_file_and_page():
    cleaned, valid, invalid = guards.validate_citations("DC becomes AC [S1]. Cells [s2]. Made up [S7].", CHUNKS)
    assert cleaned == "DC becomes AC [solar.pdf, p. 2]. Cells [solar.pdf, p. 1]. Made up."
    assert valid == [{"filename": "solar.pdf", "page": 2}, {"filename": "solar.pdf", "page": 1}]
    assert invalid == [{"id": "S7"}]


def test_paper_reference_numbers_are_not_citations():
    cleaned, valid, invalid = guards.validate_citations("As shown in [36], it works [S1].", CHUNKS)
    assert "[36]" in cleaned and len(valid) == 1 and invalid == []


def test_citation_formats_are_normalised():
    _, valid, _ = guards.validate_citations("A [Solar.pdf, page 1] B [solar.pdf, 2]", CHUNKS)
    assert [c["page"] for c in valid] == [1, 2]


def test_groundedness_parses_yes_no():
    assert guards.is_grounded(FakeLLM(["YES"]), "q", "x", CHUNKS)
    assert not guards.is_grounded(FakeLLM(["NO, the claim is missing."]), "q", "x", CHUNKS)


def test_pipeline_answers_with_citations():
    llm = FakeLLM(["The inverter converts DC to AC [solar.pdf, p. 2].", "YES"])
    out = answer_question(FakeRetriever(CHUNKS), llm, "What does the inverter do?", threshold=0)
    assert not out["abstained"]
    assert out["citations"] == [{"filename": "solar.pdf", "page": 2}]


def test_pipeline_gate_skips_llm():
    llm = FakeLLM([])
    out = answer_question(FakeRetriever(CHUNKS), llm, "Who won the 1966 World Cup?", threshold=10)
    assert out["abstained"] and out["reason"] == "relevance_gate"
    assert llm.calls == 0


def test_pipeline_abstains_when_not_grounded():
    llm = FakeLLM(["Inverters were invented in 1820 [solar.pdf, p. 2].", "NO"])
    out = answer_question(FakeRetriever(CHUNKS), llm, "When were inverters invented?", threshold=0)
    assert out["abstained"] and out["answer"] == guards.NOT_FOUND
