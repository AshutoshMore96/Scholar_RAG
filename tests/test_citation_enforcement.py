"""Unit tests for citation enforcement in the generator."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from scholar_rag.generate.cited_generator import CitedLiteratureGenerator, CITATION_RE


def make_generator(**kwargs) -> CitedLiteratureGenerator:
    defaults = dict(
        model="test",
        ollama_url="http://localhost:11434",
        self_rag=False,
        enforce_citations=True,
        abstain_threshold=0.1,
    )
    defaults.update(kwargs)
    return CitedLiteratureGenerator(**defaults)


PASSAGES = [
    {"paper_id": "2310.11511", "title": "Dense X Retrieval", "year": 2023, "text": "Propositions improve retrieval precision."},
    {"paper_id": "2212.09561", "title": "HyDE", "year": 2022, "text": "Hypothetical document embeddings bridge the lexical gap."},
]


def test_citation_regex_matches():
    text = "As shown in [2310.11511], proposition chunking improves recall [2212.09561]."
    found = CITATION_RE.findall(text)
    assert "2310.11511" in found
    assert "2212.09561" in found


def test_abstain_on_low_quality():
    gen = make_generator()
    result = gen.generate("what is rag?", PASSAGES, context_quality=0.05)
    assert result.abstained
    assert "INSUFFICIENT EVIDENCE" in result.review


def test_enforce_drops_uncited_sentence():
    gen = make_generator()
    # A review where second sentence has no citation
    review = (
        "Propositions improve retrieval [2310.11511]. "
        "This is an uncited claim with no citation. "
        "HyDE bridges the lexical gap [2212.09561]."
    )
    enforced = gen._enforce_citations(review, PASSAGES)
    assert "uncited claim" not in enforced
    assert "[2310.11511]" in enforced
    assert "[2212.09561]" in enforced


def test_format_context_includes_paper_id():
    gen = make_generator()
    ctx = gen._format_context(PASSAGES)
    assert "[2310.11511]" in ctx
    assert "[2212.09561]" in ctx


def test_generation_result_citations_used():
    gen = make_generator()
    with patch.object(gen, "_generate_review", return_value="Found in [2310.11511]."), \
         patch.object(gen, "_enforce_citations", side_effect=lambda r, p: r):
        result = gen.generate("test query", PASSAGES, context_quality=0.9)
    assert "2310.11511" in result.citations_used
