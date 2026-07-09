"""Unit tests for the document chunker."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from scholar_rag.transform.chunkers import DocumentChunker, _approx_tokens


@pytest.fixture
def chunker():
    prop_mock = MagicMock()
    # Proposition extractor returns the passage split into two sentences
    prop_mock.extract.side_effect = lambda p: [s.strip() for s in p.split(".") if s.strip()]

    hdr_mock = MagicMock()
    hdr_mock.generate.side_effect = lambda title, sec, chunk: (
        f"This passage is from {title}, {sec} section.\n\n{chunk}"
    )
    return DocumentChunker(prop_mock, hdr_mock)


SAMPLE_MD = """\
# Attention Is All You Need

## Introduction

The dominant sequence transduction models are based on complex recurrent or convolutional
neural networks. We propose the Transformer, a model architecture eschewing recurrence.

## Methods

The Transformer uses attention mechanisms exclusively. Multi-head attention allows the
model to jointly attend to information from different representation subspaces.
"""


def test_chunk_count(chunker):
    children, parents = chunker.chunk_document(
        paper_id="1706.03762",
        title="Attention Is All You Need",
        markdown_text=SAMPLE_MD,
        paper_metadata={"year": 2017},
    )
    assert len(children) > 0
    assert len(parents) > 0


def test_chunk_has_paper_id(chunker):
    children, _ = chunker.chunk_document(
        paper_id="1706.03762",
        title="Attention Is All You Need",
        markdown_text=SAMPLE_MD,
        paper_metadata={},
    )
    for c in children:
        assert c.paper_id == "1706.03762"


def test_parent_child_linkage(chunker):
    children, parents = chunker.chunk_document(
        paper_id="test",
        title="Test",
        markdown_text=SAMPLE_MD,
        paper_metadata={},
    )
    parent_ids = {p.parent_id for p in parents}
    for c in children:
        assert c.parent_id in parent_ids, f"Child {c.chunk_id} has orphaned parent_id"


def test_contextual_header_prepended(chunker):
    children, _ = chunker.chunk_document(
        paper_id="test",
        title="Test Paper",
        markdown_text=SAMPLE_MD,
        paper_metadata={},
    )
    for c in children:
        assert "This passage is from" in c.text


def test_approx_tokens():
    assert _approx_tokens("hello world") == 2  # 11 chars / 4 ≈ 2
    assert _approx_tokens("") == 1  # minimum 1
