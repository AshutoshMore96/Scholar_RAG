"""Unit tests for RRF fusion logic."""

from __future__ import annotations

from scholar_rag.retrieve.hybrid_rrf import reciprocal_rank_fusion


def test_rrf_basic():
    list1 = ["a", "b", "c"]
    list2 = ["b", "a", "d"]
    result = dict(reciprocal_rank_fusion([list1, list2]))
    # "a" appears at rank 1 in list1 and rank 2 in list2
    # "b" appears at rank 2 in list1 and rank 1 in list2 → same score as "a"
    assert result["a"] == result["b"]
    assert result["c"] > 0
    assert result["d"] > 0


def test_rrf_ordering():
    # Item consistently at top across all lists should have highest score
    list1 = ["top", "x", "y"]
    list2 = ["top", "y", "z"]
    list3 = ["top", "x", "z"]
    result = dict(reciprocal_rank_fusion([list1, list2, list3]))
    assert result["top"] == max(result.values())


def test_rrf_single_list():
    ranked = ["p1", "p2", "p3"]
    result = dict(reciprocal_rank_fusion([ranked]))
    # Scores should be strictly decreasing
    scores = [result[p] for p in ranked]
    assert scores == sorted(scores, reverse=True)


def test_rrf_empty():
    result = reciprocal_rank_fusion([[], []])
    assert result == []


def test_rrf_k_constant():
    k = 60
    result = dict(reciprocal_rank_fusion([["only"]], k=k))
    assert abs(result["only"] - 1.0 / (k + 1)) < 1e-9
