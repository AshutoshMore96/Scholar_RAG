"""
Retrieval evaluation metrics — nDCG@k, Recall@k, MRR.

Designed to run against BEIR-format datasets (SciFact, NFCorpus, etc.)
and against our golden set.
"""

from __future__ import annotations

import math
from typing import Any


def ndcg_at_k(
    retrieved_ids: list[str],
    relevant_ids: set[str],
    k: int = 10,
) -> float:
    """Normalised Discounted Cumulative Gain at k."""
    dcg = sum(
        1.0 / math.log2(rank + 2)
        for rank, pid in enumerate(retrieved_ids[:k])
        if pid in relevant_ids
    )
    ideal = sum(
        1.0 / math.log2(rank + 2)
        for rank in range(min(k, len(relevant_ids)))
    )
    return dcg / ideal if ideal > 0 else 0.0


def recall_at_k(
    retrieved_ids: list[str],
    relevant_ids: set[str],
    k: int = 20,
) -> float:
    if not relevant_ids:
        return 0.0
    hit = len(set(retrieved_ids[:k]) & relevant_ids)
    return hit / len(relevant_ids)


def mean_reciprocal_rank(
    retrieved_ids: list[str],
    relevant_ids: set[str],
) -> float:
    for rank, pid in enumerate(retrieved_ids, start=1):
        if pid in relevant_ids:
            return 1.0 / rank
    return 0.0


def evaluate_retrieval(
    queries: list[dict[str, Any]],
    retrieve_fn: Any,
    k_values: list[int] | None = None,
) -> dict[str, float]:
    """
    queries: list of {"query": str, "relevant_paper_ids": list[str]}
    retrieve_fn: callable(query_str) → list of payload dicts
    """
    if k_values is None:
        k_values = [5, 10, 20]

    agg: dict[str, list[float]] = {f"ndcg@{k}": [] for k in k_values}
    agg.update({f"recall@{k}": [] for k in k_values})
    agg["mrr"] = []

    for q in queries:
        results = retrieve_fn(q["query"])
        retrieved_ids = [r.get("paper_id", r.get("arxiv_id", "")) for r in results]
        relevant = set(q.get("relevant_paper_ids", []))
        for k in k_values:
            agg[f"ndcg@{k}"].append(ndcg_at_k(retrieved_ids, relevant, k))
            agg[f"recall@{k}"].append(recall_at_k(retrieved_ids, relevant, k))
        agg["mrr"].append(mean_reciprocal_rank(retrieved_ids, relevant))

    return {k: sum(v) / len(v) for k, v in agg.items() if v}
