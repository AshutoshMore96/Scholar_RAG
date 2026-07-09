"""
Hybrid retrieval with Reciprocal Rank Fusion (RRF).

Fuses results from:
  1. Dense vector search (BGE-M3 dense)
  2. Sparse / BM25 search (BGE-M3 sparse vectors in Qdrant)
  3. (Optional) ColBERT late-interaction re-scores

RRF formula:  score(d) = Σ_r  1 / (k + rank_r(d))   [Cormack et al., 2009]
k=60 is the standard constant that prevents top-rank outliers from dominating.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any

from qdrant_client.http import models as qm

from scholar_rag.embed.bge_m3 import BGEM3Output
from scholar_rag.store.qdrant_store import QdrantStore

RRF_K = 60


def reciprocal_rank_fusion(
    ranked_lists: list[list[str]],
    k: int = RRF_K,
) -> list[tuple[str, float]]:
    """
    Fuse multiple ranked lists using RRF.

    Parameters
    ----------
    ranked_lists : list of lists of point IDs (best-first)
    k            : smoothing constant

    Returns
    -------
    list of (point_id, rrf_score) sorted descending
    """
    scores: dict[str, float] = defaultdict(float)
    for ranked in ranked_lists:
        for rank, pid in enumerate(ranked, start=1):
            scores[pid] += 1.0 / (k + rank)
    return sorted(scores.items(), key=lambda x: x[1], reverse=True)


class HybridRetriever:
    """
    Executes dense + sparse retrieval and merges via RRF.
    Optionally applies metadata filters (year range, min citations, categories).
    """

    def __init__(
        self,
        store: QdrantStore,
        dense_weight: float = 0.7,
        sparse_weight: float = 0.3,
        top_k: int = 50,
    ) -> None:
        self.store = store
        self.dense_weight = dense_weight
        self.sparse_weight = sparse_weight
        self.top_k = top_k

    def retrieve(
        self,
        query_embedding: BGEM3Output,
        metadata_filter: qm.Filter | None = None,
    ) -> list[dict[str, Any]]:
        """
        Run hybrid retrieval for a single query embedding.
        Returns list of payload dicts (sorted by RRF score, best-first).
        """
        dense_vec = query_embedding.dense[0].tolist()
        sparse_indices = list(query_embedding.sparse[0].keys())
        sparse_values = list(query_embedding.sparse[0].values())
        sparse_qvec = qm.SparseVector(indices=sparse_indices, values=sparse_values)

        dense_results = self.store.dense_search(dense_vec, self.top_k, metadata_filter)
        sparse_results = self.store.sparse_search(sparse_qvec, self.top_k, metadata_filter)

        # Build id→payload maps
        all_points: dict[str, dict] = {}
        dense_ranked = []
        for hit in dense_results:
            pid = str(hit.id)
            all_points[pid] = hit.payload or {}
            all_points[pid]["_qdrant_id"] = pid
            all_points[pid]["_dense_score"] = hit.score
            dense_ranked.append(pid)

        sparse_ranked = []
        for hit in sparse_results:
            pid = str(hit.id)
            if pid not in all_points:
                all_points[pid] = hit.payload or {}
                all_points[pid]["_qdrant_id"] = pid
            all_points[pid]["_sparse_score"] = hit.score
            sparse_ranked.append(pid)

        fused = reciprocal_rank_fusion([dense_ranked, sparse_ranked])
        # Attach RRF scores and return top-k
        results = []
        for pid, rrf_score in fused[: self.top_k]:
            payload = all_points.get(pid, {})
            payload["_rrf_score"] = rrf_score
            results.append(payload)
        return results

    def retrieve_multi_query(
        self,
        query_embeddings: list[BGEM3Output],
        metadata_filter: qm.Filter | None = None,
    ) -> list[dict[str, Any]]:
        """
        Retrieve and fuse results across multiple query embeddings (for
        multi-query expansion or sub-question decomposition).
        """
        all_ranked: list[list[str]] = []
        all_points: dict[str, dict] = {}
        # Provenance: which query variants retrieved each point. By convention the
        # first embedding is the raw query, so `0 in variants` means the raw query
        # reached it; a point retrieved only by variants >0 was surfaced purely by
        # query expansion (HyDE / multi-query) — used to flag "deep-only" results.
        variants_by_pid: dict[str, set] = {}

        for vi, emb in enumerate(query_embeddings):
            hits = self.retrieve(emb, metadata_filter)
            ranked = []
            for h in hits:
                pid = h["_qdrant_id"]
                all_points[pid] = h
                variants_by_pid.setdefault(pid, set()).add(vi)
                ranked.append(pid)
            all_ranked.append(ranked)

        fused = reciprocal_rank_fusion(all_ranked)
        results = []
        for pid, rrf_score in fused[: self.top_k]:
            payload = all_points.get(pid, {})
            payload["_rrf_score"] = rrf_score
            payload["_from_raw"] = 0 in variants_by_pid.get(pid, {0})
            results.append(payload)
        return results
