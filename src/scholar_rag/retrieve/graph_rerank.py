"""
Citation-graph-aware reranker.

Blends the cross-encoder relevance score with two citation-graph signals:

  final_score = α · rerank_score
              + β · log(1 + cited_by_count)      [OpenAlex citation count]
              + γ · recency_score

where recency = 1 / (1 + years_since_publication).

α, β, γ are tunable hyper-parameters (default: 0.60 / 0.25 / 0.15).
This surfaces highly cited or very recent papers that the cross-encoder
might rank slightly lower simply because of phrasing differences.
"""

from __future__ import annotations

import math
from typing import Any

from scholar_rag.store.graph_store import CitationGraphStore


class CitationGraphReranker:
    def __init__(
        self,
        graph_store: CitationGraphStore,
        alpha: float = 0.60,
        beta: float = 0.25,
        gamma: float = 0.15,
        current_year: int = 2026,
    ) -> None:
        self.graph = graph_store
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.current_year = current_year

    def rerank(
        self,
        candidates: list[dict[str, Any]],
        top_k: int | None = None,
    ) -> list[dict[str, Any]]:
        """
        Expects each candidate to have:
          _rerank_score   (from cross-encoder, 0–1)
          paper_id        (arXiv id)
          year            (int, optional)
          cited_by_count  (int, optional)   [OpenAlex citation count]

        Returns candidates sorted by combined score.
        """
        for c in candidates:
            rerank_s = c.get("_rerank_score", 0.5)
            paper_id = c.get("paper_id", c.get("arxiv_id", ""))

            # Prefer payload values; fall back to graph DB lookup
            cited = c.get("cited_by_count")
            year = c.get("year")
            if cited is None or year is None:
                priors = self.graph.get_influence_prior(paper_id, self.current_year)
                log_inf = priors["log_influence"]
                recency = priors["recency"]
            else:
                log_inf = math.log1p(max(0, cited))
                years_old = max(0, self.current_year - (year or self.current_year))
                recency = 1.0 / (1.0 + years_old)

            final = (
                self.alpha * rerank_s
                + self.beta * log_inf
                + self.gamma * recency
            )
            c["_graph_rerank_score"] = round(final, 5)
            c["_log_influence"] = round(log_inf, 4)
            c["_recency"] = round(recency, 4)

        ranked = sorted(candidates, key=lambda x: x["_graph_rerank_score"], reverse=True)
        return ranked[:top_k] if top_k else ranked
