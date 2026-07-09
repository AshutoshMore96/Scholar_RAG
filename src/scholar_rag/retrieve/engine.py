"""
ScholarRAG retrieval engine — orchestrates the full query-time pipeline:

  query → HyDE → multi-query expansion → embed all queries
        → hybrid RRF retrieval → cross-encoder rerank
        → citation-graph rerank → CRAG quality check
        → (optional retry if poor quality)
        → return top-k passages for generation
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any

from loguru import logger
from qdrant_client.http import models as qm

from scholar_rag.embed.embedder import ChunkEmbedder
from scholar_rag.retrieve.crag import CRAGEvaluator
from scholar_rag.retrieve.graph_rerank import CitationGraphReranker
from scholar_rag.retrieve.hybrid_rrf import HybridRetriever
from scholar_rag.retrieve.hyde import HyDEQueryExpander
from scholar_rag.retrieve.multi_query import MultiQueryExpander
from scholar_rag.retrieve.rerank import CrossEncoderReranker
from scholar_rag.store.graph_store import CitationGraphStore
from scholar_rag.store.qdrant_store import QdrantStore


@dataclass
class RetrievalResult:
    passages: list[dict[str, Any]]
    context_quality: float
    query_variants: list[str] = field(default_factory=list)
    reformulated: bool = False


class RetrievalEngine:
    def __init__(
        self,
        qdrant_store: QdrantStore,
        graph_store: CitationGraphStore,
        embedder: ChunkEmbedder,
        hyde: HyDEQueryExpander,
        multi_query: MultiQueryExpander,
        hybrid: HybridRetriever,
        reranker: CrossEncoderReranker,
        graph_reranker: CitationGraphReranker,
        crag: CRAGEvaluator,
        top_k_candidates: int = 50,
        rerank_top_k: int = 10,
        hyde_enabled: bool = True,
        multi_query_n: int = 3,
        crag_enabled: bool = True,
    ) -> None:
        self.store = qdrant_store
        self.graph = graph_store
        self.embedder = embedder
        self.hyde = hyde
        self.multi_query = multi_query
        self.hybrid = hybrid
        self.reranker = reranker
        self.graph_reranker = graph_reranker
        self.crag = crag
        self.top_k = top_k_candidates
        self.rerank_top_k = rerank_top_k
        self.hyde_enabled = hyde_enabled
        self.multi_query_n = multi_query_n
        self.crag_enabled = crag_enabled

    def retrieve(
        self,
        query: str,
        metadata_filter: dict | None = None,
        top_k: int | None = None,
        deep: bool = False,
    ) -> RetrievalResult:
        """
        Full retrieval pipeline for a query string.

        metadata_filter keys: year_from, year_to, min_citations, categories
        top_k: how many final passages to return (defaults to config rerank_top_k).
        deep: force the full pipeline on for this query — HyDE + multi-query
              expansion + CRAG corrective retrieval (more LLM calls, higher
              quality, slower). Overrides the CPU-fast defaults.
        """
        qdrant_filter = self._build_filter(metadata_filter or {})
        final_k = top_k or self.rerank_top_k
        use_hyde = self.hyde_enabled or deep
        use_crag = self.crag_enabled or deep
        mq_n = 3 if deep else self.multi_query_n

        # ── Step 1: query expansion (each step is an optional LLM call) ── #
        texts = [query]
        if use_hyde:
            texts.append(self.hyde.expand(query))
        if mq_n > 1:
            texts.extend(self.multi_query.expand(query, n=mq_n))
        all_texts = list(dict.fromkeys(texts))
        logger.debug(f"Query variants: {len(all_texts)}")

        # ── Step 2: embed all variants (single batched pass) ─────────── #
        embeddings = self.embedder.embed_queries(all_texts)

        # ── Step 3: hybrid retrieval + RRF ──────────────────────────── #
        candidates = self.hybrid.retrieve_multi_query(embeddings, qdrant_filter)
        logger.debug(f"Hybrid retrieval returned {len(candidates)} candidates.")

        if not candidates:
            return RetrievalResult(passages=[], context_quality=0.0, query_variants=all_texts)

        # ── Step 4: cross-encoder reranking ─────────────────────────── #
        reranked = self.reranker.rerank(query, candidates, top_k=self.top_k)

        # ── Step 5: citation-graph reranking ────────────────────────── #
        final = self.graph_reranker.rerank(reranked, top_k=final_k)

        # ── Step 6: quality gate ─────────────────────────────────────── #
        # CRAG's LLM self-scoring + reformulation is non-deterministic and slow
        # on a small CPU model (the same query can flip between a great answer and
        # an abstention). When disabled, quality is driven purely by the
        # deterministic, calibrated cross-encoder score. Enable on GPU / a larger
        # judge model for the full corrective-RAG behaviour.
        crag_quality = 1.0
        reformulated = False
        if use_crag:
            needs_retry, crag_quality = self.crag.needs_reformulation(query, final)
            if needs_retry:
                logger.info(f"CRAG triggered retry (score={crag_quality:.2f}) — reformulating…")
                new_query = self.crag.reformulate(query)
                new_emb = [self.embedder.embed_query(new_query)]
                new_cands = self.hybrid.retrieve_multi_query(new_emb, qdrant_filter)
                if new_cands:
                    new_reranked = self.reranker.rerank(query, new_cands, top_k=self.top_k)
                    final = self.graph_reranker.rerank(new_reranked, top_k=final_k)
                    crag_quality = self.crag.score(query, final)
                reformulated = True

        # Expand child→parent: replace chunk text with parent window for generation
        quality = self._grounded_quality(crag_quality, final)
        final = self._expand_to_parent(final)
        return RetrievalResult(
            passages=final,
            context_quality=quality,
            query_variants=all_texts,
            reformulated=reformulated,
        )

    def _grounded_quality(self, crag_quality: float, passages: list[dict]) -> float:
        """
        Combine the CRAG LLM judge with the cross-encoder relevance score.

        The reranker score is the *calibrated* primary signal: a truly relevant
        passage scores high (~0.7+), an off-topic one low (~0.15). The CRAG LLM
        judge is used only as a **veto**: when it agrees the context is relevant
        (>= 0.5) we trust the reranker score directly, so deep mode does not
        report a lower quality than fast mode for identical retrieval. Only when
        the judge flags the context as weak (< 0.5) do we take the MIN, letting
        it pull quality down toward abstention. Skipped if the reranker fell back
        to RRF order (its scores would not be on this scale).
        """
        if not passages or not self.reranker.available:
            return crag_quality
        top_rerank = max(p.get("_rerank_score", 0.0) for p in passages)
        if crag_quality >= 0.5:
            return round(top_rerank, 3)
        return round(min(crag_quality, top_rerank), 3)

    # ------------------------------------------------------------------ #

    @staticmethod
    def _build_filter(meta: dict) -> qm.Filter | None:
        must: list = []

        year_from = meta.get("year_from")
        year_to = meta.get("year_to")
        if year_from is not None or year_to is not None:
            in_range = qm.FieldCondition(
                key="year", range=qm.Range(gte=year_from, lte=year_to)
            )
            # Papers whose year is unknown (OpenAlex enrichment gap — common for
            # brand-new arXiv papers) must NOT be silently dropped by a year
            # filter, or most of the corpus disappears. Include them via OR.
            must.append(qm.Filter(should=[
                in_range,
                qm.IsNullCondition(is_null=qm.PayloadField(key="year")),
            ]))

        # 0 means "no minimum" — don't add a redundant condition for it.
        if meta.get("min_citations"):
            must.append(qm.FieldCondition(
                key="cited_by_count", range=qm.Range(gte=meta["min_citations"])
            ))

        return qm.Filter(must=must) if must else None

    @staticmethod
    def _expand_to_parent(passages: list[dict]) -> list[dict]:
        """Replace passage text with the parent section window if available."""
        expanded = []
        for p in passages:
            parent_text = p.get("parent_text", "")
            if parent_text:
                p = {**p, "text": parent_text}
            expanded.append(p)
        return expanded
