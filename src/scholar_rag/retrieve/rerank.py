"""
Cross-encoder reranker.

A cross-encoder jointly scores (query, passage) and produces a relevance
score that is far more accurate than cosine similarity between independent
embeddings.  We apply it to the top-k RRF candidates.

Two backends behind one interface, chosen automatically:

  * **Hosted API** (default when ``JINA_API_KEY`` is set) — Jina's reranker
    endpoint.  No local model, no RAM cost — important on small machines where
    a 2 GB local reranker would push the box into swap and slow everything.
  * **Local** — sentence-transformers' CrossEncoder (e.g. bge-reranker-v2-m3),
    used when no API key is present.

If the active backend fails, it degrades gracefully: candidates are returned
in their incoming (RRF) order rather than raising, so a reranker hiccup never
breaks the whole query.
"""

from __future__ import annotations

import os
from typing import Any

import httpx
from loguru import logger


_RERANKER_MODEL = os.getenv("BGE_RERANKER_MODEL", "BAAI/bge-reranker-v2-m3")
_JINA_URL = "https://api.jina.ai/v1/rerank"


class CrossEncoderReranker:
    """
    Reranks RRF candidates, via the Jina API (if ``JINA_API_KEY`` is set) or a
    local sentence-transformers CrossEncoder.

    Usage
    -----
    reranker = CrossEncoderReranker()
    ranked = reranker.rerank(query, candidates, top_k=10)
    """

    def __init__(
        self,
        model_name: str = _RERANKER_MODEL,
        device: str = "cpu",
        batch_size: int = 32,
        max_length: int = 512,
        api_key: str | None = None,
        api_model: str | None = None,
    ) -> None:
        self.model_name = model_name
        self.device = device
        self.batch_size = batch_size
        self.max_length = max_length
        self._model: Any = None
        self._unavailable = False   # set if local loading fails, to skip retries
        # Hosted-API mode: no local model, no RAM cost.
        self.api_key = (api_key if api_key is not None
                        else os.getenv("JINA_API_KEY", "")).strip() or None
        self.api_model = (api_model or os.getenv("JINA_RERANKER_MODEL",
                          "jina-reranker-v2-base-multilingual"))
        self._api_client: httpx.Client | None = None

    @property
    def is_api(self) -> bool:
        return self.api_key is not None

    @property
    def available(self) -> bool:
        """True when a *calibrated* backend is active (API, or a loaded local
        cross-encoder) — i.e. scores are real relevance, not an RRF-order
        fallback. Used by the quality gate."""
        if self.is_api:
            return True
        return self._model is not None and not self._unavailable

    def _load(self) -> None:
        if self.is_api:
            if self._api_client is None:
                self._api_client = httpx.Client(timeout=30.0)
                logger.info(f"Reranker: Jina API ({self.api_model}).")
            return
        if self._model is not None or self._unavailable:
            return
        try:
            from sentence_transformers import CrossEncoder
            from scholar_rag.config import resolve_device
            self.device = resolve_device(self.device)
            logger.info(f"Loading reranker {self.model_name} on {self.device}…")
            self._model = CrossEncoder(
                self.model_name,
                max_length=self.max_length,
                device=self.device,
            )
            logger.success("Reranker loaded.")
        except Exception as exc:
            # Missing dep or model/version issue — degrade to RRF order.
            logger.warning(f"Reranker unavailable ({exc}); falling back to RRF order.")
            self._unavailable = True

    def rerank(
        self,
        query: str,
        candidates: list[dict[str, Any]],
        top_k: int = 10,
        text_key: str = "text",
    ) -> list[dict[str, Any]]:
        """
        Score each (query, candidate_text) pair and return the top_k
        candidates sorted by relevance (best first).  On any failure, returns
        the candidates in their incoming (RRF) order.
        """
        if not candidates:
            return []
        self._load()

        if self.is_api:
            ranked = self._rerank_api(query, candidates, text_key)
            if ranked is not None:
                return ranked[:top_k]
            # API failed → RRF-order fallback
            for c in candidates:
                c.setdefault("_rerank_score", c.get("_rrf_score", 0.0))
            return candidates[:top_k]

        if self._model is None:  # local reranker unavailable → keep RRF order
            for c in candidates:
                c.setdefault("_rerank_score", c.get("_rrf_score", 0.0))
            return candidates[:top_k]

        try:
            pairs = [[query, c.get(text_key, "")[:1024]] for c in candidates]
            scores = self._model.predict(
                pairs, batch_size=self.batch_size, show_progress_bar=False
            )
            for i, c in enumerate(candidates):
                c["_rerank_score"] = float(scores[i])
            ranked = sorted(candidates, key=lambda x: x["_rerank_score"], reverse=True)
            return ranked[:top_k]
        except Exception as exc:
            logger.warning(f"Reranking failed ({exc}); falling back to RRF order.")
            for c in candidates:
                c.setdefault("_rerank_score", c.get("_rrf_score", 0.0))
            return candidates[:top_k]

    def _rerank_api(
        self, query: str, candidates: list[dict[str, Any]], text_key: str
    ) -> list[dict[str, Any]] | None:
        """Rerank via the Jina API. Returns sorted candidates, or None on failure."""
        docs = [c.get(text_key, "")[:1024] for c in candidates]
        try:
            resp = self._api_client.post(
                _JINA_URL,
                headers={"Authorization": f"Bearer {self.api_key}",
                         "Content-Type": "application/json"},
                json={"model": self.api_model, "query": query,
                      "documents": docs, "top_n": len(docs)},
            )
            resp.raise_for_status()
            results = resp.json().get("results", [])
            if not results:
                return None
            # Map API relevance scores back onto the candidate dicts by index.
            for r in results:
                idx = r.get("index")
                if idx is None or idx >= len(candidates):
                    continue
                candidates[idx]["_rerank_score"] = float(r.get("relevance_score", 0.0))
            for c in candidates:
                c.setdefault("_rerank_score", 0.0)
            return sorted(candidates, key=lambda x: x["_rerank_score"], reverse=True)
        except Exception as exc:
            logger.warning(f"Jina rerank failed ({exc}); falling back to RRF order.")
            return None
