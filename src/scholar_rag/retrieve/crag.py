"""
CRAG — Corrective Retrieval Augmented Generation (Yan et al., 2024).

A lightweight retrieval-quality evaluator that scores the retrieved context
against the query.  If quality < threshold, it triggers query reformulation
and a second retrieval pass before passing context to the generator.

Scoring: the evaluator LLM rates context relevance on a 0–1 scale.
If below config threshold (default 0.4), the query is reformulated and
retrieval is retried once.
"""

from __future__ import annotations

import os

from loguru import logger

from scholar_rag.generate.llm import LLMClient


_OLLAMA_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
_MODEL = os.getenv("OLLAMA_MODEL", "llama3.1:8b")

EVAL_PROMPT = """\
You are a retrieval quality evaluator.

Query: {query}

Retrieved passages (top 3 shown):
{passages}

On a scale from 0.0 to 1.0, how well do these passages answer the query?
- 1.0 = fully relevant, multiple on-topic passages
- 0.5 = partially relevant, some useful information
- 0.0 = completely irrelevant

Output ONLY a single floating-point number between 0.0 and 1.0.

Score:"""

REFORMULATE_PROMPT = """\
The retrieved documents for the following query were not sufficiently relevant.
Rewrite the query to be more specific and use vocabulary from academic paper abstracts.

Original query: {query}

Rewritten query (one line only):"""


class CRAGEvaluator:
    """
    Scores retrieval quality and optionally triggers reformulation.
    """

    def __init__(
        self,
        model: str = _MODEL,
        ollama_url: str = _OLLAMA_URL,
        quality_threshold: float = 0.40,
        api_key: str | None = None,
    ) -> None:
        self.model = model
        self.threshold = quality_threshold
        self._llm = LLMClient(model=model, base_url=ollama_url, api_key=api_key,
                              timeout=60.0)

    def score(self, query: str, candidates: list[dict]) -> float:
        """Returns a relevance quality score in [0, 1]."""
        snippets = "\n\n".join(
            f"[{i+1}] {c.get('text', '')[:300]}" for i, c in enumerate(candidates[:3])
        )
        prompt = EVAL_PROMPT.format(query=query, passages=snippets)
        try:
            raw = self._llm.complete(prompt, temperature=0.0, max_tokens=10) or "0.5"
            return max(0.0, min(1.0, float(raw.split()[0])))
        except Exception as exc:
            logger.warning(f"CRAG scoring failed: {exc}")
            return 0.5  # neutral fallback

    def needs_reformulation(self, query: str, candidates: list[dict]) -> tuple[bool, float]:
        score = self.score(query, candidates)
        return score < self.threshold, score

    def reformulate(self, query: str) -> str:
        prompt = REFORMULATE_PROMPT.format(query=query)
        try:
            out = self._llm.complete(prompt, temperature=0.3, max_tokens=100)
            return (out or query).strip().split("\n")[0]
        except Exception as exc:
            logger.warning(f"CRAG reformulation failed: {exc}")
            return query
