"""
Multi-query expansion and query decomposition.

Generates N paraphrases / sub-questions from the original query so that
retrieval covers a broader semantic neighbourhood.  Particularly valuable
for multi-part research questions ("compare A vs B for task C").
"""

from __future__ import annotations

import json
import os

from loguru import logger

from scholar_rag.generate.llm import LLMClient


_OLLAMA_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
_MODEL = os.getenv("OLLAMA_MODEL", "llama3.1:8b")

EXPANSION_PROMPT = """\
You are an expert research query rewriter.  Given the research question below,
generate {n} distinct paraphrases that preserve the meaning but use different
academic vocabulary and phrasing.  These will be used to retrieve relevant papers.

Output ONLY a JSON array of {n} strings.  No explanations.

Research question: {query}

JSON array:"""

DECOMPOSE_PROMPT = """\
You are an expert at breaking down complex research questions.  Decompose the
following question into {n} simpler, independent sub-questions that together
cover the full scope of the original question.

Output ONLY a JSON array of {n} strings.  No explanations.

Research question: {query}

JSON array:"""


class MultiQueryExpander:
    def __init__(
        self,
        model: str = _MODEL,
        ollama_url: str = _OLLAMA_URL,
        n: int = 3,
        api_key: str | None = None,
    ) -> None:
        self.model = model
        self.n = n
        self._llm = LLMClient(model=model, base_url=ollama_url, api_key=api_key,
                              timeout=60.0)

    def expand(self, query: str, n: int | None = None) -> list[str]:
        """Return n paraphrases + the original query."""
        variants = self._call(EXPANSION_PROMPT, query, n or self.n)
        all_queries = list(dict.fromkeys([query] + variants))  # deduplicate, preserve order
        return all_queries

    def decompose(self, query: str) -> list[str]:
        """Return sub-questions for a complex multi-part query."""
        parts = self._call(DECOMPOSE_PROMPT, query, self.n)
        return parts or [query]

    # ------------------------------------------------------------------ #

    def _call(self, template: str, query: str, n: int) -> list[str]:
        prompt = template.format(query=query, n=n)
        try:
            raw = self._llm.complete(prompt, temperature=0.4, max_tokens=512)
            if raw.startswith("```"):
                raw = raw.split("```")[1].lstrip("json").strip()
                if "```" in raw:
                    raw = raw[: raw.index("```")]
            parsed = json.loads(raw)
            if isinstance(parsed, list) and all(isinstance(q, str) for q in parsed):
                return [q.strip() for q in parsed if q.strip()]
        except Exception as exc:
            logger.warning(f"Multi-query expansion failed: {exc}")
        return []
