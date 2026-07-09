"""
HyDE — Hypothetical Document Embeddings (Gao et al., 2022).

Strategy: ask the LLM to write a plausible abstract / answer for the query,
then embed *that* hypothetical text instead of (or alongside) the raw query.
The hypothetical document shares vocabulary and phrasing with actual papers,
bridging the lexical gap between casual questions and academic prose.
"""

from __future__ import annotations

import os

from loguru import logger

from scholar_rag.generate.llm import LLMClient


_OLLAMA_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
_MODEL = os.getenv("OLLAMA_MODEL", "llama3.1:8b")

HYDE_PROMPT = """\
You are a research scientist.  Write a concise research abstract (100–150 words)
that would directly answer the following research question.  Be precise and use
academic terminology.  Do NOT say you are writing an abstract — just write the text.

Research question: {query}

Abstract:"""


class HyDEQueryExpander:
    """
    Generates a hypothetical document for a query and returns it as an
    augmented query string to be embedded.
    """

    def __init__(
        self,
        model: str = _MODEL,
        ollama_url: str = _OLLAMA_URL,
        api_key: str | None = None,
    ) -> None:
        self.model = model
        self._llm = LLMClient(model=model, base_url=ollama_url, api_key=api_key,
                              timeout=60.0)

    def expand(self, query: str) -> str:
        """Return the hypothetical document text (falls back to raw query)."""
        try:
            hyp_doc = self._llm.complete(
                HYDE_PROMPT.format(query=query), temperature=0.3, max_tokens=250)
            logger.debug(f"HyDE generated {len(hyp_doc)} chars for query: {query[:60]}")
            return hyp_doc if hyp_doc else query
        except Exception as exc:
            logger.warning(f"HyDE failed: {exc} — using raw query.")
            return query
