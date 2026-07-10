"""
Contextual chunk headers (contextual retrieval).

Prepends a short LLM-generated context line to each chunk:
  "This passage is from the Methods section of <title>, discussing <topic>."

This reduces retrieval ambiguity for short proposition chunks that lack
surrounding context.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import httpx
from loguru import logger


_OLLAMA_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
_MODEL = os.getenv("OLLAMA_MODEL", "llama3.1:8b")
_CACHE_DIR = Path(".cache/headers")

HEADER_PROMPT = """\
Given the following metadata and chunk from a research paper, write a single
concise sentence (≤ 25 words) that contextualises the chunk so it can be
understood in isolation.  Start with "This passage is from…".

Paper title: {title}
Section: {section}
Chunk: {chunk}

Contextual sentence:"""


class ContextualHeaderGenerator:
    def __init__(
        self,
        model: str = _MODEL,
        ollama_url: str = _OLLAMA_URL,
        cache_dir: Path = _CACHE_DIR,
    ) -> None:
        self.model = model
        self.ollama_url = ollama_url.rstrip("/")
        self.cache_dir = cache_dir
        self._client = httpx.Client(timeout=60.0)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def generate(self, title: str, section: str, chunk: str) -> str:
        """Return chunk prepended with its contextual header."""
        cache_key = hashlib.sha256(f"{title}|{section}|{chunk[:200]}".encode()).hexdigest()
        cache_file = self.cache_dir / f"{cache_key}.txt"
        if cache_file.exists():
            header = cache_file.read_text().strip()
            return f"{header}\n\n{chunk}"

        header = self._call_llm(title, section, chunk)
        cache_file.write_text(header)
        return f"{header}\n\n{chunk}"

    def _call_llm(self, title: str, section: str, chunk: str) -> str:
        prompt = HEADER_PROMPT.format(title=title, section=section, chunk=chunk[:500])
        try:
            resp = self._client.post(
                f"{self.ollama_url}/api/generate",
                json={
                    "model": self.model,
                    "prompt": prompt,
                    "stream": False,
                    "options": {"temperature": 0.0, "num_predict": 60},
                },
            )
            resp.raise_for_status()
            return resp.json().get("response", "").strip()
        except Exception as exc:
            logger.warning(f"Header generation failed: {exc}")
            # graceful fallback: construct deterministic header
            return f"This passage is from {title}, {section} section."
