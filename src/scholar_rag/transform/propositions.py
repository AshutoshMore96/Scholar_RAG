"""
Proposition-based chunking (Chen et al. "Dense X Retrieval", 2023).

Each paragraph is decomposed by a local LLM into atomic, self-contained
factual statements ("propositions").  This greatly improves retrieval
precision — we retrieve individual claims, not coarse paragraphs.

The LLM call is batched and cached to avoid re-processing on re-runs.
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
_CACHE_DIR = Path(".cache/propositions")

PROPOSITION_PROMPT = """\
You are an expert at decomposing academic text into atomic propositions.

Given the following passage from a research paper, extract all distinct,
self-contained factual propositions.  Each proposition must:
1. Express exactly one fact or claim.
2. Be understandable without needing the original context
   (replace pronouns, resolve references).
3. Use precise academic language.

Output ONLY a JSON array of strings, one proposition per string.
Do NOT include any explanation, preamble, or markdown formatting.

PASSAGE:
{passage}

JSON array:"""


class PropositionExtractor:
    """
    Calls Ollama to decompose passages into propositions.
    Falls back to returning the passage as a single-item list on failure.
    """

    def __init__(
        self,
        model: str = _MODEL,
        ollama_url: str = _OLLAMA_URL,
        cache_dir: Path = _CACHE_DIR,
        batch_size: int = 32,
    ) -> None:
        self.model = model
        self.ollama_url = ollama_url.rstrip("/")
        self.cache_dir = cache_dir
        self.batch_size = batch_size
        self._client = httpx.Client(timeout=120.0)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def extract(self, passage: str) -> list[str]:
        cache_key = hashlib.sha256(passage.encode()).hexdigest()
        cache_file = self.cache_dir / f"{cache_key}.json"
        if cache_file.exists():
            return json.loads(cache_file.read_text())

        props = self._call_llm(passage)
        cache_file.write_text(json.dumps(props, ensure_ascii=False))
        return props

    def extract_batch(self, passages: list[str]) -> list[list[str]]:
        results = []
        for i in range(0, len(passages), self.batch_size):
            batch = passages[i : i + self.batch_size]
            for p in batch:
                results.append(self.extract(p))
        return results

    # ------------------------------------------------------------------ #

    def _call_llm(self, passage: str) -> list[str]:
        prompt = PROPOSITION_PROMPT.format(passage=passage[:3000])
        try:
            resp = self._client.post(
                f"{self.ollama_url}/api/generate",
                json={
                    "model": self.model,
                    "prompt": prompt,
                    "stream": False,
                    "options": {"temperature": 0.0, "num_predict": 1024},
                },
            )
            resp.raise_for_status()
            raw = resp.json().get("response", "").strip()
            # Strip potential markdown code fences
            if raw.startswith("```"):
                raw = raw.split("```")[1].lstrip("json").strip()
                if "```" in raw:
                    raw = raw[: raw.index("```")]
            propositions = json.loads(raw)
            if isinstance(propositions, list) and all(isinstance(p, str) for p in propositions):
                return [p.strip() for p in propositions if p.strip()]
        except Exception as exc:
            logger.warning(f"Proposition extraction failed: {exc}")
        return [passage]  # fallback: treat passage as single proposition
