"""
Unified LLM client for the generation / retrieval LLM steps.

Speaks two wire protocols behind one interface, chosen automatically:

  * **Ollama** (default — local CPU): POST ``/api/generate`` & ``/api/chat``.
  * **OpenAI-compatible** (Groq / OpenRouter / Together / vLLM / …): POST
    ``/chat/completions`` with a Bearer key.  Selected whenever an ``api_key``
    is supplied.

This lets "Deep Retrieval using GPU" point at a free hosted inference API
(e.g. Groq's ``llama-3.3-70b-versatile``) with no self-hosted GPU, while the
default path keeps talking to local Ollama.  Callers keep their own graceful
fallbacks — these methods raise on transport/HTTP errors.
"""

from __future__ import annotations

import json
from typing import Iterator

import httpx


class LLMClient:
    def __init__(
        self,
        *,
        model: str,
        base_url: str,
        api_key: str | None = None,
        keep_alive: str = "60m",
        timeout: float = 600.0,
    ) -> None:
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.api_key = (api_key or "").strip() or None
        self.keep_alive = keep_alive
        self._client = httpx.Client(timeout=timeout)

    @property
    def is_openai(self) -> bool:
        """OpenAI-compatible wire format is used whenever a key is present."""
        return self.api_key is not None

    @property
    def _headers(self) -> dict:
        return {"Authorization": f"Bearer {self.api_key}"} if self.api_key else {}

    # ------------------------------------------------------------------ #
    # Single-prompt completion
    # ------------------------------------------------------------------ #
    def complete(self, prompt: str, *, temperature: float = 0.1,
                 max_tokens: int = 512) -> str:
        if self.is_openai:
            # OpenAI-compatible hosts have no /generate — wrap as a chat turn.
            return self.chat(
                [{"role": "user", "content": prompt}],
                temperature=temperature, max_tokens=max_tokens,
            )
        resp = self._client.post(
            f"{self.base_url}/api/generate",
            json={
                "model": self.model, "prompt": prompt, "stream": False,
                "keep_alive": self.keep_alive,
                "options": {"temperature": temperature, "num_predict": max_tokens},
            },
        )
        resp.raise_for_status()
        return resp.json().get("response", "").strip()

    # ------------------------------------------------------------------ #
    # Chat completion
    # ------------------------------------------------------------------ #
    def chat(self, messages: list[dict], *, temperature: float = 0.1,
             max_tokens: int = 1024) -> str:
        if self.is_openai:
            resp = self._client.post(
                f"{self.base_url}/chat/completions",
                headers=self._headers,
                json={
                    "model": self.model, "messages": messages, "stream": False,
                    "temperature": temperature, "max_tokens": max_tokens,
                },
            )
            resp.raise_for_status()
            choices = resp.json().get("choices") or [{}]
            return (choices[0].get("message", {}).get("content") or "").strip()
        resp = self._client.post(
            f"{self.base_url}/api/chat",
            json={
                "model": self.model, "messages": messages, "stream": False,
                "keep_alive": self.keep_alive,
                "options": {"temperature": temperature, "num_predict": max_tokens},
            },
        )
        resp.raise_for_status()
        return resp.json().get("message", {}).get("content", "").strip()

    # ------------------------------------------------------------------ #
    # Streaming chat completion (token generator)
    # ------------------------------------------------------------------ #
    def stream_chat(self, messages: list[dict], *, temperature: float = 0.1,
                    max_tokens: int = 1024) -> Iterator[str]:
        if self.is_openai:
            with self._client.stream(
                "POST", f"{self.base_url}/chat/completions",
                headers=self._headers,
                json={
                    "model": self.model, "messages": messages, "stream": True,
                    "temperature": temperature, "max_tokens": max_tokens,
                },
            ) as resp:
                resp.raise_for_status()
                for line in resp.iter_lines():
                    if not line or not line.startswith("data:"):
                        continue
                    data = line[len("data:"):].strip()
                    if data == "[DONE]":
                        break
                    try:
                        obj = json.loads(data)
                    except json.JSONDecodeError:
                        continue
                    delta = (obj.get("choices") or [{}])[0].get("delta", {})
                    tok = delta.get("content", "")
                    if tok:
                        yield tok
            return
        with self._client.stream(
            "POST", f"{self.base_url}/api/chat",
            json={
                "model": self.model, "messages": messages, "stream": True,
                "keep_alive": self.keep_alive,
                "options": {"temperature": temperature, "num_predict": max_tokens},
            },
        ) as resp:
            resp.raise_for_status()
            for line in resp.iter_lines():
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    continue
                tok = (obj.get("message") or {}).get("content", "")
                if tok:
                    yield tok
                if obj.get("done"):
                    break
