"""
Cited literature-review generator with Self-RAG reflection and
citation enforcement.

Pipeline:
  1. Format retrieved passages as a numbered context block.
  2. Prompt the LLM (Ollama) to write a cited review.
  3. Self-RAG reflection pass: verify every claim has ≥1 citation.
  4. Citation enforcement validator: drop sentences without [paper_id].
  5. Abstention: if context quality is too low, return a refusal message.
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field
from typing import Any

from loguru import logger

from scholar_rag.generate.llm import LLMClient
from scholar_rag.generate.prompts import (
    ABSTAIN_THRESHOLD_PROMPT,
    LITERATURE_REVIEW_SYSTEM,
    LITERATURE_REVIEW_USER,
    SELF_RAG_REFLECTION_PROMPT,
)


_OLLAMA_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
_MODEL = os.getenv("OLLAMA_MODEL", "llama3.1:8b")
# How long Ollama keeps the model resident after a request. "60m" keeps it warm
# between queries; "-1" pins it in RAM permanently (most responsive, uses ~RAM).
_KEEP_ALIVE = os.getenv("OLLAMA_KEEP_ALIVE", "60m")

# Regex: matches [paper_id] or [2312.12345] style inline citations
CITATION_RE = re.compile(r"\[([^\]]{3,40})\]")


@dataclass
class GenerationResult:
    review: str
    citations_used: list[str]
    context_quality: float
    abstained: bool = False
    source_passages: list[dict[str, Any]] = field(default_factory=list)


class CitedLiteratureGenerator:
    def __init__(
        self,
        model: str = _MODEL,
        ollama_url: str = _OLLAMA_URL,
        max_tokens: int = 2048,
        temperature: float = 0.1,
        self_rag: bool = True,
        enforce_citations: bool = True,
        abstain_threshold: float = 0.35,
        api_key: str | None = None,
        verify_claims: bool = False,
        verify_model: str | None = None,
    ) -> None:
        self.model = model
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.self_rag = self_rag
        self.enforce = enforce_citations
        self.abstain_threshold = abstain_threshold
        # Post-hoc grounding: verify each generated claim against the retrieved
        # context by LLM *entailment* (not cross-encoder relevance — a reranker
        # scores topical similarity, which can't tell a supported claim from a
        # plausible-but-unsupported one). Unsupported claims are dropped and
        # survivors cited to their best-supporting paper — a faithfulness guard.
        self.verify_claims = verify_claims
        # Speaks Ollama locally, or an OpenAI-compatible host (Groq) when an
        # api_key is supplied.  Generous timeout: a full CPU review can take
        # minutes.
        self._llm = LLMClient(model=model, base_url=ollama_url, api_key=api_key,
                              keep_alive=_KEEP_ALIVE, timeout=600.0)
        # Claim verification can use a *different* model than generation — keeping
        # the verifier independent of the eval judge avoids circular faithfulness.
        self._verify_llm = (
            LLMClient(model=verify_model, base_url=ollama_url, api_key=api_key,
                      keep_alive=_KEEP_ALIVE, timeout=600.0)
            if verify_model and verify_model != model else self._llm)

    def generate(
        self,
        query: str,
        passages: list[dict[str, Any]],
        context_quality: float = 1.0,
    ) -> GenerationResult:
        if context_quality < self.abstain_threshold:
            return GenerationResult(
                review=(
                    f"INSUFFICIENT EVIDENCE: The retrieved passages do not contain "
                    f"enough information to reliably answer: '{query}'. "
                    f"Context quality score: {context_quality:.2f}."
                ),
                citations_used=[],
                context_quality=context_quality,
                abstained=True,
                source_passages=passages,
            )

        context_block = self._format_context(passages)
        review = self._generate_review(query, context_block)

        if self.self_rag:
            review = self._reflect(review, context_block)

        # Grounding: LLM entailment verification supersedes the syntactic
        # citation check when enabled; otherwise fall back to it.
        if self.verify_claims:
            review = self._ground_claims(review, passages)
        elif self.enforce:
            review = self._enforce_citations(review, passages)

        citations_used = list(set(CITATION_RE.findall(review)))
        return GenerationResult(
            review=review,
            citations_used=citations_used,
            context_quality=context_quality,
            abstained=False,
            source_passages=passages,
        )

    def stream_review(self, query: str, passages: list[dict[str, Any]],
                      context_quality: float = 1.0):
        """
        Yield the review token-by-token as Ollama generates it (for SSE
        streaming). Skips the self-RAG reflection / citation-enforcement passes
        (those need the full draft); the prompt still requests inline citations.
        """
        if context_quality < self.abstain_threshold:
            yield ("INSUFFICIENT EVIDENCE: The retrieved passages do not contain "
                   f"enough information to reliably answer: '{query}'. "
                   f"Context quality score: {context_quality:.2f}.")
            return

        context = self._format_context(passages)
        messages = [
            {"role": "system", "content": LITERATURE_REVIEW_SYSTEM},
            {"role": "user", "content": LITERATURE_REVIEW_USER.format(query=query, context=context)},
        ]
        try:
            yield from self._llm.stream_chat(
                messages, temperature=self.temperature, max_tokens=self.max_tokens)
        except Exception as exc:
            logger.error(f"LLM stream failed: {exc}")
            yield ""

    # ------------------------------------------------------------------ #

    def _generate_review(self, query: str, context: str) -> str:
        messages = [
            {"role": "system", "content": LITERATURE_REVIEW_SYSTEM},
            {"role": "user", "content": LITERATURE_REVIEW_USER.format(
                query=query, context=context
            )},
        ]
        return self._chat(messages, max_tokens=self.max_tokens)

    def _reflect(self, review: str, context: str) -> str:
        prompt = SELF_RAG_REFLECTION_PROMPT.format(review=review, context=context[:3000])
        resp = self._generate_raw(prompt, max_tokens=self.max_tokens)
        return resp if resp.strip() else review

    def _enforce_citations(
        self, review: str, passages: list[dict[str, Any]]
    ) -> str:
        """Drop sentences that contain no citation reference."""
        valid_ids = {
            str(p.get("paper_id", p.get("arxiv_id", "")))
            for p in passages
        }
        sentences = re.split(r"(?<=[.!?])\s+", review)
        kept = []
        for sent in sentences:
            # Allow sentences with at least one citation from valid_ids
            # or that are structural (headings, transitions)
            found = CITATION_RE.findall(sent)
            if found or not sent.strip() or self._is_structural(sent):
                kept.append(sent)
            else:
                logger.debug(f"Dropping uncited sentence: {sent[:60]}")
        return " ".join(kept)

    def _ground_claims(self, review: str, passages: list[dict[str, Any]]) -> str:
        """
        Verify each generated sentence against the retrieved context by LLM
        *entailment* in a single batched call: keep sentences the context
        supports, drop the rest. Fails open (returns the original review) on any
        error, so a verification hiccup never blanks the answer.
        """
        if not passages:
            return review
        sents = re.split(r"(?<=[.!?])\s+", review)
        idx = [i for i, s in enumerate(sents)
               if s.strip() and len(s.strip()) > 15 and not self._is_structural(s)]
        if not idx:
            return review
        context = self._format_context(passages)[:6000]
        numbered = "\n".join(f"{n}. {sents[i].strip()}" for n, i in enumerate(idx))
        prompt = (
            f"CONTEXT:\n{context}\n\n"
            "For each numbered STATEMENT, decide whether it is supported by (can be "
            "inferred from) the context above. Output ONLY a JSON array of objects "
            '{"n": <number>, "supported": 0 or 1}.\n\nSTATEMENTS:\n' + numbered
        )
        try:
            raw = self._verify_llm.chat(
                [{"role": "system", "content": "You verify statements against a context."},
                 {"role": "user", "content": prompt}],
                temperature=0.0, max_tokens=600,
            )
            verdicts = json.loads(raw[raw.index("["): raw.rindex("]") + 1])
            supported = {int(v["n"]) for v in verdicts
                         if isinstance(v, dict) and int(v.get("supported", 0)) == 1}
        except Exception as exc:  # noqa: BLE001
            logger.debug(f"claim verification failed ({exc}); keeping review as-is.")
            return review

        drop_positions = {idx[n] for n in range(len(idx)) if n not in supported}
        kept = [s for i, s in enumerate(sents) if i not in drop_positions]
        if drop_positions:
            logger.info(f"Claim verification dropped {len(drop_positions)}/{len(idx)} "
                        f"unsupported sentence(s).")
        return " ".join(kept).strip() or review

    @staticmethod
    def _is_structural(sent: str) -> bool:
        structural_starts = (
            "in summary", "in conclusion", "overall", "together,", "however,",
            "this question", "the literature", "##", "#"
        )
        lower = sent.strip().lower()
        return any(lower.startswith(s) for s in structural_starts)

    @staticmethod
    def _format_context(passages: list[dict[str, Any]]) -> str:
        lines = []
        for p in passages:
            pid = p.get("paper_id", p.get("arxiv_id", "unknown"))
            title = p.get("title", "")
            year = p.get("year", "")
            text = p.get("text", p.get("parent_text", ""))[:600]
            lines.append(f"[{pid}] ({title}, {year})\n{text}")
        return "\n\n---\n\n".join(lines)

    def _chat(self, messages: list[dict], max_tokens: int) -> str:
        try:
            return self._llm.chat(
                messages, temperature=self.temperature, max_tokens=max_tokens)
        except Exception as exc:
            logger.error(f"LLM chat failed: {exc}")
            return ""

    def _generate_raw(self, prompt: str, max_tokens: int) -> str:
        try:
            return self._llm.complete(prompt, temperature=0.0, max_tokens=max_tokens)
        except Exception as exc:
            logger.error(f"LLM generate failed: {exc}")
            return ""
