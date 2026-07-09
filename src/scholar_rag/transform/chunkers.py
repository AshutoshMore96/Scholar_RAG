"""
Layout-aware + proposition-based chunker.

Produces two complementary representations per document:
  - Child chunks  (propositions or small semantic windows, ≤256 tokens)
    → embedded and indexed in Qdrant
  - Parent chunks (full section windows, ≤1024 tokens)
    → stored in Qdrant payload; returned to the LLM for full context

This "parent-document / hierarchical" pattern ensures retrieval precision
(small chunks) without sacrificing generation quality (full context).
"""

from __future__ import annotations

import re
import uuid
from dataclasses import dataclass, field
from typing import Any

from loguru import logger

from scholar_rag.transform.contextual_headers import ContextualHeaderGenerator
from scholar_rag.transform.propositions import PropositionExtractor


# Rough token estimator (1 token ≈ 4 chars for English academic text)
def _approx_tokens(text: str) -> int:
    return max(1, len(text) // 4)


SECTION_PATTERN = re.compile(r"^#{1,3}\s+(.+)$", re.MULTILINE)


@dataclass
class Chunk:
    chunk_id: str
    paper_id: str
    parent_id: str          # points to the parent section chunk
    text: str               # proposition text (with contextual header prepended)
    raw_text: str           # original proposition without header
    section: str
    section_idx: int
    chunk_idx: int          # position within parent
    token_count: int
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class ParentChunk:
    parent_id: str
    paper_id: str
    section: str
    text: str               # full section window (sent to LLM)
    token_count: int


class DocumentChunker:
    """
    Orchestrates the full chunking pipeline for a single parsed document.
    """

    def __init__(
        self,
        proposition_extractor: PropositionExtractor,
        header_generator: ContextualHeaderGenerator,
        max_child_tokens: int = 256,
        max_parent_tokens: int = 1024,
        overlap_tokens: int = 32,
        add_contextual_header: bool = True,
        strategy: str = "proposition",
    ) -> None:
        self.prop_extractor = proposition_extractor
        self.header_gen = header_generator
        self.max_child = max_child_tokens
        self.max_parent = max_parent_tokens
        self.overlap = overlap_tokens
        self.add_header = add_contextual_header
        # "proposition": LLM decomposes each paragraph into atomic claims (precise
        #   but one LLM call per paragraph — slow on CPU).
        # "semantic"/"fixed": pack sentences into token windows with no LLM calls
        #   (much faster to ingest; slightly coarser retrieval units).
        self.strategy = strategy

    def chunk_document(
        self,
        paper_id: str,
        title: str,
        markdown_text: str,
        paper_metadata: dict[str, Any],
    ) -> tuple[list[Chunk], list[ParentChunk]]:
        """
        Returns (child_chunks, parent_chunks).
        """
        sections = self._split_sections(markdown_text)
        children: list[Chunk] = []
        parents: list[ParentChunk] = []

        for sec_idx, (section_name, section_text) in enumerate(sections):
            parent_windows = self._sliding_parent_windows(section_text)
            for pw_text in parent_windows:
                parent_id = str(uuid.uuid4())
                parents.append(ParentChunk(
                    parent_id=parent_id,
                    paper_id=paper_id,
                    section=section_name,
                    text=pw_text,
                    token_count=_approx_tokens(pw_text),
                ))
                # Split each paragraph in the window into child retrieval units.
                paragraphs = [p.strip() for p in pw_text.split("\n\n") if p.strip()]
                chunk_idx = 0
                for para in paragraphs:
                    if _approx_tokens(para) < 20:
                        continue  # skip captions, stubs
                    if self.strategy == "proposition":
                        units = self.prop_extractor.extract(para)   # LLM per paragraph
                    else:
                        units = self._semantic_units(para)          # no LLM
                    for prop in units:
                        if _approx_tokens(prop) < 8:
                            continue
                        if self.add_header:
                            full_text = self.header_gen.generate(title, section_name, prop)
                        else:
                            full_text = prop
                        children.append(Chunk(
                            chunk_id=str(uuid.uuid4()),
                            paper_id=paper_id,
                            parent_id=parent_id,
                            text=full_text,
                            raw_text=prop,
                            section=section_name,
                            section_idx=sec_idx,
                            chunk_idx=chunk_idx,
                            token_count=_approx_tokens(full_text),
                            metadata={
                                **paper_metadata,
                                "title": title,
                                "section": section_name,
                            },
                        ))
                        chunk_idx += 1

        logger.debug(f"{paper_id}: {len(children)} child chunks, {len(parents)} parent windows")
        return children, parents

    # ------------------------------------------------------------------ #

    def _split_sections(self, markdown: str) -> list[tuple[str, str]]:
        """Split markdown into (section_name, section_text) pairs."""
        parts = SECTION_PATTERN.split(markdown)
        # parts alternates: [preamble, section_name, body, section_name, body, ...]
        sections: list[tuple[str, str]] = []
        if parts[0].strip():
            sections.append(("Introduction", parts[0].strip()))
        i = 1
        while i + 1 < len(parts):
            name = parts[i].strip()
            body = parts[i + 1].strip()
            if body:
                sections.append((name, body))
            i += 2
        return sections or [("Full Text", markdown)]

    def _semantic_units(self, para: str) -> list[str]:
        """
        Split a paragraph into sentence-packed child units of up to
        max_child tokens — no LLM calls. Sentences are kept whole so each
        unit stays semantically coherent.
        """
        sentences = re.split(r"(?<=[.!?])\s+", para)
        units: list[str] = []
        current: list[str] = []
        current_tokens = 0
        for sent in sentences:
            sent = sent.strip()
            if not sent:
                continue
            st = _approx_tokens(sent)
            if current and current_tokens + st > self.max_child:
                units.append(" ".join(current))
                current, current_tokens = [], 0
            current.append(sent)
            current_tokens += st
        if current:
            units.append(" ".join(current))
        return units

    def _sliding_parent_windows(self, text: str) -> list[str]:
        """Slide a token window over section text with overlap."""
        words = text.split()
        if not words:
            return []
        words_per_window = self.max_parent * 4  # rough chars→words
        step = words_per_window - self.overlap * 4
        step = max(step, words_per_window // 2)

        windows = []
        for start in range(0, len(words), step):
            chunk = " ".join(words[start : start + words_per_window])
            windows.append(chunk)
            if start + words_per_window >= len(words):
                break
        return windows
