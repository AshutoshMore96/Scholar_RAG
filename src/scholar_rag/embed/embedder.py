"""
Embedding orchestrator — batches chunks through BGE-M3 and returns
the structured outputs needed for Qdrant upsert.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from loguru import logger
from tqdm import tqdm

from scholar_rag.embed.bge_m3 import BGEM3Embedder, BGEM3Output
from scholar_rag.transform.chunkers import Chunk


@dataclass
class EmbeddedChunk:
    chunk: Chunk
    dense: np.ndarray       # (1024,)
    sparse: dict[int, float]
    colbert: np.ndarray | None  # (seq_len, 128)


class ChunkEmbedder:
    def __init__(
        self,
        model_name: str = "BAAI/bge-m3",
        device: str = "cpu",
        batch_size: int = 64,
        enable_colbert: bool = False,
        max_length: int = 512,
    ) -> None:
        self.embedder = BGEM3Embedder(
            model_name=model_name,
            device=device,
            batch_size=batch_size,
            enable_colbert=enable_colbert,
            max_length=max_length,
        )
        self.batch_size = batch_size

    def embed_chunks(self, chunks: list[Chunk]) -> list[EmbeddedChunk]:
        """Embed a list of Chunk objects; returns EmbeddedChunk list."""
        texts = [c.text for c in chunks]
        embedded: list[EmbeddedChunk] = []

        for i in tqdm(range(0, len(texts), self.batch_size), desc="Embedding", unit="batch"):
            batch_chunks = chunks[i : i + self.batch_size]
            batch_texts = texts[i : i + self.batch_size]
            out: BGEM3Output = self.embedder.encode(batch_texts)

            for j, chunk in enumerate(batch_chunks):
                embedded.append(EmbeddedChunk(
                    chunk=chunk,
                    dense=out.dense[j],
                    sparse=out.sparse[j],
                    colbert=out.colbert[j] if out.colbert else None,
                ))

        logger.info(f"Embedded {len(embedded)} chunks.")
        return embedded

    def embed_query(self, query: str) -> BGEM3Output:
        return self.embedder.encode_query(query)

    def embed_queries(self, queries: list[str]) -> list[BGEM3Output]:
        """
        Embed several query variants (HyDE / multi-query) in a single batched
        forward pass, then split into per-query outputs. Much faster than one
        ``encode`` call per variant — the difference is large on MPS/GPU where
        per-call overhead dominates. Each returned output keeps the single-query
        shape (``.dense[0]`` / ``.sparse[0]``) the retrieval layer expects.
        """
        if not queries:
            return []
        out = self.embedder.encode(queries)
        results: list[BGEM3Output] = []
        for i in range(len(queries)):
            results.append(BGEM3Output(
                dense=out.dense[i : i + 1],                       # (1, 1024)
                sparse=[out.sparse[i]],                           # [dict]
                colbert=[out.colbert[i]] if out.colbert else None,
            ))
        return results
