"""
BGE-M3 multi-representation embedder.

BGE-M3 (BAAI, 2024) is unique in producing three types of embeddings from
a single model pass:
  - dense  : standard 1024-d vector for semantic similarity
  - sparse : lexical (BM25-style) weights over vocabulary tokens
  - colbert: per-token multi-vectors for late interaction

We use FlagEmbedding's BGEM3FlagModel which handles all three.
For CPU-only environments dense+sparse is the practical default;
ColBERT is enabled only when GPU memory permits.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any

import numpy as np
from loguru import logger


_MODEL_NAME = os.getenv("BGE_M3_MODEL", "BAAI/bge-m3")


@dataclass
class BGEM3Output:
    dense: np.ndarray           # shape (n, 1024) float32
    sparse: list[dict[int, float]]  # list of {token_id: weight} dicts
    colbert: list[np.ndarray] | None = None  # list of (seq_len, 128) arrays


class BGEM3Embedder:
    """
    Thin wrapper around FlagEmbedding's BGEM3FlagModel.

    Usage
    -----
    embedder = BGEM3Embedder(device="cpu", enable_colbert=False)
    out = embedder.encode(["text1", "text2"])
    # out.dense  → numpy array shape (2, 1024)
    # out.sparse → list of 2 sparse dicts
    """

    def __init__(
        self,
        model_name: str = _MODEL_NAME,
        device: str = "cpu",
        batch_size: int = 64,
        enable_colbert: bool = False,
        max_length: int = 512,
    ) -> None:
        self.model_name = model_name
        self.device = device
        self.batch_size = batch_size
        self.enable_colbert = enable_colbert
        self.max_length = max_length
        self._model: Any = None  # lazy load

    def _load(self) -> None:
        if self._model is not None:
            return
        try:
            from FlagEmbedding import BGEM3FlagModel
            from scholar_rag.config import resolve_device
            self.device = resolve_device(self.device)
            logger.info(f"Loading {self.model_name} on {self.device}…")
            self._model = BGEM3FlagModel(
                self.model_name,
                use_fp16=(self.device != "cpu"),
                device=self.device,
            )
            logger.success(f"BGE-M3 loaded.")
        except ImportError:
            logger.error("FlagEmbedding not installed.  pip install FlagEmbedding")
            raise

    def encode(self, texts: list[str]) -> BGEM3Output:
        self._load()
        all_dense: list[np.ndarray] = []
        all_sparse: list[dict[int, float]] = []
        all_colbert: list[np.ndarray] | None = [] if self.enable_colbert else None

        for i in range(0, len(texts), self.batch_size):
            batch = texts[i : i + self.batch_size]
            out = self._model.encode(
                batch,
                batch_size=len(batch),
                max_length=self.max_length,
                return_dense=True,
                return_sparse=True,
                return_colbert_vecs=self.enable_colbert,
            )
            all_dense.append(out["dense_vecs"])
            all_sparse.extend(out["lexical_weights"])
            if self.enable_colbert and all_colbert is not None:
                all_colbert.extend(out["colbert_vecs"])

        dense = np.vstack(all_dense).astype(np.float32)
        return BGEM3Output(
            dense=dense,
            sparse=all_sparse,
            colbert=all_colbert,
        )

    def encode_query(self, query: str) -> BGEM3Output:
        """Single-query variant with query-side prefix expected by BGE-M3."""
        return self.encode([query])
