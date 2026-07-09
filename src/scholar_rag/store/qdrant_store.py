"""
Qdrant vector store — manages collection schema and chunk upserts.

Collection schema (named vectors):
  dense   : float32[1024]   — BGE-M3 dense embedding
  sparse  : sparse float    — BGE-M3 lexical weights (BM25-style)
  colbert : float32[N, 128] — optional late-interaction (stored as multiple vectors)

Payload fields per point:
  paper_id, chunk_id, parent_id, title, section, text, year,
  venue, concepts[], cited_by_count, influential_citation_count
"""

from __future__ import annotations

import os
import uuid
from typing import Any

from loguru import logger
from qdrant_client import QdrantClient
from qdrant_client.http import models as qm

from scholar_rag.embed.embedder import EmbeddedChunk
from scholar_rag.transform.chunkers import ParentChunk


_QDRANT_HOST = os.getenv("QDRANT_HOST", "localhost")
_QDRANT_PORT = int(os.getenv("QDRANT_PORT", 6333))
_COLLECTION = os.getenv("QDRANT_COLLECTION", "scholar_rag")

DENSE_DIM = 1024
SPARSE_NAME = "sparse"
DENSE_NAME = "dense"


class QdrantStore:
    def __init__(
        self,
        host: str = _QDRANT_HOST,
        port: int = _QDRANT_PORT,
        collection: str = _COLLECTION,
    ) -> None:
        # If QDRANT_URL is set (e.g. a Qdrant Cloud cluster built on Colab),
        # connect there; otherwise fall back to local host:port (Docker).
        url = os.getenv("QDRANT_URL", "").strip()
        api_key = os.getenv("QDRANT_API_KEY", "").strip() or None
        if url:
            self.client = QdrantClient(url=url, api_key=api_key)
        else:
            self.client = QdrantClient(host=host, port=port)
        self.collection = collection

    # ------------------------------------------------------------------ #
    # Collection management                                               #
    # ------------------------------------------------------------------ #

    def create_collection(self, recreate: bool = False) -> None:
        existing = [c.name for c in self.client.get_collections().collections]
        if self.collection in existing:
            if recreate:
                self.client.delete_collection(self.collection)
            else:
                logger.info(f"Collection '{self.collection}' already exists — skipping creation.")
                return

        self.client.create_collection(
            collection_name=self.collection,
            vectors_config={
                DENSE_NAME: qm.VectorParams(
                    size=DENSE_DIM,
                    distance=qm.Distance.COSINE,
                    on_disk=True,
                ),
            },
            sparse_vectors_config={
                SPARSE_NAME: qm.SparseVectorParams(
                    index=qm.SparseIndexParams(on_disk=False),
                ),
            },
            optimizers_config=qm.OptimizersConfigDiff(
                indexing_threshold=20_000,
            ),
        )
        # Payload indexes for fast metadata-filtered retrieval
        for field_name, field_type in [
            ("paper_id", qm.PayloadSchemaType.KEYWORD),
            ("year", qm.PayloadSchemaType.INTEGER),
            ("cited_by_count", qm.PayloadSchemaType.INTEGER),
            ("influential_citation_count", qm.PayloadSchemaType.INTEGER),
            ("section", qm.PayloadSchemaType.KEYWORD),
            ("concepts", qm.PayloadSchemaType.KEYWORD),
        ]:
            self.client.create_payload_index(
                collection_name=self.collection,
                field_name=field_name,
                field_schema=field_type,
            )
        logger.success(f"Created Qdrant collection '{self.collection}'.")

    # ------------------------------------------------------------------ #
    # Upsert                                                              #
    # ------------------------------------------------------------------ #

    def upsert_chunks(
        self,
        embedded_chunks: list[EmbeddedChunk],
        parent_map: dict[str, ParentChunk],
        citation_map: dict[str, Any],
        batch_size: int = 256,
    ) -> None:
        """Upsert embedded child chunks with full payload."""
        points = []
        for ec in embedded_chunks:
            c = ec.chunk
            cit = citation_map.get(c.paper_id, {})
            parent = parent_map.get(c.parent_id)

            payload = {
                "paper_id": c.paper_id,
                "chunk_id": c.chunk_id,
                "parent_id": c.parent_id,
                "title": c.metadata.get("title", ""),
                "section": c.section,
                "text": c.raw_text,
                "parent_text": parent.text if parent else "",
                "year": cit.get("year"),
                "venue": cit.get("venue"),
                "concepts": cit.get("concepts", []),
                "cited_by_count": cit.get("cited_by_count", 0),
                "influential_citation_count": cit.get("influential_citation_count", 0),
                "arxiv_id": c.paper_id,
            }

            sparse_vec = qm.SparseVector(
                indices=list(ec.sparse.keys()),
                values=list(ec.sparse.values()),
            )

            points.append(qm.PointStruct(
                id=str(uuid.uuid5(uuid.NAMESPACE_DNS, c.chunk_id)),
                vector={
                    DENSE_NAME: ec.dense.tolist(),
                    SPARSE_NAME: sparse_vec,
                },
                payload=payload,
            ))

        for i in range(0, len(points), batch_size):
            batch = points[i : i + batch_size]
            self.client.upsert(collection_name=self.collection, points=batch)
            logger.debug(f"Upserted batch {i//batch_size + 1} ({len(batch)} points)")

        logger.success(f"Upserted {len(points)} chunks into '{self.collection}'.")

    # ------------------------------------------------------------------ #
    # Search primitives (used by retrieval layer)                        #
    # ------------------------------------------------------------------ #

    def dense_search(
        self,
        vector: list[float],
        top_k: int = 50,
        filters: qm.Filter | None = None,
    ) -> list[qm.ScoredPoint]:
        # query_points is the current API (replaces the removed .search()).
        return self.client.query_points(
            collection_name=self.collection,
            query=vector,
            using=DENSE_NAME,
            limit=top_k,
            query_filter=filters,
            with_payload=True,
        ).points

    def sparse_search(
        self,
        sparse_vec: qm.SparseVector,
        top_k: int = 50,
        filters: qm.Filter | None = None,
    ) -> list[qm.ScoredPoint]:
        return self.client.query_points(
            collection_name=self.collection,
            query=qm.SparseVector(indices=sparse_vec.indices, values=sparse_vec.values),
            using=SPARSE_NAME,
            limit=top_k,
            query_filter=filters,
            with_payload=True,
        ).points
