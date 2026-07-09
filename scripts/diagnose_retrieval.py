"""
Diagnostic: trace a query through hybrid retrieval → cross-encoder rerank,
to see whether a weak result is a RETRIEVAL (similarity) problem or a
RERANKING problem. Prints, for each query, the top hybrid candidates and
their cross-encoder scores.
"""
from __future__ import annotations

import sys
from scholar_rag.store.qdrant_store import QdrantStore
from scholar_rag.embed.embedder import ChunkEmbedder
from scholar_rag.retrieve.hybrid_rrf import HybridRetriever
from scholar_rag.retrieve.rerank import CrossEncoderReranker

QUERIES = ["Quantization", "LLM Quantization", "quantization of large language models"]

store = QdrantStore()
embedder = ChunkEmbedder(max_length=256)
hybrid = HybridRetriever(store=store, top_k=20)
reranker = CrossEncoderReranker()

for q in QUERIES:
    print("\n" + "=" * 78)
    print(f"QUERY: {q!r}")
    emb = embedder.embed_query(q)

    # --- retrieval (hybrid dense+sparse RRF) ---
    cands = hybrid.retrieve(emb)
    print(f"  hybrid retrieved {len(cands)} candidates. Top 5 by RRF:")
    for c in cands[:5]:
        print(f"    rrf={c.get('_rrf_score',0):.4f} | {(c.get('title') or '')[:52]}")

    # is any candidate actually about quantization?
    quant_hits = [c for c in cands if "quantiz" in (c.get("text","")+c.get("title","")).lower()]
    print(f"  candidates whose text/title mentions 'quantiz': {len(quant_hits)}")

    # --- reranking (cross-encoder) ---
    reranked = reranker.rerank(q, cands, top_k=20)
    print(f"  after cross-encoder rerank. Top 5 by rerank score:")
    for c in reranked[:5]:
        print(f"    rerank={c.get('_rerank_score',0):.4f} | {(c.get('title') or '')[:52]}")
    top = reranked[0].get("_rerank_score", 0.0) if reranked else 0.0
    print(f"  >>> TOP RERANK SCORE = {top:.4f}  (abstain if < 0.35)")

    # best quantization-mentioning candidate's rerank score
    if quant_hits:
        qbest = max((c.get("_rerank_score",0.0) for c in reranked
                     if "quantiz" in (c.get("text","")+c.get("title","")).lower()), default=0.0)
        print(f"  >>> best 'quantiz' passage rerank score = {qbest:.4f}")
