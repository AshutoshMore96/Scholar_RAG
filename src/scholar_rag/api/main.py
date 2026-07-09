"""
FastAPI service — exposes the ScholarRAG pipeline over HTTP.

Endpoints:
  POST /ask       — answer a research question
  POST /ingest    — trigger ingestion of new arXiv papers
  GET  /health    — liveness probe
  GET  /papers    — list indexed paper ids + metadata
"""

from __future__ import annotations

import os
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import json as _json

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.responses import FileResponse, StreamingResponse
from loguru import logger
from pydantic import BaseModel, Field

_STATIC = Path(__file__).parent / "static"

from scholar_rag.config import load_config
from scholar_rag.embed.embedder import ChunkEmbedder
from scholar_rag.generate.cited_generator import CitedLiteratureGenerator
from scholar_rag.retrieve.crag import CRAGEvaluator
from scholar_rag.retrieve.engine import RetrievalEngine
from scholar_rag.retrieve.graph_rerank import CitationGraphReranker
from scholar_rag.retrieve.hybrid_rrf import HybridRetriever
from scholar_rag.retrieve.hyde import HyDEQueryExpander
from scholar_rag.retrieve.multi_query import MultiQueryExpander
from scholar_rag.retrieve.rerank import CrossEncoderReranker
from scholar_rag.store.graph_store import CitationGraphStore
from scholar_rag.store.qdrant_store import QdrantStore


# ── Global singletons (loaded once at startup) ──────────────────────── #
_engine: RetrievalEngine | None = None
_generator: CitedLiteratureGenerator | None = None
_deep_generator: CitedLiteratureGenerator | None = None  # points at a GPU Ollama host if OLLAMA_DEEP_URL set
_cfg: dict | None = None
_papers_index: list[dict] | None = None  # distinct-paper corpus index, built once (see _build_papers_index)


def _prewarm() -> None:
    """Load BGE-M3 + reranker into memory and pin the LLM in Ollama, so the first
    real user query is fast (no cold start). Runs in a background thread."""
    try:
        logger.info("Pre-warming models (embedder, reranker, LLM)…")
        ret = _engine.retrieve("retrieval augmented generation", top_k=3)
        _generator.generate("warmup", ret.passages, context_quality=ret.context_quality)
        logger.success("Models pre-warmed — first query will be fast.")
    except Exception as exc:
        logger.warning(f"Pre-warm skipped ({exc}).")


def _build_papers_index() -> list[dict]:
    """
    Scroll the whole collection once and collapse it to the set of distinct
    papers (title, year, venue, citations), sorted by citation count. Cached in
    `_papers_index` so the corpus browser is instant after the first build.
    """
    global _papers_index
    if _papers_index is not None:
        return _papers_index
    papers: dict[str, dict] = {}
    client, coll = _engine.store.client, _engine.store.collection
    offset = None
    while True:
        pts, offset = client.scroll(
            collection_name=coll, limit=4000, offset=offset, with_vectors=False,
            with_payload=["paper_id", "title", "year", "venue", "cited_by_count"],
        )
        for pt in pts:
            p = pt.payload or {}
            pid = p.get("paper_id")
            if pid and pid not in papers:
                papers[pid] = {
                    "paper_id": pid, "title": p.get("title", ""), "year": p.get("year"),
                    "venue": p.get("venue"), "cited_by_count": p.get("cited_by_count", 0) or 0,
                }
        if offset is None:
            break
    _papers_index = sorted(papers.values(),
                           key=lambda x: (x["cited_by_count"], x.get("year") or 0), reverse=True)
    logger.success(f"Corpus index built: {len(_papers_index)} distinct papers.")
    return _papers_index


def _prewarm_corpus() -> None:
    try:
        _build_papers_index()
    except Exception as exc:  # noqa: BLE001
        logger.warning(f"Corpus index build skipped ({exc}).")


def _fast_llm_cfg(cfg_gen: dict) -> dict:
    """
    Resolve the default (fast-mode) generation backend.

    Prefers a hosted API (Groq) when ``DEEP_LLM_API_KEY`` is set — this keeps
    the local box free of a resident Ollama model (important on low-RAM
    machines, where Ollama + BGE-M3 together force swap). Falls back to the
    configured local Ollama when no key is present.
    """
    api_key = os.getenv("DEEP_LLM_API_KEY", "").strip()
    if api_key:
        base_url = (os.getenv("DEEP_LLM_BASE_URL", "").strip()
                    or "https://api.groq.com/openai/v1")
        model = os.getenv("FAST_LLM_MODEL", "").strip() or "llama-3.1-8b-instant"
        return {"base_url": base_url, "api_key": api_key, "model": model}
    return {"base_url": cfg_gen["ollama_url"], "api_key": None, "model": cfg_gen["llm"]}


def _deep_llm_cfg(default_model: str) -> dict | None:
    """
    Resolve the "Deep Retrieval using GPU" backend from the environment.

    Returns None when deep mode is not configured (deep toggle then runs the
    fuller pipeline on the local CPU Ollama).  Otherwise returns
    ``{base_url, api_key, model}`` for either:

      * a hosted OpenAI-compatible API — set ``DEEP_LLM_API_KEY`` (e.g. a Groq
        key); ``DEEP_LLM_BASE_URL`` defaults to Groq's endpoint, and
        ``DEEP_LLM_MODEL`` to a served model; or
      * a self-hosted Ollama GPU host — set ``OLLAMA_DEEP_URL`` (legacy).
    """
    api_key = os.getenv("DEEP_LLM_API_KEY", "").strip()
    base_url = (os.getenv("DEEP_LLM_BASE_URL", "").strip()
                or os.getenv("OLLAMA_DEEP_URL", "").strip())
    if not api_key and not base_url:
        return None
    if api_key and not base_url:
        base_url = "https://api.groq.com/openai/v1"   # sensible default for a key
    model = (os.getenv("DEEP_LLM_MODEL", "").strip()
             or os.getenv("OLLAMA_DEEP_MODEL", "").strip()
             or default_model)
    return {"base_url": base_url, "api_key": api_key or None, "model": model}


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _engine, _generator, _deep_generator, _cfg
    _cfg = load_config()
    logger.info("Initialising ScholarRAG components…")
    _engine, _generator = _build_pipeline(_cfg)
    # Optional GPU/hosted endpoint used by "Deep Retrieval using GPU".
    gen = _cfg["generation"]
    _dc = _deep_llm_cfg(gen["llm"])
    if _dc:
        _deep_generator = CitedLiteratureGenerator(
            model=_dc["model"],
            ollama_url=_dc["base_url"],
            api_key=_dc["api_key"],
            max_tokens=gen["max_tokens"], temperature=gen["temperature"],
            self_rag=gen["self_rag_reflection"], enforce_citations=gen["citation_enforcement"],
            abstain_threshold=gen["abstain_threshold"],
            verify_claims=os.getenv("VERIFY_CLAIMS", "1") == "1",
            verify_model=os.getenv("VERIFY_LLM_MODEL", "").strip() or None,
        )
        _how = "hosted API" if _dc["api_key"] else "GPU Ollama"
        logger.success(f"Deep generation via {_how}: {_dc['base_url']} ({_dc['model']})")
    logger.success("ScholarRAG ready.")
    # warm the heavy models in the background so startup stays responsive
    import threading
    threading.Thread(target=_prewarm, daemon=True).start()
    threading.Thread(target=_prewarm_corpus, daemon=True).start()
    yield


app = FastAPI(
    title="ScholarRAG",
    description="Academic literature-review generator over arXiv papers.",
    version="0.1.0",
    lifespan=lifespan,
)


# ── Request / response models ────────────────────────────────────────── #

class AskRequest(BaseModel):
    query: str = Field(..., min_length=2, max_length=500)
    year_from: int | None = None
    year_to: int | None = None
    min_citations: int | None = None
    top_k: int = Field(default=10, ge=1, le=25)
    deep: bool = Field(default=False, description="Full pipeline: HyDE + multi-query + CRAG (higher quality, slower).")


class Citation(BaseModel):
    paper_id: str
    title: str
    year: int | None
    venue: str | None
    score: float


class AskResponse(BaseModel):
    query: str
    review: str
    citations: list[Citation]
    context_quality: float
    abstained: bool
    latency_ms: int


class IngestRequest(BaseModel):
    categories: list[str] = Field(default=["cs.CL"])
    max_results: int = Field(default=100, le=1000)
    date_from: str | None = None
    query_terms: list[str] | None = Field(
        default=None,
        description=(
            "Optional topic keywords/phrases. Papers must match ANY term in "
            "title or abstract (OR-combined) on top of the category filter. "
            "Use to target a sub-topic, e.g. "
            '["large language model", "RAG", "retrieval-augmented"].'
        ),
    )
    chunking_strategy: str | None = Field(
        default=None,
        description=(
            "Override chunking for this run: 'proposition' (LLM-decomposed, "
            "precise, slow) or 'semantic' (sentence-window packing, no LLM, "
            "much faster). Defaults to configs/default.yaml."
        ),
    )
    add_contextual_header: bool | None = Field(
        default=None,
        description="Override whether to prepend LLM-generated contextual headers.",
    )


# ── Endpoints ────────────────────────────────────────────────────────── #

@app.get("/", include_in_schema=False)
def home():
    """Serve the cinematic single-page UI."""
    return FileResponse(_STATIC / "index.html")


@app.get("/health")
def health():
    return {"status": "ok", "model": os.getenv("OLLAMA_MODEL", "llama3.1:8b")}


# Small in-memory response cache: identical query+filters returns instantly.
from collections import OrderedDict
_ASK_CACHE: "OrderedDict[tuple, AskResponse]" = OrderedDict()
_ASK_CACHE_MAX = 128


@app.post("/ask", response_model=AskResponse)
def ask(req: AskRequest):
    if _engine is None or _generator is None:
        raise HTTPException(503, "Pipeline not initialised.")

    cache_key = (req.query.strip().lower(), req.top_k, req.year_from, req.year_to, req.min_citations, req.deep)
    if cache_key in _ASK_CACHE:
        _ASK_CACHE.move_to_end(cache_key)
        cached = _ASK_CACHE[cache_key].model_copy()
        cached.latency_ms = 0  # served from cache
        return cached

    t0 = time.time()
    meta_filter = {}
    if req.year_from:
        meta_filter["year_from"] = req.year_from
    if req.year_to:
        meta_filter["year_to"] = req.year_to
    if req.min_citations:
        meta_filter["min_citations"] = req.min_citations

    retrieval = _engine.retrieve(req.query, metadata_filter=meta_filter, top_k=req.top_k, deep=req.deep)
    passages = retrieval.passages[: req.top_k]

    gen = _deep_generator if (req.deep and _deep_generator) else _generator
    gen_result = gen.generate(
        req.query, passages, context_quality=retrieval.context_quality
    )

    latency_ms = int((time.time() - t0) * 1000)

    # Surface the retrieved source papers as provenance. Papers the model cited
    # inline are listed first; the rest are the other passages that informed the
    # answer (retrieval-level provenance), so sources are always visible even if
    # a smaller model's inline [paper_id] markers are sparse. On abstention we
    # show no sources — nothing was actually used to answer.
    citations = []
    seen = set()
    cited = set(gen_result.citations_used)
    for inline_first in () if gen_result.abstained else (True, False):
        for p in passages:
            pid = p.get("paper_id", p.get("arxiv_id", ""))
            if not pid or pid in seen:
                continue
            if (pid in cited) != inline_first:
                continue
            citations.append(Citation(
                paper_id=pid,
                title=p.get("title", ""),
                year=p.get("year"),
                venue=p.get("venue"),
                score=round(p.get("_graph_rerank_score", p.get("_rerank_score", 0.0)), 4),
            ))
            seen.add(pid)

    resp = AskResponse(
        query=req.query,
        review=gen_result.review,
        citations=citations,
        context_quality=round(retrieval.context_quality, 3),
        abstained=gen_result.abstained,
        latency_ms=latency_ms,
    )
    # cache successful (non-abstained) answers for instant repeats
    if not gen_result.abstained:
        _ASK_CACHE[cache_key] = resp
        if len(_ASK_CACHE) > _ASK_CACHE_MAX:
            _ASK_CACHE.popitem(last=False)
    return resp


def _passages_to_citations(passages, cited_ids: set, abstained: bool) -> list[Citation]:
    citations, seen = [], set()
    for inline_first in () if abstained else (True, False):
        for p in passages:
            pid = p.get("paper_id", p.get("arxiv_id", ""))
            if not pid or pid in seen:
                continue
            if (pid in cited_ids) != inline_first:
                continue
            citations.append(Citation(
                paper_id=pid, title=p.get("title", ""), year=p.get("year"),
                venue=p.get("venue"),
                score=round(p.get("_graph_rerank_score", p.get("_rerank_score", 0.0)), 4),
            ))
            seen.add(pid)
    return citations


def _build_paper_graph(passages, store, max_nodes: int = 14,
                       edges_per_node: int = 2, min_sim: float = 0.30) -> dict:
    """
    Build a per-query "related-papers" graph for an Obsidian-style force view.

    Nodes are the distinct papers retrieved for this query; edges connect each
    paper to its most semantically-similar neighbours (cosine over the BGE-M3
    dense vectors we already have in Qdrant). This is a *relatedness* graph, not
    a literal citation graph — the corpus has no populated citation edges.
    """
    import numpy as np

    # One node per paper: keep the best-scoring chunk for each.
    best: dict[str, dict] = {}
    # A paper is "raw-reachable" if any of its chunks came from the raw query;
    # papers reached only via HyDE/multi-query expansion (deep mode) are flagged.
    raw_reach: dict[str, bool] = {}
    for p in passages:
        pid = str(p.get("paper_id") or p.get("arxiv_id") or "")
        if not pid:
            continue
        raw_reach[pid] = raw_reach.get(pid, False) or bool(p.get("_from_raw", True))
        sc = float(p.get("_graph_rerank_score", p.get("_rerank_score",
                                                       p.get("_rrf_score", 0.0))))
        if pid not in best or sc > best[pid]["_score"]:
            best[pid] = {
                "id": pid,
                "title": (p.get("title") or pid)[:120],
                "year": p.get("year"),
                "score": round(sc, 3),
                "cited": int(p.get("cited_by_count") or 0),
                "_qid": p.get("_qdrant_id"),
                "_score": sc,
            }
    nodes = list(best.values())[:max_nodes]
    if len(nodes) < 2:
        for n in nodes:
            n["expansion"] = not raw_reach.get(n["id"], True)
            n.pop("_qid", None); n.pop("_score", None)
        return {"nodes": nodes, "edges": []}

    # Fetch dense vectors for these papers' representative chunks.
    qids = [n["_qid"] for n in nodes if n["_qid"]]
    vecs: dict[str, "np.ndarray"] = {}
    try:
        pts = store.client.retrieve(store.collection, ids=qids, with_vectors=["dense"])
        for pt in pts:
            v = pt.vector
            dv = v.get("dense") if isinstance(v, dict) else v
            if dv is not None:
                arr = np.asarray(dv, dtype=float)
                n = np.linalg.norm(arr)
                if n > 0:
                    vecs[str(pt.id)] = arr / n
    except Exception as exc:
        logger.warning(f"Graph vector fetch failed: {exc}")

    # Each node links to its top-`edges_per_node` most similar neighbours.
    usable = [n for n in nodes if str(n["_qid"]) in vecs]
    emap: dict[tuple, float] = {}
    for a in usable:
        va = vecs[str(a["_qid"])]
        sims = sorted(
            ((float(va @ vecs[str(b["_qid"])]), b["id"]) for b in usable if b["id"] != a["id"]),
            reverse=True,
        )
        for s, bid in sims[:edges_per_node]:
            if s >= min_sim:
                key = tuple(sorted((a["id"], bid)))
                emap[key] = max(emap.get(key, 0.0), s)

    for n in nodes:
        n["expansion"] = not raw_reach.get(n["id"], True)
        n.pop("_qid", None); n.pop("_score", None)
    edges = [{"source": k[0], "target": k[1], "weight": round(w, 3)}
             for k, w in emap.items()]
    return {"nodes": nodes, "edges": edges}


@app.post("/ask/stream")
def ask_stream(req: AskRequest):
    """Server-Sent-Events variant of /ask: streams the review token-by-token.
    Emits: `meta` (quality, abstained, citations) → many `token` → `done`."""
    if _engine is None or _generator is None:
        raise HTTPException(503, "Pipeline not initialised.")

    meta_filter = {}
    if req.year_from: meta_filter["year_from"] = req.year_from
    if req.year_to: meta_filter["year_to"] = req.year_to
    if req.min_citations: meta_filter["min_citations"] = req.min_citations

    t0 = time.time()
    retrieval = _engine.retrieve(req.query, metadata_filter=meta_filter, top_k=req.top_k, deep=req.deep)
    passages = retrieval.passages[: req.top_k]
    quality = retrieval.context_quality
    gen = _deep_generator if (req.deep and _deep_generator) else _generator
    abstained = quality < gen.abstain_threshold
    # citations are known from retrieval (provenance); send them upfront
    cites = _passages_to_citations(passages, set(), abstained)
    graph = _build_paper_graph(passages, _engine.store)

    def sse():
        meta = {"context_quality": round(quality, 3), "abstained": abstained,
                "citations": [c.model_dump() for c in cites], "graph": graph}
        yield f"event: meta\ndata: {_json.dumps(meta)}\n\n"
        for tok in gen.stream_review(req.query, passages, quality):
            yield f"event: token\ndata: {_json.dumps({'t': tok})}\n\n"
        yield f"event: done\ndata: {_json.dumps({'latency_ms': int((time.time()-t0)*1000)})}\n\n"

    return StreamingResponse(sse(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


@app.post("/ingest")
def ingest(req: IngestRequest, background_tasks: BackgroundTasks):
    background_tasks.add_task(_run_ingest, req)
    return {"status": "ingestion started", "categories": req.categories}


@app.get("/papers")
def list_papers(limit: int = 60, offset: int = 0, q: str = ""):
    """
    Browse the distinct-paper corpus. `q` filters by title / arXiv id.
    Served from the cached index (built once); `ready` is False only during the
    very first build.
    """
    if _engine is None:
        raise HTTPException(503, "Pipeline not initialised.")
    try:
        idx = _build_papers_index()  # cached after first call
    except Exception as exc:
        raise HTTPException(500, str(exc))
    ql = q.strip().lower()
    matched = ([p for p in idx
                if ql in (p["title"] or "").lower() or ql in p["paper_id"].lower()]
               if ql else idx)
    return {
        "papers": matched[offset: offset + limit],
        "total": len(idx),
        "matched": len(matched),
        "ready": _papers_index is not None,
    }


# ── Pipeline factory ─────────────────────────────────────────────────── #

def _build_pipeline(cfg: dict) -> tuple[RetrievalEngine, CitedLiteratureGenerator]:
    ret = cfg["retrieval"]
    gen = cfg["generation"]

    store = QdrantStore(
        host=cfg["storage"]["qdrant_host"],
        port=int(cfg["storage"]["qdrant_port"]),
        collection=cfg["storage"]["collection"],
    )
    graph = CitationGraphStore()
    embedder = ChunkEmbedder(
        model_name=cfg["embedding"]["model"],
        device=cfg["embedding"]["device"],
        batch_size=cfg["embedding"]["batch_size"],
        enable_colbert=cfg["embedding"]["enable_colbert"],
    )
    # Deep-retrieval LLM steps (HyDE / multi-query / CRAG) run on the deep
    # backend (hosted Groq API or GPU Ollama) when configured — they only fire
    # in deep mode; otherwise they use these instances' local-CPU defaults.
    _dc = _deep_llm_cfg(gen["llm"])
    _deep_kw = (
        {"model": _dc["model"], "ollama_url": _dc["base_url"], "api_key": _dc["api_key"]}
        if _dc else {}
    )
    reranker = CrossEncoderReranker(device=cfg["embedding"]["device"])
    _verify = os.getenv("VERIFY_CLAIMS", "1") == "1"   # LLM claim-grounding on generation
    engine = RetrievalEngine(
        qdrant_store=store,
        graph_store=graph,
        embedder=embedder,
        hyde=HyDEQueryExpander(**_deep_kw),
        multi_query=MultiQueryExpander(n=ret["multi_query_n"], **_deep_kw),
        hybrid=HybridRetriever(
            store=store,
            dense_weight=ret["hybrid_dense_weight"],
            sparse_weight=ret["hybrid_sparse_weight"],
            top_k=ret["top_k_candidates"],
        ),
        reranker=reranker,
        graph_reranker=CitationGraphReranker(
            graph_store=graph,
            alpha=ret["graph_rerank"]["alpha"],
            beta=ret["graph_rerank"]["beta"],
            gamma=ret["graph_rerank"]["gamma"],
        ),
        crag=CRAGEvaluator(quality_threshold=ret["crag_quality_threshold"], **_deep_kw),
        top_k_candidates=ret["top_k_candidates"],
        rerank_top_k=ret["rerank_top_k"],
        hyde_enabled=ret["hyde_enabled"],
        multi_query_n=ret["multi_query_n"],
        crag_enabled=ret.get("crag_enabled", True),
    )
    _fc = _fast_llm_cfg(gen)
    generator = CitedLiteratureGenerator(
        model=_fc["model"],
        ollama_url=_fc["base_url"],
        api_key=_fc["api_key"],
        max_tokens=gen["max_tokens"],
        temperature=gen["temperature"],
        self_rag=gen["self_rag_reflection"],
        enforce_citations=gen["citation_enforcement"],
        abstain_threshold=gen["abstain_threshold"],
        verify_claims=_verify,
        verify_model=os.getenv("VERIFY_LLM_MODEL", "").strip() or None,
    )
    _how = "Groq" if _fc["api_key"] else "local Ollama"
    logger.info(f"Fast generation via {_how}: {_fc['model']} (verify_claims={_verify})")
    return engine, generator


def _run_ingest(req: IngestRequest) -> None:
    from scholar_rag.ingest.pipeline import run_ingestion
    from scholar_rag.transform.chunkers import DocumentChunker
    from scholar_rag.transform.contextual_headers import ContextualHeaderGenerator
    from scholar_rag.transform.propositions import PropositionExtractor
    from scholar_rag.transform.parse_nougat import NougatParser
    from scholar_rag.transform.parse_grobid import GrobidParser
    import pymupdf

    cfg = load_config()
    cfg["ingestion"]["categories"] = req.categories
    cfg["ingestion"]["max_results_per_query"] = req.max_results
    if req.date_from:
        cfg["ingestion"]["date_from"] = req.date_from
    if req.query_terms:
        cfg["ingestion"]["query_terms"] = req.query_terms
    cfg.setdefault("chunking", {})
    if req.chunking_strategy:
        cfg["chunking"]["strategy"] = req.chunking_strategy
    if req.add_contextual_header is not None:
        cfg["chunking"]["add_contextual_header"] = req.add_contextual_header

    records = run_ingestion(cfg)
    ck = cfg.get("chunking", {})
    prop_ex = PropositionExtractor()
    hdr_gen = ContextualHeaderGenerator()
    chunker = DocumentChunker(
        prop_ex,
        hdr_gen,
        max_child_tokens=ck.get("max_chunk_tokens", 256),
        max_parent_tokens=ck.get("parent_chunk_tokens", 1024),
        overlap_tokens=ck.get("overlap_tokens", 32),
        add_contextual_header=ck.get("add_contextual_header", True),
        strategy=ck.get("strategy", "proposition"),
    )
    nougat = NougatParser()
    grobid = GrobidParser()

    store = QdrantStore()
    store.create_collection()
    graph = CitationGraphStore()
    em = cfg.get("embedding", {})
    embedder = ChunkEmbedder(
        model_name=em.get("model", "BAAI/bge-m3"),
        device=em.get("device", "cpu"),
        batch_size=em.get("batch_size", 64),
        enable_colbert=em.get("enable_colbert", False),
        max_length=em.get("max_length", 512),
    )

    citation_map = {}
    parent_map = {}
    all_chunks = []

    for rec in records:
        paper = rec["paper"]
        cit = rec["citation_meta"]
        # Backfill year from the arXiv publication date when OpenAlex has none
        # (brand-new papers often aren't indexed yet), so year filters and the
        # recency prior still work.
        if cit.year is None and paper.published:
            try:
                cit.year = int(paper.published[:4])
            except (ValueError, TypeError):
                pass
        citation_map[paper.arxiv_id] = cit.__dict__
        graph.upsert_paper(paper.arxiv_id, cit.__dict__)
        graph.upsert_edges(paper.arxiv_id, cit.references)

        if paper.pdf_path:
            pdf_path = Path(paper.pdf_path)
            md = nougat.parse(pdf_path) or grobid.parse(pdf_path)
            if md is None:
                doc = pymupdf.open(str(pdf_path))
                md = "\n\n".join(page.get_text() for page in doc)
        else:
            md = f"# {paper.title}\n\n## Abstract\n\n{paper.abstract}"

        chunks, parents = chunker.chunk_document(
            paper_id=paper.arxiv_id,
            title=paper.title,
            markdown_text=md,
            paper_metadata={"year": cit.year, "venue": cit.venue},
        )
        for p in parents:
            parent_map[p.parent_id] = p
        all_chunks.extend(chunks)

    embedded = embedder.embed_chunks(all_chunks)
    store.upsert_chunks(embedded, parent_map, citation_map)
    logger.success("Background ingestion complete.")
