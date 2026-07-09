# ScholarRAG — Design Decisions & Trade-off Analysis

> A record of *why* this system looks the way it does. Every component below was
> chosen against alternatives, under explicit constraints, with failure modes in
> mind. This document is written the way I would defend the architecture in a
> design review.

---

## 0. Guiding constraints (the box we designed inside)

Before any tool selection, I fixed the non-negotiable constraints. Most bad RAG
architectures come from skipping this step and reaching for the most powerful
component instead of the most *appropriate* one.

| Constraint | Consequence for design |
|---|---|
| **$0 budget, no paid APIs** | Rules out OpenAI/Cohere embeddings, Pinecone, hosted rerankers, GPT-4 judge. Everything must be self-hostable. |
| **Runs on a single machine, CPU-friendly** | Models must fit in ~16 GB RAM; no 70B LLMs; embedding/rerank must be CPU-tolerant. |
| **Academic corpus (math, citations, long docs)** | Parsing must preserve equations/tables; retrieval must handle exact symbol matching; authority signals matter. |
| **Answers must be traceable** | Generation cannot be a black box — every claim needs a citation back to a source passage. |
| **Reproducible & measurable** | Every config change must be quantifiable; "it feels better" is not acceptable evidence. |

These five constraints, not personal preference, drove ~80% of the decisions below.

---

## 1. Orchestration framework — **LlamaIndex (primary) + LangChain (utilities)**

**Alternatives considered:** Pure-Python (no framework), Haystack, LangChain-only,
LlamaIndex-only.

**Decision:** LlamaIndex as the backbone, LangChain only for a few utilities
(e.g. the RAGAS LLM wrapper).

**Reasoning:**
- A naive instinct is "no framework — just write the glue myself." That's
  defensible for a toy, but it throws away battle-tested abstractions for node
  parsing, hierarchical (parent/child) indexing, and query transformations that
  I would otherwise reimplement and re-debug.
- LlamaIndex is **retrieval-first**. Its primitives (NodeParser, retrievers,
  postprocessors, query engines) map almost 1:1 onto the advanced techniques I
  needed. LangChain is more agent/chain-first and would have fought me on
  hierarchical retrieval.
- I deliberately did **not** go all-in on a single framework. Frameworks churn
  fast; coupling every module to LlamaIndex internals is a maintenance risk. So
  the *core logic* (chunking, RRF fusion, graph reranking, citation enforcement)
  is written as plain Python that does not import the framework. The framework
  is a convenience at the edges, not the foundation.

**Trade-off accepted:** Slightly more boilerplate than a framework-maximalist
approach, in exchange for portability and the ability to reason about every step
without framework magic.

---

## 2. Document parsing — **Nougat → GROBID → PyMuPDF (cascading fallback)**

**Alternatives considered:** PyPDF2/pdfplumber (text-only), unstructured alone,
Nougat alone, a commercial parser (Mathpix, AWS Textract).

**Decision:** A three-tier cascade, ordered by fidelity, each tier catching what
the previous one fails on.

```
Nougat   (best: LaTeX equations, tables, structure)  →  fails / not installed
  └─▶ GROBID  (good: TEI XML, section hierarchy, references)  →  service down
        └─▶ PyMuPDF  (always works: raw text extraction)
```

**Reasoning:**
- **Academic PDFs are the hardest document type in RAG.** A paper is not prose —
  it is prose *interleaved with equations, multi-column layout, figures, tables,
  and a reference list*. A plain text extractor turns `∇²φ = ρ/ε₀` into garbage
  and silently corrupts the very content a researcher is searching for.
- **Nougat** (Meta's academic OCR) is the only free tool that converts PDFs to
  Markdown *with LaTeX-preserved math*. That's the gold tier. But it is heavy
  (transformer-based, slow on CPU, occasionally fails on malformed PDFs).
- **GROBID** is a mature, fast, ML-based structure extractor. It won't preserve
  equations as cleanly as Nougat, but it nails section hierarchy and reference
  parsing — and it's the right fallback.
- **PyMuPDF** is the floor: it never fails, but it's text-only. A degraded result
  beats a crashed pipeline.

**Why a cascade instead of "just pick the best one":** Robustness. Over a corpus
of thousands of PDFs, *some will break any single parser*. A senior engineer
designs for the long tail, not the happy path. The cascade guarantees every
document produces *something* usable, while the best documents get the best
treatment. The quality tier used is logged, so I can later measure how much
parser quality affects downstream retrieval.

**Trade-off accepted:** Three dependencies instead of one, and variable parse
quality across the corpus — but zero total failures and best-effort fidelity.

---

## 3. Chunking — **Proposition-based + parent/child hierarchy + contextual headers**

This is the single most consequential design choice in the system, so it gets
the most scrutiny.

**Alternatives considered:** Fixed-size (e.g. 512-token) chunks with overlap,
recursive character splitting, semantic (embedding-distance) chunking,
proposition chunking.

**Decision:** Proposition-based child chunks for *retrieval*, larger parent
windows for *generation*, with an LLM-generated contextual header on each child.

**Reasoning — the failure mode I'm designing against:**
- Fixed-size chunking is the default everywhere, and it's the default *failure*.
  A 512-token window slices through the middle of an argument: half a claim lands
  in chunk N, the other half in chunk N+1, and *neither* retrieves well for a
  query about that claim. Retrieval precision collapses.
- **Proposition chunking** (from Chen et al., *Dense X Retrieval*, 2023)
  decomposes each paragraph into atomic, self-contained statements. Each chunk
  expresses exactly one fact. This is dramatically better for precision: you
  retrieve the *claim*, not the paragraph that happens to mention it.

**The two problems proposition chunking creates, and how I solved them:**

1. **Propositions are context-poor.** "It improves accuracy by 4%." — *what*
   does, on *what task*? An atomic proposition can lose its referent.
   → **Solution: contextual headers** (Anthropic's contextual-retrieval idea).
   Before embedding, an LLM prepends one sentence: *"This passage is from the
   Methods section of <title>, discussing <topic>."* This restores the context
   the decomposition removed, so the embedding is disambiguated.

2. **Tiny chunks give the LLM too little to reason with at generation time.**
   → **Solution: parent/child hierarchy.** I embed and search over the small
   child propositions (precision), but at generation time I swap in the larger
   *parent* section window (context). Best of both: precise retrieval, rich
   generation context.

**The cost I knowingly accepted:** Proposition extraction requires an LLM call
*per paragraph at ingest time*. That is expensive and slow. I mitigated it with
aggressive on-disk caching (keyed by a hash of the passage, so re-ingestion is
free) and batching. For a portfolio/research system where ingestion is offline
and one-time, paying compute at ingest to gain precision at every future query is
the right trade. For a system with constant high-volume ingestion, I would
reconsider and possibly fall back to semantic chunking — and I've left that as a
config toggle (`chunking.strategy`) precisely so the trade-off can be re-made
without code changes.

---

## 4. Embeddings — **BGE-M3**

**Alternatives considered:** OpenAI text-embedding-3 (paid), e5-large-v2,
nomic-embed-text, GTE, ColBERTv2 standalone, BGE-M3.

**Decision:** BGE-M3 as the single embedding model, with e5/nomic as documented
fallbacks.

**Reasoning:**
- The killer feature: **BGE-M3 produces dense, sparse, *and* ColBERT
  (multi-vector) representations from one model in one forward pass.** Most
  architectures need *two separate systems* — a dense model plus a separate BM25
  index — to do hybrid search. BGE-M3 collapses that into one model, which means
  one set of weights to load, one inference pass, and consistent tokenization
  across all three representations.
- It's **multilingual** (100+ languages) — relevant because arXiv has
  non-English papers, and it future-proofs the "multilingual papers" stretch goal.
- It's genuinely strong on **MTEB** and specifically on long-document and
  retrieval benchmarks, which is exactly our domain.
- It's free and CPU-runnable (slowly) via FlagEmbedding.

**Why not OpenAI embeddings:** Best-in-class quality, but they violate the $0
constraint and the self-hostable constraint, and they cannot produce the sparse
representation I need for exact-term matching.

**Why the sparse representation matters here specifically:** Academic queries
often contain *exact symbols and rare terms* — a method name like "ColBERT", a
metric like "nDCG@10", a token like "RLHF". Pure dense embeddings are
*semantically* smooth but *lexically* blurry: they can miss an exact-string match
that a researcher absolutely expects to find. The sparse (BM25-style) channel
catches those. This is why I refused to ship dense-only retrieval.

**Trade-off accepted:** BGE-M3 is heavier than nomic-embed-text and slow on CPU
for large batches. ColBERT multi-vectors are memory-hungry, so I default
`enable_colbert: false` and turn it on only when a GPU is present — a deliberate
graceful degradation rather than an all-or-nothing requirement.

---

## 5. Vector store — **Qdrant**

**Alternatives considered:** FAISS (library), Chroma, Weaviate, Milvus, pgvector,
Pinecone (paid), Qdrant.

**Decision:** Qdrant, self-hosted via Docker.

**Reasoning:**
- **Native hybrid search.** Qdrant supports *named vectors* — I can store a
  `dense` vector and a `sparse` vector on the *same point* and query either, then
  fuse. Few free stores do dense+sparse natively; with FAISS I'd be hand-rolling
  and synchronizing a separate sparse index, which is a correctness and
  maintenance liability.
- **Rich payload filtering with indexes.** The citation-graph reranker and the UI
  filters need fast metadata queries — "papers since 2021 with ≥50 citations."
  Qdrant lets me build payload indexes on `year`, `cited_by_count`, `concepts`,
  etc., so filtered retrieval stays fast. FAISS has no concept of payload at all.
- **Production ergonomics for free:** persistence, snapshots, a REST+gRPC API, a
  dashboard, and horizontal scaling if the corpus grows past one machine.

**Why not FAISS:** FAISS is a brilliant *library*, but it's not a *database*. No
payload, no filtering, no persistence story, no hybrid. I'd be rebuilding half of
Qdrant to use it. For a flat, static, dense-only index FAISS would win on raw
speed — but that's not our problem shape.

**Why not pgvector:** Attractive if Postgres were already in the stack, but its
hybrid and filtering story is weaker, and I didn't want to run a full RDBMS just
for vectors.

**Trade-off accepted:** Running a Docker service (operational weight) versus an
in-process library. Worth it for the hybrid + payload capabilities that the
academic use case demands.

---

## 6. Citation graph — **DuckDB (default) / Neo4j (optional)**

**Alternatives considered:** Neo4j Community, NetworkX (in-memory), a Postgres
table, DuckDB.

**Decision:** DuckDB as the default edge store, with Neo4j as a documented
swap-in for graph-heavy stretch goals (community detection, GraphRAG).

**Reasoning:**
- The graph reranker needs *one* simple query: given a paper id, return its
  influence and recency priors. That is a **lookup and aggregation** workload, not
  a multi-hop traversal workload. For that, a columnar analytical DB is perfect.
- **DuckDB is embedded** — zero server, a single file, blistering analytical
  speed, runs anywhere Python runs. For the actual queries we issue, it is both
  simpler and faster than standing up Neo4j.
- I kept the store behind an interface (`graph_store.py`) so that *if* a stretch
  goal needs real graph algorithms — citation-community detection for
  "survey by sub-topic", PageRank-style authority — I can swap in Neo4j without
  touching the reranker. **Don't pay for graph-database complexity until you have
  a graph-algorithm problem.** Right now I don't, so I don't.

**Trade-off accepted:** DuckDB can't do efficient deep multi-hop traversals. The
day a feature needs them, I switch. Until then, simplicity wins.

---

## 7. Generation LLM — **Ollama running Llama 3.1 8B (Qwen2.5 / Mistral as swaps)**

**Alternatives considered:** GPT-4/Claude via API (paid), a 70B local model,
Llama 3.1 8B / Qwen2.5 7B / Mistral 7B via Ollama.

**Decision:** A 7-8B-class instruct model served locally through Ollama.

**Reasoning:**
- **Ollama** gives me an OpenAI-style HTTP API over local models with one-line
  model management (`ollama pull`). It handles quantization, GGUF, and memory
  management so I don't have to. The whole pipeline talks to it over plain HTTP,
  which means swapping the model is a config change, not a code change.
- **Why 8B and not 70B:** the single-machine constraint. An 8B model quantized
  fits comfortably in RAM and generates at usable speed on CPU/consumer GPU. A
  70B model would blow the memory budget and make the per-query latency
  unacceptable — and crucially, **the generation step here is heavily
  constrained by retrieved context and citation enforcement**, so raw LLM
  reasoning horsepower matters *less* than it would in open-ended generation. The
  retrieval stack is doing the heavy lifting; the LLM is mostly synthesizing
  grounded passages.
- The LLM is used in *several* roles — proposition extraction, contextual
  headers, HyDE, multi-query expansion, CRAG scoring, generation, and as the
  RAGAS judge. Keeping them all on one local model keeps the system coherent and
  free.

**Trade-off accepted:** An 8B model hallucinates and reasons less reliably than a
frontier model. I countered this *architecturally* rather than by spending money:
Self-RAG reflection, citation enforcement, and abstention (sections below) exist
precisely to make a smaller model *safe to ship* by refusing to let it make
unsupported claims. **Design around the model's weaknesses instead of paying to
remove them.**

---

## 8. Retrieval strategy — the layered stack

Each layer addresses a *specific, named* failure mode. I did not add techniques
because they're fashionable; I added each one to fix a concrete defect, and each
is independently toggleable so its contribution can be measured (see the ablation
harness).

| Layer | Failure mode it fixes | Why this technique |
|---|---|---|
| **HyDE** | A short, casual query embeds poorly against dense academic prose. | Generate a hypothetical *abstract* for the query and embed that — it shares vocabulary and structure with real papers, closing the lexical/register gap. |
| **Multi-query expansion** | One phrasing retrieves one neighborhood; synonyms and sub-aspects are missed. | Generate paraphrases/sub-questions and union their results — recall goes up, especially for multi-part questions. |
| **Hybrid (dense + sparse) + RRF** | Dense misses exact terms; sparse misses paraphrase. | Run both, fuse with **Reciprocal Rank Fusion**. RRF (k=60) is rank-based, so it's robust to the two channels having totally different score scales — no fragile score normalization. |
| **Cross-encoder rerank** | Top-k by vector similarity is noisy; the bi-encoder scored query and passage *independently*. | A cross-encoder reads (query, passage) *jointly* and is far more accurate. Run it only on the ~50 fused candidates — too slow for the whole corpus, perfect for a shortlist. |
| **Citation-graph rerank** | The most *relevant* passage isn't always from the most *authoritative* paper. | Blend `α·rerank + β·log(1+influential_citations) + γ·recency`. Surfaces work that is both on-topic and trustworthy/current. Weights are tunable and ablatable. |
| **CRAG** | Sometimes retrieval just *fails* and we'd generate from garbage. | A lightweight evaluator scores context quality; below threshold it reformulates the query and retries before generating. Fail *loudly and correctably*, not silently. |

**Why RRF specifically (and not weighted score averaging):** Dense cosine scores
and sparse BM25 scores live on incomparable scales. Averaging them requires
normalization that is brittle and corpus-dependent. RRF throws away the magnitudes
and uses only *ranks*, which is exactly the robustness property I want when fusing
heterogeneous retrievers. This is a small decision that prevents a whole class of
silent bugs.

**Why rerank a shortlist, not the corpus:** Cross-encoders are O(candidates) and
~100× slower per item than a bi-encoder lookup. Running one over the full corpus
per query is infeasible. The standard, correct pattern is *cheap recall-oriented
retrieval → expensive precision-oriented reranking on a shortlist*. The whole
pipeline is shaped like a funnel for this reason.

---

## 9. Generation safety — **Self-RAG + citation enforcement + abstention**

**The problem:** A local 8B model *will* occasionally state things the retrieved
passages don't support. In an academic tool, an unsupported claim with a
fabricated-looking citation is the worst possible output — it's confidently wrong
in a domain where trust is everything.

**The three-layer defense:**
1. **Self-RAG reflection** — after drafting, the model re-reads its own answer
   against the context and marks unsupported claims.
2. **Citation enforcement** — a deterministic, regex-based validator drops any
   sentence that doesn't carry a citation id resolving to a retrieved passage.
   This is *code*, not the LLM policing itself, so it can't be talked out of it.
3. **Abstention** — if the CRAG context-quality score is below threshold, the
   system refuses: *"INSUFFICIENT EVIDENCE."* A RAG system that knows when to say
   "I don't know" is worth more than one that always answers.

**Design principle:** *Make grounding a property of the system, not a hope about
the model.* The deterministic validator is the load-bearing piece — I never trust
the model to enforce its own honesty.

---

## 10. Evaluation — **RAGAS + BEIR retrieval metrics + MLflow + ablation harness**

**Alternatives considered:** Eyeballing a few queries (the common anti-pattern),
RAGAS only, custom metrics only.

**Decision:** Two complementary metric families, every run logged, every technique
ablatable.

**Reasoning:**
- **You cannot improve what you don't measure, and "it works on my 3 test
  questions" is not measurement.** This is the line between a demo and an
  engineering artifact.
- **RAGAS** covers the *generation* side: faithfulness, answer relevancy, context
  precision/recall. Using the *local* LLM as judge keeps eval free and
  reproducible.
- **Classic IR metrics** (nDCG@k, Recall@k, MRR) on **BEIR-SciFact** cover the
  *retrieval* side independently — because a great generator can mask a mediocre
  retriever and vice versa. I want to see each half on its own.
- **MLflow** logs every configuration (chunker, embedder, top-k, α/β/γ) against
  its metrics, so improvements are attributable and regressions are caught.
- **The ablation harness** is the centerpiece of the "seasoned engineer" claim:
  it toggles each technique off in turn and quantifies its individual
  contribution. This is how I *prove* HyDE or reranking is earning its
  complexity, rather than asserting it. If a technique doesn't move the metrics,
  it gets cut. Complexity must pay rent.

**Trade-off accepted:** A local LLM judge is noisier than GPT-4-as-judge. I accept
the noise to stay free and reproducible, and I rely on the deterministic IR
metrics as the un-foolable ground truth alongside it.

---

## 11. Serving — **FastAPI + Streamlit**

**Decision:** FastAPI for the service layer, Streamlit for the UI.

**Reasoning:**
- **FastAPI**: async, automatic OpenAPI docs, Pydantic validation at the boundary.
  The `/ask`, `/ingest`, `/health`, `/papers` contract is typed and
  self-documenting. Pydantic request models (with `min_length`, ranges) reject
  malformed input *before* it touches the pipeline.
- **Streamlit**: a researcher-facing UI in pure Python with zero frontend build.
  For a research/portfolio tool whose users are scientists, not consumers, this
  is exactly the right level of investment. Citations render as expandable cards
  with arXiv links and a visible context-quality score — *traceability made
  visible*, which is the whole point of the product.

**Trade-off accepted:** Streamlit is not a production consumer frontend (no fine
UI control, re-runs top-to-bottom). That's a non-goal — this is a research tool,
and Streamlit's speed-to-build is worth far more here than pixel control.

---

## 12. Containerization — **Docker Compose**

**Decision:** Compose for the stateful services (Qdrant, GROBID, MLflow); app and
Ollama run natively for performance during development.

**Reasoning:** One command reproduces the entire backing infrastructure on any
machine — the reproducibility constraint, satisfied. I keep the heavy model
services (Ollama) and the hot-reloading app *outside* Compose during dev because
native execution is faster to iterate on and Ollama is more performant with
direct hardware access. The Dockerfile exists for the day this needs to ship as a
single deployable unit.

---

## 13. Meta-principles (what a reviewer should take away)

1. **Constraints first, tools second.** Every choice traces back to the five
   constraints in §0, not to novelty.
2. **Each technique fixes a named failure mode** — and is independently toggleable
   so its value can be measured. No cargo-culting.
3. **Design around the model's limits instead of buying a bigger model.** The
   safety stack exists *because* the LLM is small and free.
4. **Robustness over the happy path.** Cascading parsers, graceful degradation
   (ColBERT off on CPU), abstention, fallbacks everywhere. Real corpora are messy.
5. **Simplicity until complexity is justified by data.** DuckDB over Neo4j,
   shortlist reranking over full-corpus, framework at the edges only. Complexity
   must pay rent in measured metrics.
6. **Traceability is a feature, not an afterthought.** Citations are enforced by
   code and surfaced in the UI, because in an academic tool, an answer you can't
   verify is worthless.
