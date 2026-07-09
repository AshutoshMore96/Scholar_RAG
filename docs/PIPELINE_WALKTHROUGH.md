# ScholarRAG — End-to-End Pipeline Walkthrough

> This document follows a single piece of data from a raw arXiv PDF all the way
> to a cited sentence in a literature review, and then follows a user's question
> back through the system to that same sentence. For every step it answers three
> questions: **what** is happening, **which tool** is doing it, and **how** it
> works.
>
> Read this alongside the diagrams in `diagrams/` (HLD, LLD, techniques map).

---

## The two halves of the system

ScholarRAG has two distinct execution paths. Confusing them is the most common
source of misunderstanding, so they are separated here.

```
OFFLINE  (runs occasionally, builds the knowledge base)
   arXiv → ingest → transform → embed → store
              ↓
        [ Qdrant + DuckDB now contain the corpus ]
              ↓
ONLINE   (runs on every user question, in milliseconds-to-seconds)
   question → expand → retrieve → rerank → check → generate → answer
```

The **offline** path is slow, batch, and idempotent — you run it to ingest new
papers. The **online** path is latency-sensitive and runs per query. They meet at
the storage layer: offline *writes* it, online *reads* it.

---

# PART A — The Offline Indexing Path

*Goal: turn messy PDFs into a searchable, citation-aware knowledge base.*

---

## Step 1 — Ingestion: get the papers and their reputations

**What:** Fetch paper metadata and PDFs from arXiv, then enrich each paper with
citation-graph signals (how influential is it, how recent, what does it cite).

**Tools:** `httpx` (HTTP), the **arXiv API**, **OpenAlex API**, **Semantic
Scholar API**, `tenacity` (retries).

**Code:** `src/scholar_rag/ingest/`

### 1a. Crawl arXiv — `arxiv_crawler.py`

- **How:** `ArxivCrawler.fetch()` builds an arXiv API query like
  `(cat:cs.CL OR cat:cs.IR) AND submittedDate:[20200101 TO *]`, calls the API,
  and parses the returned **Atom XML** feed with `ElementTree`.
- Each entry becomes an `ArxivPaper` dataclass (id, title, abstract, authors,
  categories, dates, pdf_url, doi).
- It **paginates automatically** (100 results per request) and sleeps ≥3 s
  between calls — arXiv's published rate-limit etiquette. Violating it gets you
  IP-banned, so this is non-optional.
- `download_pdf()` saves the PDF and a JSON sidecar to `data/raw/`. It is
  **resumable**: if `<id>.pdf` already exists, it's skipped. Re-running the
  crawler never re-downloads.

### 1b. Enrich citations — `citation_enrich.py`

- **What:** arXiv tells you *what* a paper says but not *how important* it is.
  That signal comes from the scholarly graph.
- **How:** For each arXiv id, `CitationEnricher.enrich()` queries:
  - **OpenAlex** → `cited_by_count`, publication year, venue, concept tags,
    and the list of works this paper references.
  - **Semantic Scholar** → `influentialCitationCount` (citations that actually
    build on the work, not just name-drop it), and the TLDR summary.
- Results are merged into a `CitationMeta` object and cached as a
  `<id>_citation.json` sidecar, so enrichment also never repeats.
- **Why two sources:** they have complementary coverage and complementary fields.
  S2's *influential* citation count is the signal the graph-reranker leans on; it
  has no OpenAlex equivalent. OpenAlex has the cleaner reference graph. Using both
  and merging fills each other's gaps.

### 1c. Orchestrate — `pipeline.py`

- `run_ingestion()` ties 1a and 1b together with `tqdm` progress bars and returns
  a list of `{paper, citation_meta}` records ready for transformation.

**Output of Step 1:** `data/raw/` full of PDFs + metadata JSON + citation JSON.

---

## Step 2 — Transformation: turn PDFs into clean, atomic, contextualized chunks

This is the most involved stage. It has four sub-steps.

**Code:** `src/scholar_rag/transform/`

### 2a. Parse PDF → Markdown — `parse_nougat.py` → `parse_grobid.py` → PyMuPDF

- **What:** Convert the binary PDF into structured Markdown text, preserving
  equations, tables, and section headings.
- **How (cascade):**
  1. **Nougat** (`NougatParser`) runs the `nougat` CLI; it's a vision-transformer
     OCR that outputs Markdown with **LaTeX-preserved math**. Best quality.
  2. If Nougat is unavailable or fails → **GROBID** (`GrobidParser`) POSTs the PDF
     to the GROBID Docker service, gets back **TEI XML**, and converts it to
     Markdown (title, abstract, sections, paragraphs).
  3. If GROBID is down too → **PyMuPDF** does raw text extraction. Always works.
- **Why:** academic PDFs carry meaning in their math and structure; a naive text
  dump destroys exactly what researchers search for. The cascade guarantees a
  result for every document while giving the best ones the best treatment.

### 2b. Decompose into propositions — `propositions.py`

- **What:** Break each paragraph into atomic, self-contained factual statements.
- **Tool:** the local **Ollama LLM** (Llama 3.1 8B).
- **How:** `PropositionExtractor.extract()` sends the paragraph with a strict
  prompt — *"output a JSON array of atomic propositions, resolve pronouns, one
  fact each"* — parses the JSON, and **caches** the result keyed by a SHA-256 of
  the passage. Re-ingestion of the same text is free.
- **Why:** retrieval precision. A query about a single claim retrieves the
  *proposition* expressing it, not a 500-word paragraph that merely contains it.

### 2c. Add contextual headers — `contextual_headers.py`

- **What:** Prepend one context sentence to each proposition before embedding.
- **Tool:** the local **Ollama LLM**.
- **How:** `ContextualHeaderGenerator.generate()` asks the LLM for a sentence like
  *"This passage is from the Methods section of <title>, discussing <topic>."* and
  prepends it to the proposition. Also cached.
- **Why:** propositions lose their referent when isolated. The header restores it,
  so the embedding is unambiguous. (See DESIGN_DECISIONS §3.)

### 2d. Build the parent/child hierarchy — `chunkers.py`

- **What:** Produce two linked representations per document.
- **How:** `DocumentChunker.chunk_document()`:
  1. Splits the Markdown into **sections** by heading.
  2. Slides a large window over each section → **parent chunks** (~1024 tokens,
     the rich context for generation).
  3. For each paragraph in a parent, calls 2b + 2c to produce **child chunks**
     (atomic propositions with headers, the precise retrieval units).
  4. Each child stores a `parent_id` pointer back to its parent window.
- **Output:** `(child_chunks, parent_chunks)` with every child linked to a parent
  and every chunk carrying full metadata (paper_id, section, title, year, venue).

**Output of Step 2:** lists of small, contextualized, atomic child chunks (for
search) plus larger parent windows (for generation), fully linked.

---

## Step 3 — Embedding: turn text into vectors

**What:** Convert each child chunk into the numeric representations that make
search possible.

**Tool:** **BGE-M3** via **FlagEmbedding**.

**Code:** `src/scholar_rag/embed/`

- **How:** `ChunkEmbedder.embed_chunks()` batches chunk texts through
  `BGEM3Embedder.encode()`, which in **one forward pass** produces:
  - **dense** — a 1024-d float vector capturing *meaning* (semantic similarity).
  - **sparse** — a `{token_id: weight}` map capturing *exact terms* (BM25-style
    lexical signal).
  - **colbert** (optional) — per-token vectors for late-interaction matching;
    off by default on CPU for memory reasons.
- Each chunk becomes an `EmbeddedChunk(chunk, dense, sparse, colbert)`.
- **Why one model for all three:** consistency and efficiency — one set of
  weights, one tokenizer, one inference pass produces every representation hybrid
  search needs. (See DESIGN_DECISIONS §4.)

**Output of Step 3:** every child chunk paired with its dense + sparse vectors.

---

## Step 4 — Storage: write the searchable index and the citation graph

**What:** Persist vectors + metadata for fast hybrid search, and persist the
citation edges for authority-aware reranking.

**Tools:** **Qdrant** (vectors), **DuckDB** (graph).

**Code:** `src/scholar_rag/store/`

### 4a. Vector store — `qdrant_store.py`

- **How:** `create_collection()` defines a collection with **named vectors** —
  a `dense` vector and a `sparse` vector *on the same point* — plus **payload
  indexes** on `year`, `cited_by_count`, `influential_citation_count`, `section`,
  `concepts` for fast filtering.
- `upsert_chunks()` writes each child chunk as a Qdrant point: both vectors plus a
  payload containing the chunk text, its **parent_text** (so generation can pull
  full context without a second store), title, section, and all citation metadata.
- **Why Qdrant:** native dense+sparse hybrid and indexed payload filtering — the
  two capabilities this use case is built on. (See DESIGN_DECISIONS §5.)

### 4b. Citation graph — `graph_store.py`

- **How:** `CitationGraphStore` keeps two DuckDB tables: `papers` (id, year,
  citation counts, concepts) and `cites` (directed src→dst edges).
  `get_influence_prior()` returns, for any paper, `log(1+influential_citations)`
  and a `recency = 1/(1+years_old)` score.
- **Why:** the graph reranker (Step 8) reads these priors to boost authoritative
  and recent work. DuckDB is embedded and analytical — ideal for this lookup.

**Output of Step 4 (and of the entire offline path):** a populated Qdrant
collection and a populated DuckDB graph. The knowledge base now exists.

---

# PART B — The Online Query Path

*Goal: answer a research question with a traceable, cited literature review.*

A user submits a question (via Streamlit UI or `POST /ask`). The FastAPI handler
in `api/main.py` hands it to `RetrievalEngine.retrieve()` in
`retrieve/engine.py`, which runs the following funnel. **Every step narrows or
sharpens the candidate set.**

---

## Step 5 — Query expansion: ask the question in more, better ways

**What:** A single user phrasing is a weak retrieval signal. Generate richer and
multiple formulations.

**Tools:** the local **Ollama LLM**.

**Code:** `retrieve/hyde.py`, `retrieve/multi_query.py`

### 5a. HyDE — `hyde.py`

- **How:** `HyDEQueryExpander.expand()` asks the LLM to *write a hypothetical
  100-150 word abstract that would answer the question*, then we embed **that**
  instead of the bare query.
- **Why:** a hypothetical abstract uses the same dense academic vocabulary as real
  papers, so its embedding lands near real answers — closing the gap between a
  casual question and formal prose.

### 5b. Multi-query expansion — `multi_query.py`

- **How:** `MultiQueryExpander.expand()` asks the LLM for N paraphrases of the
  question (and can `decompose()` a multi-part question into sub-questions).
- **Why:** different phrasings retrieve different relevant neighborhoods; unioning
  them raises recall.

**Output of Step 5:** a set of query texts = `[HyDE doc] + [paraphrases]`.

---

## Step 6 — Embed the queries

- **How:** each query variant is embedded with the **same BGE-M3 model**
  (`ChunkEmbedder.embed_query()`), producing dense + sparse vectors per variant —
  symmetric with how documents were embedded in Step 3.
- **Why symmetry matters:** query and document must live in the same vector space,
  produced by the same model, or similarity is meaningless.

---

## Step 7 — Hybrid retrieval + RRF: pull candidates from the index

**What:** For every query variant, search Qdrant with both vectors and fuse
everything into one ranked candidate list.

**Tool:** **Qdrant** + **Reciprocal Rank Fusion**.

**Code:** `retrieve/hybrid_rrf.py`

- **How:**
  1. For each query embedding, `HybridRetriever.retrieve()` runs a **dense**
     search and a **sparse** search against Qdrant (with any metadata filter —
     year range, min citations).
  2. The two ranked lists are merged with **RRF**:
     `score(d) = Σ 1/(60 + rank_in_list)`.
  3. `retrieve_multi_query()` then fuses across *all* query variants the same way.
- **Why RRF:** dense cosine and sparse BM25 scores are on incomparable scales.
  RRF uses only **ranks**, sidestepping fragile score normalization, and rewards
  documents that multiple retrievers/queries agree on.

**Output of Step 7:** ~50 candidate passages, recall-oriented.

---

## Step 8 — Reranking: sharpen relevance, then weigh authority

Two reranking passes, in order.

### 8a. Cross-encoder rerank — `rerank.py`

- **Tool:** **bge-reranker-v2-m3** via FlagEmbedding.
- **How:** `CrossEncoderReranker.rerank()` scores each `(query, passage)` pair
  **jointly** (the model reads both together, unlike the independent bi-encoder
  embeddings) and re-sorts the ~50 candidates by this sharper relevance score.
- **Why a shortlist:** cross-encoders are slow; running one over the whole corpus
  is infeasible, but over 50 candidates it's fast and dramatically more accurate.

### 8b. Citation-graph rerank — `graph_rerank.py`

- **Tool:** the **DuckDB** citation graph.
- **How:** `CitationGraphReranker.rerank()` computes a blended final score:
  `α·rerank_score + β·log(1+influential_citations) + γ·recency`
  (defaults α=0.6, β=0.25, γ=0.15), reading influence/recency from payload or the
  graph store.
- **Why:** the most *textually* relevant passage isn't always from the most
  *trustworthy* paper. This nudges authoritative and recent work upward, which is
  what a researcher actually wants in a literature review.

**Output of Step 8:** the top ~10 passages, relevance- *and* authority-ranked.

---

## Step 9 — CRAG: quality gate before generation

**What:** Decide whether the retrieved context is good enough to answer from.

**Tool:** the local **Ollama LLM** as an evaluator.

**Code:** `retrieve/crag.py`

- **How:** `CRAGEvaluator.needs_reformulation()` asks the LLM to score context
  relevance 0-1. If it's below threshold (default 0.4), `reformulate()` rewrites
  the query to be more specific and the engine runs **one more retrieval pass**
  before continuing.
- **Why:** retrieval sometimes just fails. Detecting that and *correcting* it
  beats silently generating from irrelevant context.

### Parent expansion

Right before handing off, `engine._expand_to_parent()` swaps each child
proposition's text for its larger **parent window** (pulled from the Qdrant
payload). The system *retrieved* with precise atoms but *generates* with rich
context — the parent/child payoff.

**Output of Step 9:** a final, quality-checked set of context passages (each now
the full parent window) plus a `context_quality` score.

---

## Step 10 — Generation: write the cited review, safely

**What:** Synthesize the passages into a literature review where every claim is
traceable — and refuse if the evidence is too thin.

**Tool:** the local **Ollama LLM**, wrapped in deterministic validators.

**Code:** `generate/cited_generator.py`, `generate/prompts.py`

- **How** (`CitedLiteratureGenerator.generate()`):
  1. **Abstain check** — if `context_quality < abstain_threshold`, immediately
     return *"INSUFFICIENT EVIDENCE"* and stop. The system knows when to say no.
  2. **Draft** — format passages as a numbered, id-tagged context block and prompt
     the LLM to write a 200-400 word review where *every claim cites `[paper_id]`*.
  3. **Self-RAG reflection** — the model re-reads its draft against the context
     and drops claims it can't support.
  4. **Citation enforcement** — a deterministic regex validator (`_enforce_citations`)
     removes any sentence lacking a citation id that resolves to a real retrieved
     passage. This is *code*, not the model policing itself.
  5. Return a `GenerationResult`: the review, the list of citation ids actually
     used, the context-quality score, and the abstain flag.
- **Why this stack:** a small local LLM will occasionally over-claim. Reflection +
  enforcement + abstention make grounding a *property of the system* rather than a
  hope about the model. (See DESIGN_DECISIONS §9.)

### Optional — ReAct agent — `generate/agent.py`

For multi-hop questions ("what does paper X cite that's also relevant to Y?"), a
ReAct agent can iteratively call `search_papers`, `expand_citation`, and
`fetch_abstract` tools before answering.

**Output of Step 10:** the cited literature review.

---

## Step 11 — Serve the answer back

**Tools:** **FastAPI** (API), **Streamlit** (UI).

**Code:** `api/main.py`, `ui/app.py`

- **How:** `api/main.py`'s `/ask` handler assembles an `AskResponse` — the review,
  the citation cards (paper_id, title, year, venue, score), the context-quality
  score, the abstain flag, and the measured latency.
- The **Streamlit UI** renders the review, then shows each source as an expandable
  card with its score and a direct **arXiv link**, plus a color-coded
  context-quality indicator. *Traceability made visible.*

**Output of Step 11:** what the user sees — a cited mini-literature-review they can
verify passage by passage.

---

# PART C — The Evaluation Loop (the feedback that keeps it honest)

*Goal: prove the pipeline works and that each technique earns its place.*

**Tools:** **RAGAS**, custom IR metrics, **MLflow**.

**Code:** `eval/`

| Step | What | Tool | How |
|---|---|---|---|
| **RAGAS** | Score generation quality | `eval/ragas_eval.py` + local LLM judge | Runs the pipeline over a golden Q&A set, computes faithfulness / answer-relevancy / context-precision / context-recall, logs to MLflow. |
| **IR metrics** | Score retrieval in isolation | `eval/retrieval_metrics.py` | nDCG@k, Recall@k, MRR against BEIR-SciFact — measures the retriever independently of the generator. |
| **Ablations** | Quantify each technique's contribution | `eval/ablation.py` | Toggles HyDE / multi-query / hybrid / rerank / graph-rerank off one at a time, reruns metrics, logs each config to MLflow for side-by-side comparison. |

**Why:** this is the difference between a demo and an engineering artifact. The
ablation harness *proves* each layer of the funnel earns its complexity — if a
technique doesn't move the numbers, it gets cut. Everything is logged to MLflow so
improvements are attributable and regressions are caught. (See DESIGN_DECISIONS
§10.)

---

## One-page summary: data's journey

```
A PDF on arXiv
  │  arxiv_crawler (httpx + arXiv API)         ── download
  │  citation_enrich (OpenAlex + S2)           ── attach influence/recency
  ▼
Raw PDF + metadata + citation JSON  (data/raw/)
  │  Nougat → GROBID → PyMuPDF                 ── parse to Markdown (+math)
  │  propositions (Ollama LLM)                 ── split into atomic facts
  │  contextual_headers (Ollama LLM)           ── disambiguate each fact
  │  chunkers                                  ── parent/child hierarchy
  ▼
Atomic, contextualized child chunks + parent windows
  │  BGE-M3 (FlagEmbedding)                    ── dense + sparse vectors
  ▼
Embedded chunks
  │  qdrant_store                              ── upsert vectors + payload
  │  graph_store (DuckDB)                      ── store citation edges
  ▼
═══════════ KNOWLEDGE BASE READY ═══════════
  ▲
  │  A user question arrives
  │  HyDE + multi_query (Ollama LLM)           ── expand the question
  │  BGE-M3                                    ── embed query variants
  │  hybrid_rrf (Qdrant dense+sparse + RRF)    ── pull ~50 candidates
  │  rerank (bge-reranker-v2-m3)               ── sharpen to relevance
  │  graph_rerank (DuckDB priors)              ── weigh authority/recency → top 10
  │  crag (Ollama LLM)                         ── quality gate / retry
  │  cited_generator (Ollama LLM + validators) ── grounded, cited review / abstain
  ▼
FastAPI → Streamlit                            ── cited review + source cards
  │
  └─ RAGAS + IR metrics + ablations → MLflow   ── prove it works
```
