# ScholarRAG — Architecture & Engineering Notes

A retrieval-augmented generation system that writes **grounded, citation-backed
literature reviews** over a corpus of **1,785 arXiv papers (147,580 chunks)**.
Ask a research question; get a synthesized review with inline `[arxiv_id]`
citations and a ranked source list — or an explicit *abstention* when the corpus
doesn't support an answer.

This document is the part most RAG demos skip: **why** the pipeline is built the
way it is, and a walk through two real debugging investigations — a **10×
latency regression** and a **quality metric that lied** — with the measurements
that drove each fix.

---

## 1. What it actually does

**Retrieval quality is the whole game.** An LLM can only be as good as the
passages you put in front of it, so most of the engineering lives *before*
generation. The pipeline is deliberately heavier than a "embed → cosine search →
stuff into a prompt" tutorial:

- **Hybrid retrieval** — dense (semantic) *and* sparse (lexical) vectors from a
  single BGE-M3 pass, fused with Reciprocal Rank Fusion.
- **Query expansion** (deep mode) — HyDE hypothetical documents + multi-query
  paraphrasing to bridge the vocabulary gap between casual questions and
  academic prose.
- **Two-stage reranking** — a cross-encoder for semantic precision, then a
  **citation-graph reranker** that blends relevance with scholarly influence and
  recency.
- **Corrective retrieval (CRAG)** — an LLM judge scores context relevance and
  can trigger query reformulation + a second retrieval pass.
- **Grounded generation** — cited-review synthesis with self-reflection,
  citation enforcement, and a **calibrated abstention gate** so the system says
  "insufficient evidence" instead of hallucinating.

---

## 2. System architecture

Two pipelines: an **offline ingestion** path that builds the index, and an
**online query** path that serves reviews.

### Ingestion (offline)

```
arXiv API ──▶ GROBID (PDF ─▶ structured sections)
                 │
                 ▼
        Chunking  (parent/child hierarchy, contextual headers,
                   proposition- or semantic-based splitting)
                 │
                 ▼
        BGE-M3  ──▶  dense (1024-d)  +  sparse (lexical weights)
                 │
                 ├──▶ Qdrant Cloud   (named vectors, payload indexes,
                 │                     INT8 scalar quantization)
                 │
   OpenAlex ─────┴──▶ citation enrichment (cited_by_count, year)
                          └──▶ DuckDB citation graph
```

### Query (online)

```
             question
                │
     ┌──────────┴───────────┐
     │  deep mode only:      │
     │  HyDE + multi-query×3 │
     └──────────┬───────────┘
                ▼
     BGE-M3 embed variants (single batched pass)
                ▼
     per-variant hybrid search (dense + sparse)  ──▶  Qdrant Cloud
                ▼
     Reciprocal Rank Fusion  ──▶  ~20 candidates
                ▼
     cross-encoder rerank   (Jina API, or local bge-reranker)
                ▼
     citation-graph rerank   α·rerank + β·log(1+cited) + γ·recency
                ▼
     ┌───────── deep mode only: CRAG judge ─────────┐
     │  score < threshold → reformulate + retry once │
     └───────────────────┬───────────────────────────┘
                ▼
     quality gate  →  abstain?  ── yes ──▶  "insufficient evidence"
                │
                no
                ▼
     cited generation (Groq / Ollama, token-streamed over SSE)  ──▶  UI
```

### Where each component runs

The system is **backend-agnostic by design** — every model-serving piece can run
locally *or* on a hosted API, selected by environment config, through one
abstraction ([`generate/llm.py`](src/scholar_rag/generate/llm.py) speaks both the
Ollama and OpenAI wire formats):

| Component | Local mode | Hosted mode | Why |
|---|---|---|---|
| Query embedding (BGE-M3) | in-process | in-process | dense+sparse must match the corpus vectors — kept local |
| Reranking | `bge-reranker-v2-m3` | **Jina API** | 2.3 GB local model; offloaded on RAM-constrained hosts |
| Generation | Ollama (Llama 3.x) | **Groq** (Llama-3.1-8B / 3.3-70B) | nobody self-hosts a 70B for a demo; config-swappable |
| Vector store | Docker Qdrant | **Qdrant Cloud** | managed ANN + persistence |

This duality is a feature, not a compromise — the same code runs fully
self-contained on a workstation with a GPU, or scales out to free hosted
inference on an 8 GB laptop. **The two investigations below are the story of
discovering *why* that flexibility mattered.**

---

## 3. Investigation #1 — a 10× latency regression that wasn't in the code

**Symptom.** "Deep" retrieval took **30–100 s** per query. The obvious suspect
was the newly-added hosted LLM (network! cold starts!). The obvious suspect was
wrong.

### Step 1 — measure every stage, don't guess

I instrumented the deep pipeline stage by stage instead of theorizing:

| Stage | Time | Backend |
|---|---:|---|
| HyDE expand | 0.7 s | Groq |
| Multi-query expand | 0.2 s | Groq |
| **Embed 5 variants** | **83.6 s** | local CPU |
| Hybrid retrieval | 0.5 s | Qdrant |
| **Cross-encoder rerank** | **23.4 s** | local CPU |
| CRAG judge | 0.2 s | Groq |

The hosted APIs totalled **~1.1 s**. The entire cost was two *local* stages:
embedding and reranking, **107 s combined**.

### Step 2 — the numbers don't reconcile

Benchmarked in isolation, the same operations were fast:

- Embedding 5 short texts: **1.3 s** (not 83 s)
- Reranking 20 candidates: **6.9 s** (not 23 s)

A **~40–60× slowdown appeared only inside the full process.** Code that's fast
alone and slow together points at a *shared resource*, not an algorithm.

### Step 3 — follow the memory

The machine is an **8 GB Mac**. `sysctl vm.swapusage` told the whole story:

```
vm.swapusage: total = 12288M  used = 10998M   ← 11 GB in swap
```

The resident set:

| Model | RAM |
|---|---:|
| BGE-M3 embedder | ~2.3 GB |
| bge-reranker-v2-m3 | ~2.3 GB |
| Ollama `llama3.2:3b` | ~2.5 GB |
| Docker Desktop VM (qdrant + grobid + mlflow) | ~4.0 GB |

~11 GB of models fighting over 8 GB of RAM → **constant swap thrashing**. Every
tensor op was faulting pages to disk. It wasn't the LLM, the network, or the
device — it was **memory pressure slowing every stage at once.**

A confirming detail: switching to Apple's MPS GPU made it *worse* (the process
**hung**). On Apple Silicon, MPS shares the same 8 GB unified memory — no relief,
more contention.

### Step 4 — attack the memory, and measure the fix

I proved the hypothesis before committing to it. Swapping the 2.3 GB reranker for
an 80 MB MiniLM cross-encoder — same query, same process:

| | 2.3 GB reranker resident | 80 MB reranker |
|---|---:|---:|
| Embed 5 variants | 4–19 s | **0.30 s** |
| Rerank 20 | 14–32 s | **0.13 s** |
| Peak RSS | ~4.6 GB → swap | **0.93 GB** |

Freeing the reranker's memory didn't just speed up reranking — **it sped up
embedding too**, because the whole box stopped swapping. That confirmed the
diagnosis: *memory, not compute*.

### The fixes (all measured, all reversible via config)

1. **Reranking → Jina API.** Frees 2.3 GB; ~150 ms/query; quality ≥ the local
   model. ([`retrieve/rerank.py`](src/scholar_rag/retrieve/rerank.py) auto-selects
   API vs local, with graceful fallback to RRF order if the API is down.)
2. **Generation → Groq for both modes.** Drops the resident 2.5 GB Ollama model
   entirely — this alone took embedding from **44 s → 2.4 s** by ending the last
   of the swap.
3. **Batched query embedding.** All variants in one BGE-M3 pass instead of a
   Python loop ([`embed/embedder.py::embed_queries`](src/scholar_rag/embed/embedder.py)).
4. **Stopped unused Docker containers** — `scholar_qdrant` was pure dead weight
   (the app is on Qdrant *Cloud*); GROBID/MLflow are ingestion-only. ~4 GB back.

### Result

| Stage of the fix | Deep latency |
|---|---:|
| Original (all resident, swapping) | **~47 s** |
| + Ollama model unloaded | 23 s |
| + unused Docker containers stopped | 16 s |
| + generation moved to Groq (Ollama gone) | **~4.5 s** |

**Deep: 47 s → 4.5 s. Fast: ~1.5 s. Embedding: 44 s → 2.4 s.** Not one line of
retrieval logic changed — the entire win was understanding the resource budget of
an 8 GB machine and moving the two memory-heavy models off it.

> **Lesson:** when isolated benchmarks and in-process timings disagree by an order
> of magnitude, stop optimizing the algorithm and go look at the shared resource.
> The bug was never in the code.

---

## 4. Investigation #2 — a quality metric that lied

**Symptom.** Deep mode reported **lower** context-quality (0.80) than fast mode
(0.98) on the *same* query — even though deep retrieves *more* relevant papers.
That's backwards. Either deep retrieval was broken, or the metric was.

### Root cause

Quality was computed as `min(CRAG_judge, top_reranker_score)`:

```python
# before
return round(min(crag_quality, top_rerank), 3)   # 0.8 vs 0.98 → 0.8
```

The intent was a safety veto: an over-optimistic LLM judge shouldn't be able to
*inflate* a weak retrieval. But `min()` also lets a merely-*cautious* judge
**cap a strong one**. In deep mode the Groq judge returned a perfectly reasonable
0.8 ("relevant, multiple on-topic passages"), which then clamped the reranker's
calibrated 0.976. The retrieval was excellent; the *scoreboard* was broken.

### The fix — asymmetric, and honest about which signal to trust

The cross-encoder score is *calibrated* (a real relevance model). The LLM judge
is a coarse sanity check. So use the judge as a **veto, not a cap**:

```python
# after — the judge only pulls quality down when it flags weak context
if crag_quality >= 0.5:
    return round(top_rerank, 3)              # judge concurs → trust the reranker
return round(min(crag_quality, top_rerank), 3)  # judge objects → veto toward abstention
```

Deep now reports **0.966–0.976** — matching fast mode for identical retrieval,
while still abstaining when the judge genuinely flags weak context.
([`retrieve/engine.py::_grounded_quality`](src/scholar_rag/retrieve/engine.py))

### A subtlety worth surfacing

After moving reranking to Jina, the *displayed* score dropped to ~0.79 again —
but this time it's **calibration, not a bug**: Jina's relevance scores simply run
lower-magnitude than bge-reranker's (top ~0.7–0.8 vs ~0.97). The retrieved papers
are identical and relevant. Knowing the difference between "the metric is wrong"
and "the metric is on a different scale" is exactly the kind of thing that bites
teams who trust dashboards blindly.

---

## 5. Does "deep" retrieval actually do anything? (an honest evaluation)

It would be easy to add HyDE + multi-query + CRAG and *assume* they help. I
measured it. Comparing the retrieved paper sets, fast vs deep:

| Query | Overlap | Papers only deep found |
|---|---|---|
| "mixture of experts routing in transformers" | 4 / 6 | 2 |
| "how do RAG systems reduce hallucination" | 3 / 7 | 4 |
| "efficient fine-tuning on limited hardware" | 3 / 6 | 3 |

**30–70 % of deep's results were papers the plain query never surfaced** — the
expansion genuinely widens recall, most on natural-language questions where the
user's wording differs from the papers' vocabulary.

But it's a **recall/precision trade-off, not a free upgrade.** On the
hallucination query, HyDE's broadened vocabulary also pulled in an off-topic EDA
survey that the reranker scored 1.06 — a false positive from query drift. So:

- **Deep** for broad/exploratory review, or when a plain query returns little.
- **Fast** for precise, well-phrased questions where tight on-topic results and
  sub-2 s latency matter.

Reporting the trade-off honestly — rather than claiming deep is strictly better —
is the difference between a demo and an evaluation.

The UI makes this visible: each query renders a force-directed knowledge graph of
the retrieved papers (linked by semantic similarity, colored by relevance), and
in deep mode the papers surfaced *only* by query-expansion are marked with a pink
halo — so "what deep retrieval added" is something you can see, not just assert:

![Deep-mode knowledge graph — pink halos mark papers surfaced only by query-expansion](docs/assets/knowledge-graph.svg)

---

## 6. Measuring it — the RAGAS metric suite

Claiming the pipeline "works" isn't enough — I measure it. Four standard RAGAS
metrics run over a golden set of research questions, each with a reference answer:

- **faithfulness** — fraction of the answer's atomic claims inferable from the
  retrieved context (a hallucination check)
- **answer relevancy** — cosine similarity between the question and questions
  reverse-generated from the answer (embedded with BGE-M3)
- **context precision** — average-precision of the retrieved contexts judged
  useful (rewards relevant-*and-early* ranking)
- **context recall** — fraction of the reference answer's claims attributable to
  the retrieved context

**Implemented natively, not via the `ragas` package** — a deliberate call: the
library unconditionally imports a `langchain_community.chat_models.vertexai`
module that current LangChain no longer ships, an unresolvable conflict in this
environment. The metrics are just well-specified LLM-judge procedures, so I
reproduce them against the same Groq client the app uses — lighter, reproducible,
dependency-free, and consistent with the rest of the stack.

### Results — 6-question golden set, Groq (llama-3.3-70b) judge

| Metric | Fast (8B gen) | Deep (70B gen) | Δ |
|---|---|---|---|
| answer relevancy | 0.81 | 0.80 | −0.01 |
| context precision | 0.71 | 0.76 | **+0.05** |
| context recall | 0.63 | 0.57 | −0.05 |
| **faithfulness** | **0.34** | **0.47** | **+0.14** |

Two real findings fall out of this:

1. **Faithfulness was low in fast mode (0.34)**, consistently across questions
   (0.11–0.64): the fast 8B generator asserts claims not fully grounded in the
   retrieved passages. Measuring localized the problem to *generation*, not
   retrieval — the retrieval metrics were already solid — and pointed at a fix.

2. **The fix worked — partly.** Routing generation to the deep-mode 70B model
   lifted faithfulness **+40 % (0.34 → 0.47)**: the larger model stays closer to
   its context. But it isn't free — context recall dipped (−0.05), consistent
   with the query-drift trade-off measured in §5 (deep's HyDE / multi-query
   expansion broadens retrieval and can pull in context that doesn't cover the
   reference's specific claims).

The honest read: deep mode buys markedly more faithful, higher-precision answers
at a small cost in recall — one coherent story across the retrieval eval (§5) and
this generation eval. Faithfulness at 0.47 is still the number to chase next
(stricter citation enforcement, or reranking the generator's own claims against
the context).

Run it: `make eval` (fast) or `python -m scholar_rag.eval.run_eval --deep`.

### Chasing faithfulness — claim grounding

Faithfulness was the number to fix, so I added a post-generation grounding pass —
and the process of getting it right is more interesting than the feature.

**The obvious approach (cross-encoder reranking of claims) fails, instructively.**
The natural idea: score each generated claim against the passages with the
reranker, drop the unsupported ones. But a cross-encoder scores *topical
relevance*, not *entailment*. Calibrated on real output, every claim scores high
regardless of whether the context actually supports it:

```
0.945  "The Switch Transformer incorporates various mechanisms…"
0.549  "This is achieved through a gating mechanism that selects…"
0.963  "EvoMoE … has been shown to…"
```

A plausible-but-unsupported claim scores as high as a grounded one, because both
are *about* the topic. No threshold separates them — **reranking cannot ground
claims.** (Worth knowing before building on it.)

**The mechanism that works is LLM entailment verification**
([`_ground_claims`](src/scholar_rag/generate/cited_generator.py)): one batched
judge call verdicts each sentence against the retrieved context; unsupported
sentences are dropped, survivors kept and cited. On a sample MoE answer it dropped
2 of 9 claims while keeping the review coherent. It's wired into both generators
(`verify_claims`, on by default).

**Measuring the lift without gaming it.** The verifier and the eval judge must be
*different* models, or faithfulness rises by construction — you'd be deleting
exactly what the judge penalizes. So claims are verified by **Llama-4-Scout-17B**
and judged by a separate **Llama-3.1-8B** (`verify_model` / `EVAL_LLM_MODEL`).
Grounding off vs on, 6-question golden set:

| Metric | grounding off | grounding on | Δ |
|---|---|---|---|
| **faithfulness** | 0.53 | **0.82** | **+0.30 (+56 %)** |
| answer relevancy | 0.83 | 0.88 | +0.05 |
| context precision | 0.61 | 0.61 | 0.00 |
| context recall | 0.81 | 0.81 | 0.00 |

Claim-grounding lifts faithfulness **+56 %**, and answer relevancy edges up too
(pruning unsupported sentences tightens the answer). The retrieval metrics are
**identical** — exactly as they should be, because grounding is a generation-side
post-process that never touches retrieval. That invariance is the tell that the
intervention does what it claims and the number isn't an artifact.

(The judge is 8B rather than 70B only because Groq's free tier caps the 70B at
100 K tokens/day; the *delta* is apples-to-apples since both runs share the judge,
and the verifier is a third, independent model so the comparison isn't circular.)

---

## 7. Notable design decisions

- **Hybrid over pure-dense.** Dense vectors miss exact terms (model names,
  datasets, `arxiv_id`s); sparse lexical weights catch them. RRF fuses the two
  rankings without tuning a score-scale mixing weight.
- **Citation-graph reranking.** Pure relevance ignores that some papers *are* the
  seminal reference. Blending `β·log(1+cited_by_count)` (diminishing returns on
  raw popularity) with `γ·recency` surfaces influential-and-current work.
- **Calibrated abstention.** The generator refuses when the top reranked passage
  is weak — measured on the cross-encoder's scale, not the LLM's self-assessment.
  A RAG system that can say "I don't know" is more trustworthy than one that
  always answers.
- **INT8 quantization** on the Qdrant vectors keeps the 147K-chunk corpus inside
  the free-tier memory budget with negligible recall loss.
- **Null-year-safe filtering.** OpenAlex doesn't index days-old arXiv papers, so
  ~79 % of chunks initially had `year=null`; a naïve range filter silently
  dropped them. The filter now matches *year-in-range OR year-is-null*.

---

## 8. What I'd do next

- **A/B the citation-graph reranker** with RAGAS (context precision/recall) to
  quantify its lift over plain vector search, and tune α/β/γ against it.
- **Tighten deep-mode precision** — the query-drift false positives suggest
  either dropping the most speculative HyDE variant or a relevance floor after
  reranking.
- **Cache query embeddings** for repeated/similar questions.
- **Streaming citations first** (already partially done — the source list is sent
  before the first generated token, so the UI shows provenance immediately).

---

## Appendix — running it

- **Local (self-contained):** BGE-M3 + bge-reranker + Ollama, Qdrant in Docker.
  `make up && make serve`.
- **Hosted (8 GB-friendly):** set `DEEP_LLM_API_KEY` (Groq) + `JINA_API_KEY`;
  reranking and generation move to APIs, only BGE-M3 stays local. Deploy to a
  free Hugging Face Space — see [`deploy/HF_SPACES_DEPLOY.md`](deploy/HF_SPACES_DEPLOY.md).

*Every number in this document was measured on the target 8 GB machine during
development, not estimated.*
