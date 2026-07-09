"""Generates ScholarRAG_Colab_Ingest.ipynb — run once to (re)build the notebook."""
import json, pathlib

def md(src):  return {"cell_type": "markdown", "metadata": {}, "source": src}
def code(src): return {"cell_type": "code", "metadata": {}, "outputs": [], "execution_count": None, "source": src}

cells = []

cells.append(md("""# ScholarRAG — GPU Ingestion on Colab (T4)

Crawl arXiv → parse → chunk → **embed on GPU (BGE-M3)** → write to Qdrant, in a
schema **identical to the local ScholarRAG server**, so your FastAPI/Streamlit
can query the result unchanged.

**Before running:** `Runtime ▸ Change runtime type ▸ Hardware accelerator ▸ T4 GPU`.

Two ways to persist the vectors (set in the Config cell):
1. **Qdrant Cloud** (recommended, free 1 GB tier) — point `QDRANT_URL`/`QDRANT_API_KEY`
   at your cluster. Your local server then just needs those same creds.
2. **Local + snapshot** — leave `QDRANT_URL` blank; the notebook runs Qdrant
   on-disk inside Colab and produces a downloadable snapshot to restore into your
   local Docker Qdrant."""))

cells.append(md("## 1. Install dependencies & check GPU"))
cells.append(code("""!pip install -q FlagEmbedding qdrant-client httpx pymupdf tenacity tqdm
import torch
print("CUDA available:", torch.cuda.is_available(), "|", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU only")
assert torch.cuda.is_available(), "Enable the T4 GPU: Runtime > Change runtime type > T4 GPU"
"""))

cells.append(md("## 2. Config — edit these"))
cells.append(code('''# ── corpus scope ───────────────────────────────────────────────────────────
CATEGORIES  = ["cs.CL", "cs.LG", "cs.AI", "cs.IR"]
QUERY_TERMS = [
    "large language model", "retrieval-augmented generation", "quantization",
    "knowledge distillation", "parameter-efficient fine-tuning", "instruction tuning",
    "reinforcement learning from human feedback", "in-context learning",
    "chain-of-thought reasoning", "mixture of experts", "model alignment", "hallucination",
]
MAX_PAPERS_PER_CATEGORY = 250          # 4 cats x 250 = up to 1000 papers
DATE_FROM = "2023-01-01"

# ── embedding (GPU) ────────────────────────────────────────────────────────
BGE_MODEL   = "BAAI/bge-m3"
EMBED_BATCH = 128                       # T4 handles 128 comfortably at fp16
MAX_LENGTH  = 256

# ── chunking ───────────────────────────────────────────────────────────────
MAX_CHILD_TOKENS  = 512                  # 512 keeps chunk COUNT (and storage) in check;
                                         #   256 doubles the point count. ~1585 papers @512 ~ 60k pts.
MAX_PARENT_TOKENS = 1024
STORE_PARENT_TEXT = True                 # False = don't store the parent window on each child
                                         #   (saves the most storage; loses child->parent context at gen time)
FORCE_RECHUNK     = False                # True = re-parse+re-chunk even if a cached chunks.pkl exists
                                         #   (set True after changing MAX_CHILD_TOKENS / parent sizing)

# ── Qdrant target ──────────────────────────────────────────────────────────
COLLECTION     = "scholar_rag"
QDRANT_URL     = ""                      # e.g. "https://xyz.cloud.qdrant.io:6333"; blank = local+snapshot
QDRANT_API_KEY = ""                      # Qdrant Cloud API key (if using cloud)
MAX_POINTS_CAP = 0                       # 0 = no cap. Set e.g. 70000 to stop before the free 1 GB tier fills.
RECREATE_COLLECTION = False              # False = keep an existing collection (resume-safe). True = wipe & rebuild
                                         #   (set True once after re-chunking, since chunk ids change)

# ── enrichment ─────────────────────────────────────────────────────────────
ENRICH_OPENALEX = False                 # True = citation counts/venue (slower ~1s/paper); year is backfilled from arXiv regardless
'''))

cells.append(md("""## 2.5 Setup — mount Drive & cache paths

Every stage caches to Drive so re-runs (and fresh sessions after a disconnect)
**skip work already done**: PDFs, chunks, enrichment, and the embed checkpoint
all persist here."""))
cells.append(code('''import os
try:
    from google.colab import drive; drive.mount("/content/drive")
    BK = "/content/drive/MyDrive/scholar_rag_backup"
except Exception:
    BK = "/content/scholar_rag_backup"     # no Drive -> local (won't survive disconnect)
PDF_DIR    = f"{BK}/pdfs"
CHUNKS_PKL = f"{BK}/chunks.pkl"
CITE_PKL   = f"{BK}/cite_map.pkl"
META_PKL   = f"{BK}/papers_meta.pkl"
CKPT       = f"{BK}/embed_ckpt.txt"
os.makedirs(PDF_DIR, exist_ok=True)
print("cache dir:", BK)
'''))

cells.append(md("## 3. arXiv crawler"))
cells.append(code('''import httpx, time, re
from xml.etree import ElementTree as ET
from dataclasses import dataclass, field
from tqdm.auto import tqdm

NS = {"atom": "http://www.w3.org/2005/Atom", "arxiv": "http://arxiv.org/schemas/atom"}
_client = httpx.Client(timeout=60.0, follow_redirects=True)

@dataclass
class Paper:
    arxiv_id: str; title: str; abstract: str; published: str
    categories: list = field(default_factory=list); pdf_bytes: bytes = b""

def _build_query(cats, terms, date_from):
    clauses = ["(" + " OR ".join(f"cat:{c}" for c in cats) + ")"]
    if terms:
        tc = []
        for t in terms:
            tt = f'"{t}"' if " " in t else t
            tc += [f"abs:{tt}", f"ti:{tt}"]
        clauses.append("(" + " OR ".join(tc) + ")")
    if date_from:
        clauses.append(f"submittedDate:[{date_from.replace('-','')}0000 TO 20991231235959]")
    return " AND ".join(clauses)

def fetch_meta(cats, terms, max_results, date_from):
    q = _build_query(cats, terms, date_from); out = []; fetched = 0
    while fetched < max_results:
        params = {"search_query": q, "start": fetched, "max_results": min(100, max_results - fetched),
                  "sortBy": "submittedDate", "sortOrder": "descending"}
        r = _client.get("https://export.arxiv.org/api/query", params=params); r.raise_for_status()
        root = ET.fromstring(r.text); entries = root.findall("atom:entry", NS)
        if not entries: break
        for e in entries:
            aid = (e.findtext("atom:id", "", NS) or "").split("/abs/")[-1]
            if not aid: continue
            out.append(Paper(
                arxiv_id=aid,
                title=re.sub(r"\\s+"," ",(e.findtext("atom:title","",NS) or "").strip()),
                abstract=(e.findtext("atom:summary","",NS) or "").strip(),
                published=(e.findtext("atom:published","",NS) or "")[:10],
                categories=[c.get("term","") for c in e.findall("atom:category",NS)],
            ))
        fetched += len(entries)
        if len(entries) < 100: break
        time.sleep(3.0)   # arXiv rate limit
    return out

papers = []
for cat in CATEGORIES:
    got = fetch_meta([cat], QUERY_TERMS, MAX_PAPERS_PER_CATEGORY, DATE_FROM)
    print(f"{cat}: {len(got)} papers")
    papers += got
# dedupe by arxiv_id
seen = {}; papers = [seen.setdefault(p.arxiv_id, p) for p in papers if p.arxiv_id not in seen]
print("TOTAL unique papers:", len(papers))
'''))

cells.append(md("""## 4. Download PDFs — skips ones already on Drive

Each PDF is saved to `PDF_DIR`; re-runs load existing files from disk instead of
re-fetching, so an interrupted download resumes and never re-hits arXiv for what
you already have."""))
cells.append(code('''import concurrent.futures as cf

def _fn(aid): return f"{PDF_DIR}/{aid.replace('/', '_')}.pdf"

def dl(p):
    fn = _fn(p.arxiv_id)
    if os.path.exists(fn) and os.path.getsize(fn) > 0:   # already downloaded -> load from disk
        p.pdf_bytes = open(fn, "rb").read(); return p
    try:
        r = _client.get(f"https://arxiv.org/pdf/{p.arxiv_id}")
        if r.status_code == 200 and r.content[:4] == b"%PDF":
            open(fn, "wb").write(r.content); p.pdf_bytes = r.content
    except Exception:
        pass
    return p

todo = [p for p in papers if not (os.path.exists(_fn(p.arxiv_id)) and os.path.getsize(_fn(p.arxiv_id)) > 0)]
print(f"{len(papers) - len(todo)} already on disk, downloading {len(todo)}")
with cf.ThreadPoolExecutor(max_workers=4) as ex:   # modest concurrency = polite to arXiv
    list(tqdm(ex.map(dl, papers), total=len(papers), desc="PDFs (cached+new)"))
papers = [p for p in papers if getattr(p, "pdf_bytes", b"")]
import pickle
pickle.dump([{k: v for k, v in p.__dict__.items() if k != "pdf_bytes"} for p in papers],
            open(META_PKL, "wb"))   # metadata cached for the restore cell
print("papers with PDF:", len(papers))
'''))

cells.append(md("## 5. Parse (PyMuPDF) + semantic chunk"))
cells.append(code('''import fitz, uuid  # PyMuPDF

def approx_tokens(t): return max(1, len(t)//4)

def parse_pdf(b):
    try:
        doc = fitz.open(stream=b, filetype="pdf")
        return "\\n\\n".join(page.get_text() for page in doc)
    except Exception:
        return ""

SENT = re.compile(r"(?<=[.!?])\\s+")
def semantic_units(para, max_child):
    units, cur, tok = [], [], 0
    for s in SENT.split(para):
        s = s.strip()
        if not s: continue
        st = approx_tokens(s)
        if cur and tok + st > max_child:
            units.append(" ".join(cur)); cur, tok = [], 0
        cur.append(s); tok += st
    if cur: units.append(" ".join(cur))
    return units

def parent_windows(text, max_parent):
    words = text.split()
    if not words: return []
    wpw = max(64, int(max_parent * 0.75)); step = max(wpw//2, wpw - 96); out = []  # ~0.75 words/token
    for i in range(0, len(words), step):
        out.append(" ".join(words[i:i+wpw]))
        if i + wpw >= len(words): break
    return out

@dataclass
class Chunk:
    chunk_id: str; paper_id: str; parent_id: str; text: str
    title: str; year: int; section: str; parent_text: str

def chunk_paper(p, year):
    md_text = parse_pdf(p.pdf_bytes) or f"# {p.title}\\n\\n{p.abstract}"
    children = []
    for pw in parent_windows(md_text, MAX_PARENT_TOKENS):
        pid = str(uuid.uuid4())
        for para in [x.strip() for x in pw.split("\\n\\n") if x.strip()]:
            if approx_tokens(para) < 20: continue
            for u in semantic_units(para, MAX_CHILD_TOKENS):
                if approx_tokens(u) < 8: continue
                children.append(Chunk(str(uuid.uuid4()), p.arxiv_id, pid, u,
                                      p.title, year, "body", pw))
    return children

def year_of(p):
    try: return int(p.published[:4])
    except Exception: return None

import pickle
from dataclasses import asdict
if os.path.exists(CHUNKS_PKL) and not FORCE_RECHUNK:
    chunk_dicts = pickle.load(open(CHUNKS_PKL, "rb"))          # cached -> skip parsing
    print(f"loaded {len(chunk_dicts)} cached chunks (set FORCE_RECHUNK=True to redo)")
else:
    all_chunks = []
    for p in tqdm(papers, desc="Parsing+chunking"):
        all_chunks += chunk_paper(p, year_of(p))
    chunk_dicts = [asdict(c) for c in all_chunks]
    pickle.dump(chunk_dicts, open(CHUNKS_PKL, "wb"))
    print(f"parsed+chunked {len(chunk_dicts)} chunks -> cached to {CHUNKS_PKL}")
'''))

cells.append(md("""## 6. (Optional) OpenAlex enrichment — batched & transparent

**Expect `enriched: 0` for a newest-first crawl:** OpenAlex indexes arXiv papers
with a lag, so brand-new (2026) papers aren't there yet (HTTP 404). `year` is
backfilled from the arXiv date regardless, so filters/recency still work. Turn
this on and set `DATE_FROM` to an older window (e.g. 2022–2024) only if you want
citation counts. This version batches 50 DOIs per request (fast, polite pool) and
prints the HTTP status breakdown so 0 is never a mystery."""))
cells.append(code('''import pickle
cite_map = {}  # arxiv_id -> {cited_by_count, venue, concepts}
if os.path.exists(CITE_PKL):                       # cached -> skip re-enrichment
    cite_map = pickle.load(open(CITE_PKL, "rb"))
    print(f"loaded cached cite_map: {len(cite_map)}")
elif ENRICH_OPENALEX:
    MAILTO = "scholar@example.com"
    oc = httpx.Client(timeout=30, follow_redirects=True,
                      headers={"User-Agent": f"ScholarRAG/0.1 (mailto:{MAILTO})"})
    doi_of = lambda aid: "10.48550/arxiv." + aid.split("v")[0]
    doi2aid = {doi_of(p.arxiv_id): p.arxiv_id for p in papers}
    dois = list(doi2aid)
    status = {}
    for i in tqdm(range(0, len(dois), 50), desc="OpenAlex (batched x50)"):
        batch = dois[i:i+50]
        try:
            r = oc.get("https://api.openalex.org/works",
                       params={"filter": "doi:" + "|".join(batch), "per-page": 50, "mailto": MAILTO})
            status[r.status_code] = status.get(r.status_code, 0) + 1
            if r.status_code == 200:
                for w in r.json().get("results", []):
                    wdoi = (w.get("doi") or "").replace("https://doi.org/", "").lower()
                    aid = doi2aid.get(wdoi)
                    if aid:
                        cite_map[aid] = {
                            "cited_by_count": w.get("cited_by_count", 0),
                            "venue": ((w.get("primary_location") or {}).get("source") or {}).get("display_name"),
                            "concepts": [c["display_name"] for c in w.get("concepts", [])[:8]],
                        }
        except Exception:
            status["err"] = status.get("err", 0) + 1
        time.sleep(0.2)
    print("OpenAlex request status counts:", status)
    print(f"enriched: {len(cite_map)} / {len(papers)}")
    pickle.dump(cite_map, open(CITE_PKL, "wb"))     # cache
    if not cite_map:
        print("NOTE: 0 is expected for newest-first crawls — new arXiv papers aren't "
              "indexed by OpenAlex yet (404). Year is backfilled from arXiv anyway. "
              "Set DATE_FROM to an older window (e.g. 2022-2024) to get citation data.")
'''))

cells.append(md("## 7. Connect to Qdrant + create collection (schema matches local server)"))
cells.append(code('''from qdrant_client import QdrantClient
from qdrant_client.http import models as qm

if QDRANT_URL:
    client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY or None)
    print("Using Qdrant Cloud:", QDRANT_URL)
else:
    client = QdrantClient(path="/content/qdrant_local")   # on-disk, snapshot later
    print("Using local on-disk Qdrant at /content/qdrant_local")

DENSE, SPARSE, DIM = "dense", "sparse", 1024
existing = [c.name for c in client.get_collections().collections]
if COLLECTION in existing and RECREATE_COLLECTION:
    client.delete_collection(COLLECTION); existing.remove(COLLECTION)
    print("deleted existing collection (RECREATE_COLLECTION=True)")

if COLLECTION not in existing:
    client.create_collection(
        collection_name=COLLECTION,
        vectors_config={DENSE: qm.VectorParams(size=DIM, distance=qm.Distance.COSINE, on_disk=True)},
        sparse_vectors_config={SPARSE: qm.SparseVectorParams(index=qm.SparseIndexParams(on_disk=False))},
    )
    for f, t in [("paper_id", qm.PayloadSchemaType.KEYWORD), ("year", qm.PayloadSchemaType.INTEGER),
                 ("cited_by_count", qm.PayloadSchemaType.INTEGER), ("section", qm.PayloadSchemaType.KEYWORD),
                 ("concepts", qm.PayloadSchemaType.KEYWORD)]:
        client.create_payload_index(collection_name=COLLECTION, field_name=f, field_schema=t)
    print("created collection:", COLLECTION)
else:
    print(f"keeping existing collection '{COLLECTION}' with "
          f"{client.count(COLLECTION).count} points (set RECREATE_COLLECTION=True to wipe)")
'''))

cells.append(md("""## 7.5 Restore after a disconnect (skip crawl/download/parse)

Every stage above already cached to Drive, so after a disconnect you don't re-run
them. Just re-run **imports + Config + Setup (2.5)**, then this cell, then the
Qdrant cell (7) and the embed cell (8) — it resumes from its checkpoint."""))
cells.append(code('''from types import SimpleNamespace
import pickle
meta = pickle.load(open(META_PKL, "rb"))
_pdf = lambda a: (open(f"{PDF_DIR}/{a.replace('/', '_')}.pdf", "rb").read()
                  if os.path.exists(f"{PDF_DIR}/{a.replace('/', '_')}.pdf") else b"")
papers      = [SimpleNamespace(**m, pdf_bytes=_pdf(m["arxiv_id"])) for m in meta]
chunk_dicts = pickle.load(open(CHUNKS_PKL, "rb"))
cite_map    = pickle.load(open(CITE_PKL, "rb")) if os.path.exists(CITE_PKL) else {}
print(f"restored {len(papers)} papers, {len(chunk_dicts)} chunks, {len(cite_map)} enriched")
'''))

cells.append(md("""## 8. Embed on GPU — resumable, checkpointed, storage-aware

- **Resumable:** checkpoints the chunk index to Drive after every batch; a
  disconnect just means re-run this cell (it picks up where it stopped).
- **Storage guard:** prints live point count; stops at `MAX_POINTS_CAP` (if set)
  so you don't blow past the Qdrant Cloud free 1 GB tier mid-run.
- Honors `STORE_PARENT_TEXT` (drop it to save the most space)."""))
cells.append(code('''from FlagEmbedding import BGEM3FlagModel
import os, uuid   # CKPT / paths come from the Setup cell (2.5)

# work off dicts so this cell runs whether chunks are in memory or just restored
try:
    chunk_dicts
except NameError:
    chunk_dicts = [c if isinstance(c, dict) else c.__dict__ for c in all_chunks]

start = int(open(CKPT).read()) if os.path.exists(CKPT) else 0
print(f"resuming at chunk {start}/{len(chunk_dicts)}  (STORE_PARENT_TEXT={STORE_PARENT_TEXT}, cap={MAX_POINTS_CAP or 'none'})")

model = BGEM3FlagModel(BGE_MODEL, use_fp16=True, device="cuda")

def upsert_batch(cd, dense, sparse):
    pts = []
    for c, dv, sv in zip(cd, dense, sparse):
        cit = cite_map.get(c["paper_id"], {})
        payload = {"paper_id": c["paper_id"], "chunk_id": c["chunk_id"], "parent_id": c["parent_id"],
                   "arxiv_id": c["paper_id"], "title": c["title"], "section": c["section"],
                   "text": c["text"], "year": c["year"], "venue": cit.get("venue"),
                   "concepts": cit.get("concepts", []), "cited_by_count": cit.get("cited_by_count", 0)}
        if STORE_PARENT_TEXT:
            payload["parent_text"] = c["parent_text"]
        pts.append(qm.PointStruct(
            id=str(uuid.uuid5(uuid.NAMESPACE_DNS, c["chunk_id"])),
            vector={"dense": dv.tolist(),
                    "sparse": qm.SparseVector(indices=[int(k) for k in sv.keys()],
                                              values=[float(v) for v in sv.values()])},
            payload=payload))
    client.upsert(collection_name=COLLECTION, points=pts)

texts = [c["text"] for c in chunk_dicts]
for i in tqdm(range(start, len(texts), EMBED_BATCH), desc="Embedding (GPU)"):
    out = model.encode(texts[i:i+EMBED_BATCH], batch_size=EMBED_BATCH, max_length=MAX_LENGTH,
                       return_dense=True, return_sparse=True, return_colbert_vecs=False)
    upsert_batch(chunk_dicts[i:i+EMBED_BATCH], out["dense_vecs"], out["lexical_weights"])
    open(CKPT, "w").write(str(i + EMBED_BATCH))
    if MAX_POINTS_CAP and (i + EMBED_BATCH) >= MAX_POINTS_CAP:
        print(f"reached MAX_POINTS_CAP={MAX_POINTS_CAP}; stopping."); break

print("points in collection:", client.count(collection_name=COLLECTION).count)
'''))

cells.append(md("## 9. Sanity check — hybrid-ish search"))
cells.append(code('''q = "large language model quantization methods"
qo = model.encode([q], return_dense=True, return_sparse=True, return_colbert_vecs=False)
hits = client.query_points(collection_name=COLLECTION, query=qo["dense_vecs"][0].tolist(),
                           using="dense", limit=5, with_payload=["title","year","paper_id"]).points
for h in hits:
    print(round(h.score,3), "|", (h.payload.get("title") or "")[:70])
'''))

cells.append(md("""## 10. Persist / use the result

**If you used Qdrant Cloud:** you're done — the collection is live. On your local
machine, just set these in `scholar-rag/.env` (NOT `.env.example`):
```
QDRANT_URL=<your cloud url>
QDRANT_API_KEY=<your key>
```
`QdrantStore` already honors them (they override `QDRANT_HOST/PORT`), so the local
FastAPI/Streamlit query the cloud collection with no code change.

**If you used local on-disk Qdrant:** create + download a snapshot, then restore
into your local Docker Qdrant."""))
cells.append(code('''if not QDRANT_URL:
    snap = client.create_snapshot(collection_name=COLLECTION)
    print("snapshot:", snap)
    # Zip the storage dir for download
    import shutil
    shutil.make_archive("/content/scholar_rag_qdrant", "zip", "/content/qdrant_local")
    from google.colab import files
    files.download("/content/scholar_rag_qdrant.zip")
    print("Downloaded scholar_rag_qdrant.zip — see README for restoring into local Docker Qdrant.")
'''))

nb = {"cells": cells,
      "metadata": {"accelerator": "GPU",
                   "colab": {"provenance": [], "gpuType": "T4"},
                   "kernelspec": {"name": "python3", "display_name": "Python 3"},
                   "language_info": {"name": "python"}},
      "nbformat": 4, "nbformat_minor": 0}

out = pathlib.Path(__file__).parent / "ScholarRAG_Colab_Ingest.ipynb"
out.write_text(json.dumps(nb, indent=1))
print("wrote", out, "with", len(cells), "cells")
