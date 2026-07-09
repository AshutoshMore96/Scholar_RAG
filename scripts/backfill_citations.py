"""
Backfill cited_by_count into the Qdrant collection from OpenAlex — no re-embed.
Fetches citation counts (batched, 50 DOIs/request) and updates payloads in place
via set_payload with a paper_id filter. Only papers OpenAlex actually has get
updated (2026 papers stay 0, since they aren't indexed yet).
"""
from __future__ import annotations
import os, time, httpx
from dotenv import load_dotenv
load_dotenv("/Users/ashutosh/PycharmProjects/RAGPipeline/scholar-rag/.env")
from qdrant_client import QdrantClient
from qdrant_client.http import models as qm

COLL = "scholar_rag"
c = QdrantClient(url=os.environ["QDRANT_URL"], api_key=os.environ["QDRANT_API_KEY"], timeout=60)

# 1. distinct paper_ids in the collection
print("scanning collection for distinct papers…", flush=True)
paper_ids: set[str] = set()
off = None
while True:
    pts, off = c.scroll(COLL, limit=1024, offset=off, with_payload=["paper_id"])
    for p in pts:
        pid = p.payload.get("paper_id")
        if pid:
            paper_ids.add(pid)
    if off is None:
        break
print(f"distinct papers: {len(paper_ids)}", flush=True)

# 2. batched OpenAlex lookup by arXiv DOI
MAILTO = "scholar@example.com"
oc = httpx.Client(timeout=30, follow_redirects=True,
                  headers={"User-Agent": f"ScholarRAG/0.1 (mailto:{MAILTO})"})
doi_of = lambda a: "10.48550/arxiv." + a.split("v")[0]
doi2aid = {doi_of(a): a for a in paper_ids}
dois = list(doi2aid)
cite: dict[str, int] = {}
status = {}
for i in range(0, len(dois), 50):
    batch = dois[i:i + 50]
    try:
        r = oc.get("https://api.openalex.org/works",
                   params={"filter": "doi:" + "|".join(batch), "per-page": 50, "mailto": MAILTO})
        status[r.status_code] = status.get(r.status_code, 0) + 1
        if r.status_code == 200:
            for w in r.json().get("results", []):
                wdoi = (w.get("doi") or "").replace("https://doi.org/", "").lower()
                aid = doi2aid.get(wdoi)
                n = w.get("cited_by_count") or 0
                if aid and n > 0:
                    cite[aid] = int(n)
    except Exception as e:
        status["err"] = status.get("err", 0) + 1
    time.sleep(0.3)
    if (i // 50) % 5 == 0:
        print(f"  OpenAlex {i+len(batch)}/{len(dois)} — matched so far: {len(cite)}", flush=True)
print(f"OpenAlex status: {status} | papers with citations: {len(cite)}", flush=True)

# 3. update payloads in place (all points of each paper)
for j, (aid, n) in enumerate(cite.items(), 1):
    c.set_payload(COLL, payload={"cited_by_count": n},
                  points=qm.Filter(must=[qm.FieldCondition(key="paper_id",
                                                           match=qm.MatchValue(value=aid))]))
    if j % 100 == 0:
        print(f"  updated {j}/{len(cite)} papers", flush=True)
print(f"payload update done for {len(cite)} papers", flush=True)

# 4. verify
gt0 = c.count(COLL, count_filter=qm.Filter(must=[qm.FieldCondition(
    key="cited_by_count", range=qm.Range(gt=0))])).count
total = c.count(COLL).count
top = sorted(cite.values(), reverse=True)[:5]
print(f"RESULT: points with cited_by_count>0 = {gt0}/{total} | top citation counts: {top}", flush=True)
