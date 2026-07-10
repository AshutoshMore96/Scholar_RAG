# GPU Ingestion on Colab (T4)

`ScholarRAG_Colab_Ingest.ipynb` builds a 500–1000 paper corpus on a free Colab
T4 GPU in ~15–30 min (vs. 4–8 h on CPU), writing vectors in the **exact schema
the local ScholarRAG server expects**.

## Why GPU (and not PySpark)

- **Embedding is the bottleneck** and it's GPU-bound: BGE-M3 runs ~15–30× faster
  on a T4 than on CPU. The notebook uses `use_fp16=True` + batch 128.
- **PySpark won't help at this scale.** The other stages are rate-limited network
  I/O (arXiv asks ≥3 s between requests — you can't parallelize past that) or
  GPU-bound work (helped by batching, not by distributing across nodes you don't
  have). Spark only pays off at 100k+ papers on a real multi-GPU cluster.

## Steps

1. Upload the notebook to [Colab](https://colab.research.google.com/) →
   `Runtime ▸ Change runtime type ▸ T4 GPU`.
2. Edit the **Config** cell (categories, query terms, paper count, Qdrant target).
3. Run all cells.

## Persisting the result — two options

### A. Qdrant Cloud (recommended)
1. Create a free cluster at https://cloud.qdrant.io (1 GB tier fits ~1000 papers).
2. In the notebook Config cell set `QDRANT_URL` and `QDRANT_API_KEY`.
3. Back on your machine, put the same two values in `scholar-rag/.env`:
   ```
   QDRANT_URL=https://<cluster>.cloud.qdrant.io:6333
   QDRANT_API_KEY=<key>
   ```
   `QdrantStore` already honors these (they override `QDRANT_HOST/PORT`), so the
   FastAPI server queries the cloud collection with no code change.

### B. Local on-disk + snapshot
1. Leave `QDRANT_URL` blank — the notebook runs Qdrant on-disk in Colab and
   downloads `scholar_rag_qdrant.zip` (a snapshot) at the end.
2. Restore it into your local Docker Qdrant:
   ```bash
   # copy the snapshot into the qdrant volume, then via the API:
   curl -X POST "http://localhost:6333/collections/scholar_rag/snapshots/upload" \
        -H "Content-Type:multipart/form-data" \
        -F "snapshot=@scholar_rag.snapshot"
   ```
   (See the Qdrant snapshots docs for the exact restore command for your version.)

## Notes

- The notebook uses **PyMuPDF** for parsing (fast) rather than Nougat/GROBID —
  fine for text; equations aren't LaTeX-preserved. Run the local pipeline if you
  need Nougat-quality math parsing.
- Chunking is **semantic** (no per-paragraph LLM), matching the local CPU default.
- OpenAlex enrichment is **off by default** in the notebook for speed; `year` is
  still backfilled from the arXiv date so year filters and the recency prior work.
- To rebuild the notebook after editing `_build_colab_nb.py`: `python _build_colab_nb.py`.
