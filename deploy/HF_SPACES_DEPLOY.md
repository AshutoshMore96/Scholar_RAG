# Deploying ScholarRAG to Hugging Face Spaces (free tier)

The app is already set up for a **Docker Space**: the `Dockerfile` at the repo
root serves the FastAPI app (with the web UI at `/`) on port **7860**, with
BGE-M3 baked into the image. Reranking (Jina), generation (Groq) and vectors
(Qdrant Cloud) are all external APIs, so the Space only needs ~3–4 GB RAM —
well within the free tier's 16 GB.

Everything here is free and needs no credit card.

---

## 1. Create the Space

1. Go to <https://huggingface.co/new-space> (sign in / sign up — free).
2. **Owner**: you. **Space name**: e.g. `scholar-rag`.
3. **SDK**: choose **Docker** → **Blank**.
4. **Hardware**: `CPU basic · 2 vCPU · 16 GB` (free).
5. Visibility: Public or Private (both free). Create the Space.

You now have an empty git repo at
`https://huggingface.co/spaces/<you>/scholar-rag`.

---

## 2. Add secrets and variables

In the Space: **Settings → Variables and secrets**.

**Secrets** (sensitive — stored encrypted, never shown in logs):

| Name | Value |
|------|-------|
| `QDRANT_URL` | your Qdrant Cloud URL (`https://…cloud.qdrant.io:6333`) |
| `QDRANT_API_KEY` | your Qdrant Cloud API key |
| `DEEP_LLM_API_KEY` | your Groq key (`gsk_…`) |
| `JINA_API_KEY` | your Jina key (`jina_…`) |

**Variables** (non-sensitive config):

| Name | Value |
|------|-------|
| `QDRANT_COLLECTION` | `scholar_rag` |
| `DEEP_LLM_MODEL` | `llama-3.3-70b-versatile` |
| `FAST_LLM_MODEL` | `llama-3.1-8b-instant` |

> These are the only 7 vars the query server reads. GROBID / MLflow / Ollama
> are **not** used at query time. `.env` is git-ignored and `.dockerignore`d, so
> your local secrets never reach the Space — the Space injects them at runtime.

---

## 3. Push the code

From the project directory (`scholar-rag/`):

```bash
# one-time: log in with a HF token that has "write" scope
#   (https://huggingface.co/settings/tokens)
pip install -U huggingface_hub
huggingface-cli login

# init a repo here if there isn't one, then add the Space as a remote
git init                       # skip if already a git repo
git add .                      # .gitignore keeps .env, .venv, data/raw out
git commit -m "Deploy ScholarRAG to HF Spaces"

git remote add space https://huggingface.co/spaces/<you>/scholar-rag
git push space HEAD:main       # HF Spaces build on the `main` branch
```

> If the push is rejected because the Space already has a commit (the auto-created
> README), run `git pull space main --allow-unrelated-histories` first, or force
> with `git push -f space HEAD:main` (fine for a fresh Space).

The Space starts building automatically. **First build ~8–12 min** (installs
CPU torch + downloads/bakes BGE-M3). Watch **Logs** in the Space UI.

---

## 4. Verify

- When the build finishes, the Space shows the ScholarRAG UI at
  `https://<you>-scholar-rag.hf.space`.
- Logs should show:
  - `Fast generation via Groq: llama-3.1-8b-instant`
  - `Deep generation via hosted API: … (llama-3.3-70b-versatile)`
  - `Reranker: Jina API (jina-reranker-v2-base-multilingual)` (on first query)
  - `Models pre-warmed — first query will be fast.`
- Ask a question in the UI, or hit the API:
  ```bash
  curl -N -X POST https://<you>-scholar-rag.hf.space/ask/stream \
    -H 'Content-Type: application/json' \
    -d '{"query":"knowledge distillation for LLMs","deep":false,"top_k":6}'
  ```

---

## Notes / gotchas

- **Idle behavior**: a free Space sleeps after ~48 h of no traffic; the next
  visit triggers a ~30–60 s wake (container restart + model load), then it's warm
  again for the session.
- **The whole chain must be up**: Qdrant Cloud's free cluster can also pause on
  inactivity — if queries return no passages, check the Qdrant dashboard.
- **Redeploy**: make changes locally, then `git commit` and `git push space
  HEAD:main` again. Only code changes trigger a rebuild; changing a Secret/Variable
  restarts the container without a full rebuild.
- **Updating the corpus** (ingesting new papers) is a separate, heavier job — run
  it locally / on Colab as before; it writes to the same Qdrant Cloud collection
  the Space reads. The Space itself is read-only over the corpus.
- `Dockerfile.local` is the original FastAPI+Streamlit image, kept for reference;
  HF Spaces uses `Dockerfile`.
