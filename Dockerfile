# ScholarRAG — Hugging Face Spaces (Docker SDK) image.
#
# Serves the FastAPI app (which also hosts the web UI at "/") on port 7860.
# All heavy services are external APIs — reranking (Jina), generation (Groq),
# vectors (Qdrant Cloud) — so the only in-process model is BGE-M3 for query
# embedding. It is baked into the image below so cold starts are fast.
#
# Secrets (QDRANT_URL, QDRANT_API_KEY, DEEP_LLM_API_KEY, JINA_API_KEY, …) are
# injected by HF Spaces at runtime — never bake them into the image.
FROM python:3.11-slim

# System build deps (FlagEmbedding / torch need a compiler for some wheels).
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential git curl \
    && rm -rf /var/lib/apt/lists/*

# HF Spaces runs the container as uid 1000 — set up a matching non-root user.
RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH \
    HF_HOME=/home/user/.cache/huggingface \
    PYTHONUNBUFFERED=1 \
    TOKENIZERS_PARALLELISM=false

WORKDIR /app

# Install CPU-only torch first so pip doesn't pull the ~2 GB CUDA build.
RUN pip install --no-cache-dir --user \
    torch --index-url https://download.pytorch.org/whl/cpu

# Install the package (remaining deps resolved from pyproject; torch already
# satisfied so the CPU build is kept).
COPY --chown=user:user pyproject.toml README.md ./
COPY --chown=user:user src ./src
RUN pip install --no-cache-dir --user -e .

# App assets needed at query time: config + the 780 KB citation graph.
COPY --chown=user:user configs ./configs
COPY --chown=user:user data/citations.duckdb ./data/citations.duckdb

# Bake BGE-M3 into the image so the first request doesn't wait on a 2.3 GB
# download. Cached in HF_HOME; FlagEmbedding picks it up from the hub cache.
RUN python -c "from huggingface_hub import snapshot_download; snapshot_download('BAAI/bge-m3')"

EXPOSE 7860
CMD ["uvicorn", "scholar_rag.api.main:app", "--host", "0.0.0.0", "--port", "7860"]
