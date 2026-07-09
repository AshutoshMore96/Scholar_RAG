"""
Deploy ScholarRAG (FastAPI app + web UI) on Modal — free tier, no card, no HF PRO.

Serves the exact same app as locally: BGE-M3 embedding runs in-process (baked into
the image), while reranking (Jina), generation (Groq) and vectors (Qdrant Cloud)
are external APIs. Scales to zero when idle ($0), spins up on demand.

Deploy:
    pip install modal            # once
    modal setup                  # once (browser auth)
    # create the secret from your .env (see deploy script), then:
    modal deploy modal/scholar_rag_modal.py
    # -> prints the public https://<workspace>--scholar-rag-web.modal.run URL
"""
import modal

app = modal.App("scholar-rag")

# ── Image: runtime deps + BGE-M3 baked in ─────────────────────────────────────
image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("build-essential", "git")
    # CPU-only torch first, so FlagEmbedding doesn't pull the ~2 GB CUDA build.
    .pip_install("torch", index_url="https://download.pytorch.org/whl/cpu")
    .pip_install(
        "FlagEmbedding>=1.2.0",
        "sentence-transformers>=3.0.0",
        "qdrant-client>=1.9.0",
        "fastapi>=0.111.0",
        "httpx",
        "duckdb>=0.10.0",
        "loguru",
        "pydantic>=2.0",
        "numpy",
        "PyYAML",
        "python-dotenv",
        "rank-bm25>=0.2.2",
        "huggingface_hub",
    )
    # Bake BGE-M3 into the image so cold starts don't wait on a 2.3 GB download.
    .run_commands(
        "python -c \"from huggingface_hub import snapshot_download; snapshot_download('BAAI/bge-m3')\""
    )
    .env({
        "PYTHONPATH": "/app/src",
        "HF_HUB_DISABLE_PROGRESS_BARS": "1",
        "TOKENIZERS_PARALLELISM": "false",
    })
    # App source + config + the 780 KB citation graph (NOT .env / data/raw / .venv).
    .add_local_dir("src", "/app/src")
    .add_local_dir("configs", "/app/configs")
    .add_local_file("data/citations.duckdb", "/app/data/citations.duckdb")
)


@app.function(
    image=image,
    secrets=[modal.Secret.from_name("scholar-rag-secrets")],
    cpu=2.0,
    memory=4096,
    scaledown_window=300,   # stay warm 5 min after the last request, then scale to 0
    min_containers=0,       # $0 when idle
    timeout=600,
)
@modal.concurrent(max_inputs=20)   # one warm container serves many requests
@modal.asgi_app()
def web():
    import os
    os.chdir("/app")   # so data/citations.duckdb and configs/ resolve
    from scholar_rag.api.main import app as fastapi_app
    return fastapi_app
