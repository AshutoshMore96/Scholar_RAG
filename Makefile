.PHONY: help up down ingest embed serve eval test lint fmt

PYTHON := python -m

help:
	@echo "ScholarRAG — available targets:"
	@echo "  up        Start all Docker services (Qdrant, Ollama, GROBID, MLflow)"
	@echo "  down      Stop and remove containers"
	@echo "  ingest    Run the full ingestion pipeline (crawl → enrich → parse → chunk → embed → store)"
	@echo "  embed     Re-embed chunks only (useful after model switch)"
	@echo "  serve     Start FastAPI + Streamlit"
	@echo "  eval      Run RAGAS + retrieval metrics + ablations"
	@echo "  test      Run unit tests"
	@echo "  lint      Ruff + mypy"
	@echo "  fmt       Ruff auto-format"

up:
	docker compose up -d
	@echo "Waiting for services to be ready…"
	@sleep 10
	docker compose ps

down:
	docker compose down

ingest:
	$(PYTHON) scholar_rag.api.main &
	@echo "Triggering ingestion via API…"
	curl -s -X POST http://localhost:8000/ingest \
	  -H "Content-Type: application/json" \
	  -d '{"categories": ["cs.CL", "cs.IR", "cs.LG"], "max_results": 200}' | python -m json.tool

embed:
	@echo "Re-embedding is handled within the ingestion pipeline."
	@echo "Run 'make ingest' to trigger a full re-index."

serve:   ## Start the API + web UI at http://localhost:8000/
	@echo "  Web UI:   http://localhost:8000/"
	@echo "  API docs: http://localhost:8000/docs"
	uvicorn scholar_rag.api.main:app --host 0.0.0.0 --port 8000 --reload

eval:   ## RAG evaluation — RAGAS metrics (Groq judge) on the golden set. Stop `serve` first.
	$(PYTHON) scholar_rag.eval.run_eval

test:
	$(PYTHON) pytest tests/ -v --tb=short

lint:
	$(PYTHON) ruff check src/ tests/
	$(PYTHON) mypy src/

fmt:
	$(PYTHON) ruff format src/ tests/

pull-models:
	ollama pull llama3.1:8b
	ollama pull nomic-embed-text
	@echo "Models pulled."

create-collection:
	$(PYTHON) -c "from scholar_rag.store.qdrant_store import QdrantStore; QdrantStore().create_collection()"
	@echo "Qdrant collection created."
