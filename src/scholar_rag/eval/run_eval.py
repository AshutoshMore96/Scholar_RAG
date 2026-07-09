"""
CLI: evaluate ScholarRAG on the golden set with the native RAGAS metric suite.

    python -m scholar_rag.eval.run_eval                 # full golden set, fast mode
    python -m scholar_rag.eval.run_eval --deep --limit 3

Judge = Groq (via DEEP_LLM_API_KEY); embeddings = BGE-M3. Results are written to
data/eval/ragas_results.{json,csv}. Run this with the API server stopped — it
needs exclusive access to the DuckDB citation graph.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

from scholar_rag.config import load_config
from scholar_rag.api.main import _build_pipeline
from scholar_rag.eval.ragas_eval import build_judge, load_golden_set, run_evaluation


def main() -> None:
    ap = argparse.ArgumentParser(description="ScholarRAG RAG evaluation (RAGAS metrics, Groq judge).")
    ap.add_argument("--golden", default="data/eval/golden_set.json")
    ap.add_argument("--top-k", type=int, default=6)
    ap.add_argument("--deep", action="store_true", help="use deep retrieval for each question")
    ap.add_argument("--limit", type=int, default=0, help="evaluate only the first N questions")
    args = ap.parse_args()

    cfg = load_config()
    engine, generator = _build_pipeline(cfg)

    def pipeline_fn(question: str):
        r = engine.retrieve(question, top_k=args.top_k, deep=args.deep)
        return generator.generate(question, r.passages, context_quality=r.context_quality)

    golden = load_golden_set(Path(args.golden))
    if args.limit:
        golden = golden[: args.limit]

    run_evaluation(
        golden, pipeline_fn,
        llm=build_judge(), embedder=engine.embedder,
        run_name="ragas_native_deep" if args.deep else "ragas_native_fast",
    )


if __name__ == "__main__":
    main()
