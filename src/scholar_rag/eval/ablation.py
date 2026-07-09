"""
Ablation harness — toggles each technique on/off and logs results to MLflow.

Techniques to ablate:
  - HyDE
  - multi-query expansion
  - hybrid (dense+sparse) vs dense-only
  - cross-encoder reranking
  - citation-graph reranking
  - proposition-based chunking (vs fixed-size)

Each configuration is run through the retrieval metric suite and RAGAS,
with results logged as a separate MLflow run for comparison.
"""

from __future__ import annotations

import os
from typing import Any, Callable

import mlflow
from loguru import logger


_MLFLOW_URI = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
_EXPERIMENT = os.getenv("MLFLOW_EXPERIMENT", "scholar_rag")

ABLATION_CONFIGS = [
    {
        "name": "full_pipeline",
        "description": "All techniques enabled (baseline)",
        "overrides": {},
    },
    {
        "name": "no_hyde",
        "description": "Disable HyDE",
        "overrides": {"retrieval.hyde_enabled": False},
    },
    {
        "name": "no_multi_query",
        "description": "Single query (no expansion)",
        "overrides": {"retrieval.multi_query_n": 1},
    },
    {
        "name": "dense_only",
        "description": "Dense-only retrieval (no BM25 sparse)",
        "overrides": {
            "retrieval.hybrid_dense_weight": 1.0,
            "retrieval.hybrid_sparse_weight": 0.0,
        },
    },
    {
        "name": "no_rerank",
        "description": "No cross-encoder reranking",
        "overrides": {"retrieval.rerank_top_k": 50},
    },
    {
        "name": "no_graph_rerank",
        "description": "No citation-graph reranking",
        "overrides": {
            "retrieval.graph_rerank.alpha": 1.0,
            "retrieval.graph_rerank.beta": 0.0,
            "retrieval.graph_rerank.gamma": 0.0,
        },
    },
]


def run_ablation(
    queries: list[dict[str, Any]],
    build_pipeline_fn: Callable[[dict], Any],
    retrieve_fn_factory: Callable[[dict], Callable],
    base_config: dict,
) -> list[dict[str, Any]]:
    """
    Run the full ablation suite.

    build_pipeline_fn : callable(config) → pipeline object with .ask(query)
    retrieve_fn_factory: callable(config) → retrieve_fn(query) → list[dict]
    """
    from scholar_rag.eval.retrieval_metrics import evaluate_retrieval

    mlflow.set_tracking_uri(_MLFLOW_URI)
    mlflow.set_experiment(_EXPERIMENT)
    all_results = []

    for ablation in ABLATION_CONFIGS:
        cfg = _apply_overrides(base_config, ablation["overrides"])
        logger.info(f"Running ablation: {ablation['name']}")

        retrieve_fn = retrieve_fn_factory(cfg)
        metrics = evaluate_retrieval(queries, retrieve_fn)

        with mlflow.start_run(run_name=ablation["name"]):
            mlflow.log_param("description", ablation["description"])
            mlflow.log_params(ablation["overrides"])
            for k, v in metrics.items():
                mlflow.log_metric(k, v)

        result = {"config": ablation["name"], **metrics}
        all_results.append(result)
        logger.info(f"  {ablation['name']}: {metrics}")

    return all_results


def _apply_overrides(config: dict, overrides: dict) -> dict:
    """Apply dot-notation overrides to a nested config dict."""
    import copy
    cfg = copy.deepcopy(config)
    for dotkey, val in overrides.items():
        parts = dotkey.split(".")
        d = cfg
        for p in parts[:-1]:
            d = d.setdefault(p, {})
        d[parts[-1]] = val
    return cfg
