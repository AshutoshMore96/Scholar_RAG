"""
RAG evaluation — the RAGAS metric suite, implemented natively against our own
Groq LLM judge (no `ragas`/`langchain` dependency).

Why native? The `ragas` package has an unresolvable dependency conflict in this
environment (it unconditionally imports a `langchain_community.chat_models.vertexai`
module that current LangChain no longer ships). The metrics themselves are just
well-specified LLM-judge procedures, so we reproduce them directly — which is
lighter, reproducible, and consistent with the rest of the pipeline (Groq + BGE-M3).

Metrics
-------
  * **faithfulness**      — fraction of the answer's atomic claims that are
                            inferable from the retrieved context (hallucination check).
  * **answer_relevancy**  — mean cosine similarity between the question and
                            questions reverse-generated from the answer (BGE-M3).
  * **context_precision** — average-precision of the retrieved contexts, judged
                            useful for answering (rewards relevant-and-early ranking).
  * **context_recall**    — fraction of the reference answer's claims attributable
                            to the retrieved context (needs a ground-truth answer).

Each LLM step asks for JSON and is parsed leniently, with retry/back-off on Groq
rate limits. Reference-free metrics (faithfulness, answer_relevancy) run without a
ground truth; the two context metrics use it when present.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any, Callable

import numpy as np
from loguru import logger

from scholar_rag.generate.llm import LLMClient


# ── Judge LLM (Groq) ──────────────────────────────────────────────────────────
def build_judge() -> LLMClient:
    """A Groq-backed judge (reuses the deep-mode key). Falls back to local Ollama."""
    api_key = os.getenv("DEEP_LLM_API_KEY", "").strip() or None
    if api_key:
        base = os.getenv("DEEP_LLM_BASE_URL", "").strip() or "https://api.groq.com/openai/v1"
        model = os.getenv("EVAL_LLM_MODEL", "").strip() or "llama-3.3-70b-versatile"
        return LLMClient(model=model, base_url=base, api_key=api_key, timeout=120.0)
    return LLMClient(model=os.getenv("OLLAMA_MODEL", "llama3.1:8b"),
                     base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
                     timeout=120.0)


def _extract_json(text: str):
    """Pull the first balanced JSON array/object out of an LLM response."""
    for op, cl in (("[", "]"), ("{", "}")):
        i = text.find(op)
        if i < 0:
            continue
        depth = 0
        for j in range(i, len(text)):
            if text[j] == op:
                depth += 1
            elif text[j] == cl:
                depth -= 1
                if depth == 0:
                    try:
                        return json.loads(text[i:j + 1])
                    except json.JSONDecodeError:
                        break
    return None


def _judge(llm: LLMClient, system: str, user: str, max_tokens: int = 900, retries: int = 4):
    """One JSON-returning judge call, with back-off on rate limits."""
    last = ""
    for attempt in range(retries):
        try:
            out = llm.chat(
                [{"role": "system", "content": system}, {"role": "user", "content": user}],
                temperature=0.0, max_tokens=max_tokens,
            )
            parsed = _extract_json(out or "")
            if parsed is not None:
                return parsed
        except Exception as exc:  # noqa: BLE001
            last = str(exc)
            msg = last.lower()
            if "429" in msg or "rate" in msg or "quota" in msg:
                if "per day" in msg or "tpd" in msg:
                    # Free-tier daily token cap — no point retrying this run.
                    logger.error(
                        "Judge model hit its daily token limit (TPD). Set "
                        "EVAL_LLM_MODEL=llama-3.1-8b-instant (higher free limit) or "
                        "wait for the daily reset.")
                    raise SystemExit(2)
                time.sleep(4 * (attempt + 1))
                continue
            time.sleep(1)
    if last:
        logger.debug(f"judge gave up after {retries} tries: {last[:120]}")
    return None


# ── Metrics ───────────────────────────────────────────────────────────────────
def faithfulness(llm: LLMClient, answer: str, contexts: list[str]) -> float | None:
    if not answer.strip() or not contexts:
        return None
    ctx = "\n\n".join(contexts)[:6000]
    claims = _judge(
        llm, "You decompose text into atomic factual claims.",
        "Break the ANSWER into a list of standalone factual claims. Ignore inline "
        "citation markers like [2312.10997v1]. Output ONLY a JSON array of strings.\n\n"
        f"ANSWER:\n{answer}",
    )
    claims = [c for c in (claims or []) if isinstance(c, str) and c.strip()][:20]
    if not claims:
        return None
    verdicts = _judge(
        llm, "You verify whether statements are supported by a context.",
        f"CONTEXT:\n{ctx}\n\nFor each STATEMENT, decide if it can be directly inferred "
        'from the context. Output ONLY a JSON array of objects {"verdict": 0 or 1}.\n\n'
        "STATEMENTS:\n" + json.dumps(claims),
    )
    vs = [int(v.get("verdict", 0)) for v in (verdicts or []) if isinstance(v, dict)]
    return round(sum(vs) / len(vs), 3) if vs else None


def answer_relevancy(llm: LLMClient, embedder, question: str, answer: str,
                     n: int = 3) -> float | None:
    if not answer.strip():
        return None
    gen = _judge(
        llm, "You write questions that a given answer would address.",
        f"Generate {n} distinct questions that the ANSWER below would directly and "
        "completely address. Output ONLY a JSON array of strings.\n\nANSWER:\n" + answer,
    )
    gen = [q for q in (gen or []) if isinstance(q, str) and q.strip()][:n]
    if not gen:
        return None
    embs = embedder.embed_queries([question] + gen)
    qv = np.asarray(embs[0].dense[0], dtype=float)
    qv /= np.linalg.norm(qv) + 1e-9
    sims = []
    for e in embs[1:]:
        v = np.asarray(e.dense[0], dtype=float)
        v /= np.linalg.norm(v) + 1e-9
        sims.append(float(qv @ v))
    return round(sum(sims) / len(sims), 3) if sims else None


def context_precision(llm: LLMClient, question: str, contexts: list[str],
                      reference: str | None = None) -> float | None:
    if not contexts:
        return None
    payload = [{"i": i, "context": c[:800]} for i, c in enumerate(contexts)]
    res = _judge(
        llm, "You judge whether a context is useful for answering a question.",
        f"QUESTION: {question}\nREFERENCE ANSWER: {reference or '(none provided)'}\n\n"
        "For each CONTEXT, decide if it is USEFUL for answering the question / "
        'supporting the reference answer (1) or not (0). Output ONLY a JSON array of '
        'objects {"i": index, "verdict": 0 or 1}.\n\nCONTEXTS:\n' + json.dumps(payload),
    )
    if not res:
        return None
    vmap = {r["i"]: int(r.get("verdict", 0)) for r in res if isinstance(r, dict) and "i" in r}
    rel = [vmap.get(i, 0) for i in range(len(contexts))]
    # Average precision: mean of precision@k over the ranks that are relevant.
    hits, running = 0, 0.0
    for k, r in enumerate(rel):
        if r:
            hits += 1
            running += hits / (k + 1)
    total = sum(rel)
    return round(running / total, 3) if total else 0.0


def context_recall(llm: LLMClient, reference: str | None, contexts: list[str]) -> float | None:
    if not reference or not reference.strip() or not contexts:
        return None
    ctx = "\n\n".join(contexts)[:6000]
    claims = _judge(
        llm, "You decompose text into atomic factual claims.",
        "Break the REFERENCE answer into standalone factual claims. Output ONLY a JSON "
        f"array of strings.\n\nREFERENCE:\n{reference}",
    )
    claims = [c for c in (claims or []) if isinstance(c, str) and c.strip()][:15]
    if not claims:
        return None
    verdicts = _judge(
        llm, "You check whether claims are supported by a context.",
        f"CONTEXT:\n{ctx}\n\nFor each CLAIM, decide if it can be attributed to "
        '(found or supported in) the context. Output ONLY a JSON array of objects '
        '{"verdict": 0 or 1}.\n\nCLAIMS:\n' + json.dumps(claims),
    )
    vs = [int(v.get("verdict", 0)) for v in (verdicts or []) if isinstance(v, dict)]
    return round(sum(vs) / len(vs), 3) if vs else None


# ── Orchestration ─────────────────────────────────────────────────────────────
def run_evaluation(
    qa_pairs: list[dict[str, Any]],
    pipeline_fn: Callable[[str], Any],
    llm: LLMClient | None = None,
    embedder: Any = None,
    out_dir: str | os.PathLike = "data/eval",
    run_name: str = "ragas_native",
) -> dict[str, Any]:
    """
    Evaluate `pipeline_fn` on QA pairs. Each pair: {question, ground_truth?}.
    `pipeline_fn(question)` must return an object with `.review` and
    `.source_passages`. Writes a per-question CSV + summary JSON to `out_dir`,
    logs to MLflow only if it is reachable, and returns the aggregate.
    """
    llm = llm or build_judge()
    if embedder is None:
        raise ValueError("run_evaluation needs an `embedder` for answer_relevancy.")

    rows: list[dict[str, Any]] = []
    for idx, qa in enumerate(qa_pairs, 1):
        q = qa["question"]
        ref = qa.get("ground_truth") or qa.get("reference")
        logger.info(f"[{idx}/{len(qa_pairs)}] evaluating: {q[:70]}")
        result = pipeline_fn(q)
        answer = getattr(result, "review", "") or ""
        passages = getattr(result, "source_passages", []) or []
        contexts = [p.get("text") or p.get("parent_text") or "" for p in passages[:6]]
        contexts = [c for c in contexts if c.strip()]
        row = {
            "question": q,
            "faithfulness": faithfulness(llm, answer, contexts),
            "answer_relevancy": answer_relevancy(llm, embedder, q, answer),
            "context_precision": context_precision(llm, q, contexts, ref),
            "context_recall": context_recall(llm, ref, contexts),
            "abstained": getattr(result, "abstained", False),
            "n_contexts": len(contexts),
        }
        rows.append(row)
        logger.info("  " + " · ".join(
            f"{k}={row[k]}" for k in
            ("faithfulness", "answer_relevancy", "context_precision", "context_recall")))

    metric_keys = ("faithfulness", "answer_relevancy", "context_precision", "context_recall")
    agg = {}
    for k in metric_keys:
        vals = [r[k] for r in rows if isinstance(r[k], (int, float))]
        agg[k] = round(sum(vals) / len(vals), 3) if vals else None

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "ragas_results.json").write_text(json.dumps(
        {"run": run_name, "aggregate": agg, "per_question": rows}, indent=2))
    _write_csv(out / "ragas_results.csv", rows, metric_keys)
    logger.success(f"RAGAS-native aggregate: {agg}")
    logger.info(f"Wrote {out/'ragas_results.json'} and .csv")

    _maybe_log_mlflow(run_name, agg, rows)
    return {"aggregate": agg, "per_question": rows}


def _write_csv(path: Path, rows: list[dict], metric_keys) -> None:
    import csv
    cols = ["question", *metric_keys, "abstained", "n_contexts"]
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow({c: r.get(c) for c in cols})


def _maybe_log_mlflow(run_name: str, agg: dict, rows: list[dict]) -> None:
    """Log to MLflow only if the tracking server is actually reachable."""
    uri = os.getenv("MLFLOW_TRACKING_URI", "").strip()
    if not uri:
        return
    try:
        import mlflow
        mlflow.set_tracking_uri(uri)
        mlflow.set_experiment(os.getenv("MLFLOW_EXPERIMENT", "scholar_rag"))
        with mlflow.start_run(run_name=run_name):
            for k, v in agg.items():
                if v is not None:
                    mlflow.log_metric(k, v)
        logger.info(f"Logged metrics to MLflow at {uri}")
    except Exception as exc:  # noqa: BLE001
        logger.warning(f"MLflow logging skipped ({exc}).")


def load_golden_set(path: Path) -> list[dict[str, Any]]:
    data = json.loads(Path(path).read_text())
    return data["questions"] if isinstance(data, dict) and "questions" in data else data
