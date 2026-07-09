"""Centralised config loading — merges default.yaml with an optional experiment override."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv

load_dotenv()

# config.py lives at <project>/src/scholar_rag/config.py, so the project root
# (which holds configs/) is three names up: scholar_rag → src → <project>.
_ROOT = Path(__file__).resolve().parents[2]  # scholar-rag/ project root


def resolve_device(device: str) -> str:
    """
    Map a requested torch device to one that actually exists on this machine,
    so a config of ``mps`` (Apple GPU) or ``cuda`` degrades to ``cpu`` elsewhere
    instead of crashing at model load.
    """
    device = (device or "cpu").strip().lower()
    if device in ("mps", "cuda"):
        try:
            import torch
            if device == "mps" and not torch.backends.mps.is_available():
                return "cpu"
            if device == "cuda" and not torch.cuda.is_available():
                return "cpu"
        except Exception:
            return "cpu"
    return device


def _expand_env(obj: Any) -> Any:
    """Recursively expand ${VAR} placeholders using environment variables."""
    if isinstance(obj, str):
        import re
        return re.sub(r"\$\{(\w+)\}", lambda m: os.environ.get(m.group(1), m.group(0)), obj)
    if isinstance(obj, dict):
        return {k: _expand_env(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_expand_env(i) for i in obj]
    return obj


def load_config(experiment: str | None = None) -> dict[str, Any]:
    base_path = _ROOT / "configs" / "default.yaml"
    with base_path.open() as f:
        cfg: dict = yaml.safe_load(f)

    if experiment:
        exp_path = _ROOT / "configs" / "experiments" / f"{experiment}.yaml"
        with exp_path.open() as f:
            override = yaml.safe_load(f) or {}
        cfg = _deep_merge(cfg, override)

    return _expand_env(cfg)


def _deep_merge(base: dict, override: dict) -> dict:
    merged = base.copy()
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(merged.get(k), dict):
            merged[k] = _deep_merge(merged[k], v)
        else:
            merged[k] = v
    return merged
