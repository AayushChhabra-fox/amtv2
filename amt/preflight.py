"""Smoke-check a candidate before any Databricks submission."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np


def _log_loss(labels: np.ndarray, scores: np.ndarray) -> float:
    clipped = np.clip(scores, 1e-6, 1 - 1e-6)
    return float(-np.mean(labels * np.log(clipped) + (1 - labels) * np.log(1 - clipped)))


def smoke_candidate(candidate_dir: Path) -> dict[str, float]:
    """Import model.py, fit a separable toy set, and require the loss to fall."""
    model_path = candidate_dir / "model.py"
    if not model_path.is_file():
        raise FileNotFoundError(model_path)
    spec = importlib.util.spec_from_file_location(f"amt_candidate_{candidate_dir.name}", model_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot import {model_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    ranker = module.LogisticRanker()
    rng = np.random.default_rng(0)
    features = rng.normal(size=(80, 3))
    labels = (features[:, 0] + 0.5 * features[:, 1] > 0).astype(float)
    frame = {"X": features, "y": labels}
    ranker.fit(frame, frame, {"seed": 0})
    scores = np.asarray(ranker.score(frame), dtype=float)
    if scores.shape != labels.shape or not np.isfinite(scores).all():
        raise RuntimeError("smoke scores are missing or non-finite")
    baseline = _log_loss(labels, np.full_like(labels, 0.5))
    trained = _log_loss(labels, scores)
    if trained >= baseline:
        raise RuntimeError(f"smoke loss did not decrease: {trained:.4f} >= {baseline:.4f}")
    return {"log_loss": trained, "baseline_log_loss": baseline}
