"""Graded NDCG@10, click NDCG@10, MRR, coverage@10, and a day-blocked bootstrap.

Gains are ``2**relevance - 1`` with a ``log2(rank + 1)`` discount. Click NDCG
binarizes every positive label. The bootstrap resamples days, then queries
inside each sampled day, and reports the per-query paired delta.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from harness.interface import RankingFrame

PRIMARY_METRIC = "ndcg@10_graded"


def graded_ndcg_at_k(frame: RankingFrame, scores: np.ndarray, k: int = 10) -> float:
    return _mean(_per_query_ndcg(frame, scores, k, binary=False))


def click_ndcg_at_k(frame: RankingFrame, scores: np.ndarray, k: int = 10) -> float:
    return _mean(_per_query_ndcg(frame, scores, k, binary=True))


def mrr(frame: RankingFrame, scores: np.ndarray) -> float:
    _check_scores(frame, scores)
    labels = [float(value) for value in frame.column("label")]
    items = [str(value) for value in frame.column("item_id")]
    total = 0.0
    groups = frame.groups()
    if not groups:
        return float("nan")
    for start, end, _query_id in groups:
        order = _order(start, end, scores, items)
        reciprocal = 0.0
        for rank, index in enumerate(order, start=1):
            if labels[index] > 0:
                reciprocal = 1.0 / rank
                break
        total += reciprocal
    return total / len(groups)


def coverage_at_k(frame: RankingFrame, scores: np.ndarray, k: int = 10) -> float:
    """Share of eval items that appear in at least one top-k list."""

    _check_scores(frame, scores)
    items = [str(value) for value in frame.column("item_id")]
    catalog = set(items)
    if not catalog:
        return float("nan")
    recommended: set[str] = set()
    for start, end, _query_id in frame.groups():
        for index in _order(start, end, scores, items)[:k]:
            recommended.add(items[index])
    return len(recommended) / len(catalog)


def metrics(frame: RankingFrame, scores: np.ndarray, k: int = 10) -> dict[str, float]:
    graded_key = f"ndcg@{k}_graded"
    click_key = f"ndcg@{k}_click"
    coverage_key = f"coverage@{k}"
    return {
        graded_key: graded_ndcg_at_k(frame, scores, k),
        click_key: click_ndcg_at_k(frame, scores, k),
        "mrr": mrr(frame, scores),
        coverage_key: coverage_at_k(frame, scores, k),
    }


def paired_day_blocked_delta(
    frame: RankingFrame,
    scores: np.ndarray,
    baseline_scores: np.ndarray,
    *,
    k: int = 10,
    n_resamples: int = 1000,
    seed: int = 0,
) -> dict[str, float]:
    """Paired per-query graded-NDCG delta with a day-blocked bootstrap CI."""

    candidate = _per_query_ndcg(frame, scores, k, binary=False)
    baseline = _per_query_ndcg(frame, baseline_scores, k, binary=False)
    days = _query_days(frame)
    shared = [query_id for query_id in candidate if query_id in baseline]
    if not shared:
        return {"point": float("nan"), "ci_lo": float("nan"), "ci_hi": float("nan")}
    by_day: dict[Any, list[float]] = {}
    point_values: list[float] = []
    for query_id in shared:
        delta = candidate[query_id] - baseline[query_id]
        point_values.append(delta)
        by_day.setdefault(days[query_id], []).append(delta)
    day_keys = sorted(by_day, key=str)
    rng = np.random.default_rng(seed)
    stats = np.empty(n_resamples, dtype=float)
    for draw in range(n_resamples):
        chosen_days = rng.choice(len(day_keys), size=len(day_keys), replace=True)
        sample: list[float] = []
        for day_index in chosen_days:
            values = by_day[day_keys[int(day_index)]]
            picks = rng.choice(len(values), size=len(values), replace=True)
            sample.extend(values[int(index)] for index in picks)
        stats[draw] = float(np.mean(sample))
    low, high = np.quantile(stats, [0.025, 0.975])
    return {
        "point": float(np.mean(point_values)),
        "ci_lo": float(low),
        "ci_hi": float(high),
    }


def _per_query_ndcg(
    frame: RankingFrame,
    scores: np.ndarray,
    k: int,
    *,
    binary: bool,
) -> dict[Any, float]:
    _check_scores(frame, scores)
    labels = [float(value) for value in frame.column("label")]
    items = [str(value) for value in frame.column("item_id")]
    per_query: dict[Any, float] = {}
    for start, end, query_id in frame.groups():
        order = _order(start, end, scores, items)
        relevances = []
        for index in order:
            relevance = labels[index]
            if binary:
                relevance = 1.0 if relevance > 0 else 0.0
            relevances.append(relevance)
        ideal = _dcg(sorted(relevances, reverse=True), k)
        if ideal <= 0:
            continue
        per_query[query_id] = _dcg(relevances, k) / ideal
    return per_query


def _dcg(relevances: list[float], k: int) -> float:
    total = 0.0
    for rank, relevance in enumerate(relevances[:k], start=1):
        total += (2.0**relevance - 1.0) / math.log2(rank + 1)
    return total


def _order(start: int, end: int, scores: np.ndarray, items: list[str]) -> list[int]:
    return sorted(range(start, end), key=lambda index: (-float(scores[index]), items[index]))


def _query_days(frame: RankingFrame) -> dict[Any, Any]:
    return {query_id: frame.rows[start]["event_day"] for start, _end, query_id in frame.groups()}


def _mean(values: dict[Any, float]) -> float:
    if not values:
        return float("nan")
    return float(sum(values.values()) / len(values))


def _check_scores(frame: RankingFrame, scores: np.ndarray) -> np.ndarray:
    array = np.asarray(scores, dtype=float)
    if array.shape != (len(frame),):
        raise ValueError(f"scores length {array.shape} does not match frame {len(frame)}")
    if not np.isfinite(array).all():
        raise ValueError("scores must be finite")
    return array
