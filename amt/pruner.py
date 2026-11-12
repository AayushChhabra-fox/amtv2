"""Deterministic rung pruner.

``decide`` applies the section 5.3 rules to synthetic or real MLflow runs and
returns a ``RungDecision``. It compares candidates on the paired delta against
the same-rung LightGBM reference, and against every ``ref.*`` delta the runs
carry. Search never scores ``select_holdout`` or ``final_holdout``.

The CLI and the Databricks pruner job call ``decide`` and then append ledger
rows. The function itself only reads the run records it is given.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from amt.ledger import append, read_rows
from amt.schemas import (
    Brief,
    DeltaVsBaseline,
    Kind,
    LedgerRow,
    Operator,
    Rung,
    load_brief,
)

# Metric names the search pruner is allowed to read. Holdout keys are not here.
SEARCH_METRIC_KEYS: frozenset[str] = frozenset(
    {
        "ndcg@10_graded",
        "delta_point",
        "delta_ci_lo",
        "delta_ci_hi",
        "latency_p95_ms",
        "dbu",
        "wall_min",
    }
)
_HOLDOUT_SPLITS = frozenset({"select_holdout", "final_holdout"})
_LGBM_FAMILIES = frozenset({"lgbm_reference", "lgbm", "lightgbm"})
_SEARCH_RUNGS = frozenset({"R1", "R2", "R3"})
_NO_METRIC = "run produced no metric"


class HoldoutReadError(RuntimeError):
    """A search decision was pointed at a holdout split."""


class PrunerError(ValueError):
    """The rung cannot be decided from the runs that were passed in."""


class QueryDelta(BaseModel):
    """One query's paired graded-NDCG delta versus the same-rung baseline."""

    model_config = ConfigDict(extra="forbid")

    query_id: str
    event_day: str
    delta: float


class RunView(BaseModel):
    """One seed of one node, already scored on ``search_val``."""

    model_config = ConfigDict(extra="forbid")

    node_id: str = Field(min_length=1)
    family: str = Field(min_length=1)
    seed: int = Field(default=0, ge=0)
    rung: Rung = "R1"
    split: str = "search_val"
    kind: Kind = "model"
    operator: Operator = "draft"
    claim: str = ""
    parent_id: str | None = None
    round: int = Field(default=0, ge=0)
    metric: float | None = None
    delta: DeltaVsBaseline | None = None
    ref_deltas: dict[str, DeltaVsBaseline] = Field(default_factory=dict)
    query_deltas: list[QueryDelta] = Field(default_factory=list)
    latency_p95_ms: float | None = None
    dbu: float | None = None
    wall_min: float | None = None
    divergent: bool = False
    preflight: str | None = None
    review: str | None = None
    git_sha: str = "unknown"
    mlflow_run_id: str | None = None
    job_run_id: str | None = None


class NodeDecision(BaseModel):
    """The rule that fired for one node at the end of a rung."""

    model_config = ConfigDict(extra="forbid")

    node_id: str
    family: str
    status: str
    rule: str
    kill_reason: str | None = None
    delta: DeltaVsBaseline | None = None
    ref_deltas: dict[str, DeltaVsBaseline] = Field(default_factory=dict)
    seeds: list[int] = Field(default_factory=list)
    note: str | None = None


class RungDecision(BaseModel):
    """Survivors, promotions, and the reason for every node."""

    model_config = ConfigDict(extra="forbid")

    rung: Rung
    survivors: list[str]
    promoted: list[str]
    diversity_slot: str | None = None
    decisions: list[NodeDecision]


@dataclass
class _Summary:
    node_id: str
    family: str
    delta: DeltaVsBaseline | None
    ref_deltas: dict[str, DeltaVsBaseline]
    dbu: float | None
    latency_p95_ms: float | None
    query_deltas: list[QueryDelta]
    seeds: list[RunView]


def decide(
    rung: str,
    runs: Sequence[Any],
    baseline_runs: Sequence[Any],
    brief: Brief,
    *,
    n_resamples: int = 1000,
    seed: int = 0,
    shared_cluster: bool = False,
) -> RungDecision:
    """Apply hard kills, the rung floor, and from R2 dominance, halving, and promotion.

    ``rung`` is ``R0`` through ``R3``. ``runs`` are the candidate seeds.
    ``baseline_runs`` are the same-rung LightGBM reference and any ``ref.*``
    baselines. Both sequences must already be scored on ``search_val``.
    """

    if rung not in {"R0", "R1", "R2", "R3"}:
        raise PrunerError(f"unknown rung {rung}")
    _reject_holdouts(list(runs) + list(baseline_runs))
    loaded = _coerce_all(runs)
    candidates = [view for view in loaded if view.kind != "baseline"]
    baselines = [view for view in loaded if view.kind == "baseline"]
    baselines.extend(_coerce_all(baseline_runs))
    for view in candidates:
        if view.rung != rung:
            raise PrunerError(f"{view.node_id} is on {view.rung}, not {rung}")
    baselines = [view for view in baselines if view.rung == rung]
    if rung in _SEARCH_RUNGS and not _lgbm_baselines(baselines, rung):
        raise PrunerError(f"{rung} requires a same-rung LightGBM reference scored on search_val")

    grouped = _group(candidates)
    preliminary: dict[str, NodeDecision] = {}
    latency_suspect: set[str] = set()
    alive: list[_Summary] = []
    for node_id in sorted(grouped):
        seeds = grouped[node_id]
        summary = _summarize(seeds)
        hard = _hard_kill(seeds, summary, brief)
        if hard is not None and hard.rule == "latency" and shared_cluster:
            latency_suspect.add(node_id)
            hard = None
        if hard is not None:
            preliminary[node_id] = hard
            continue
        if rung in _SEARCH_RUNGS:
            missing = _missing_delta(summary)
            if missing is not None:
                preliminary[node_id] = missing
                continue
            floor = _floor_kill(summary, brief, rung)
            if floor is not None:
                preliminary[node_id] = floor
                continue
        alive.append(summary)

    dominated: dict[str, NodeDecision] = {}
    # R1 only drops hard failures and models that are clearly worse than the
    # same-rung baseline. A small sample should not halve or dominate.
    if rung in {"R2", "R3"}:
        dominated = _dominance_kills(alive, n_resamples=n_resamples, seed=seed)
        alive = [node for node in alive if node.node_id not in dominated]

    diversity_slot: str | None = None
    promoted: list[str] = []
    if rung in {"R2", "R3"}:
        kept, diversity_slot = _halve(alive, rung)
        if rung == "R3":
            promoted = _promote(kept)
    else:
        kept = sorted(alive, key=_rank_key)

    kept_ids = {node.node_id for node in kept}
    decisions: list[NodeDecision] = []
    for node_id in sorted(grouped):
        if node_id in preliminary:
            decisions.append(preliminary[node_id])
            continue
        if node_id in dominated:
            decisions.append(dominated[node_id])
            continue
        summary = next(node for node in alive if node.node_id == node_id)
        if node_id not in kept_ids:
            point = summary.delta.point if summary.delta is not None else float("nan")
            decisions.append(
                _node(
                    summary,
                    status="killed",
                    rule="halving",
                    kill_reason=f"halving: delta point {_num(point)} outside the kept half",
                )
            )
            continue
        if node_id in promoted:
            decisions.append(_node(summary, status="promoted", rule="promoted", kill_reason=None))
        elif node_id == diversity_slot:
            decisions.append(_node(summary, status="kept", rule="diversity", kill_reason=None))
        else:
            decisions.append(_node(summary, status="kept", rule="survive", kill_reason=None))

    for decision in decisions:
        if decision.node_id in latency_suspect and decision.status in {"kept", "promoted"}:
            decision.note = "latency_suspect"

    return RungDecision(
        rung=rung,  # type: ignore[arg-type]
        survivors=[node.node_id for node in kept],
        promoted=promoted,
        diversity_slot=diversity_slot,
        decisions=decisions,
    )


def commit_rung(
    root: Path,
    run_id: str,
    runs: Sequence[Any],
    decision: RungDecision,
) -> Path:
    """Write ``runs/<id>/rung_<r>_survivors.json`` and append one ledger row per seed."""

    path = root / "runs" / run_id / f"rung_{decision.rung}_survivors.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(decision.model_dump_json(indent=2) + "\n")
    by_node = {item.node_id: item for item in decision.decisions}
    existing = {
        (row.node_id, row.rung, row.seed)
        for row in read_rows(root)
        if row.run_id == run_id
    }
    for view in _coerce_all(runs):
        if view.kind == "baseline" or view.node_id not in by_node:
            continue
        if (view.node_id, decision.rung, view.seed) in existing:
            continue
        append(root, _ledger_from(run_id, view, by_node[view.node_id]))
    return path


def ledger_row(
    *,
    run_id: str,
    node_id: str,
    family: str,
    claim: str,
    rung: str,
    seed: int,
    metric: float | None,
    dbu: float | None,
    wall_min: float | None,
    git_sha: str,
    mlflow_run_id: str | None,
    job_run_id: str | None,
) -> LedgerRow:
    """Ledger row for the walking skeleton, which has a single absolute metric."""

    status, reason = _legacy_status(metric)
    return LedgerRow(
        run_id=run_id,
        node_id=node_id,
        round=0,
        operator="draft",
        family=family,
        claim=claim,
        rung=rung,  # type: ignore[arg-type]
        seed=seed,
        metric=metric if metric is not None and math.isfinite(metric) else None,
        dbu=dbu,
        wall_min=wall_min,
        status=status,  # type: ignore[arg-type]
        kill_reason=reason,
        git_sha=git_sha,
        mlflow_run_id=mlflow_run_id,
        job_run_id=job_run_id,
    )


def _legacy_status(metric: float | None) -> tuple[str, str | None]:
    if metric is None:
        return "failed", _NO_METRIC
    if not math.isfinite(metric):
        return "failed", "nan or divergent scores"
    return "kept", None


def _reject_holdouts(raws: Sequence[Any]) -> None:
    for raw in raws:
        node_id, split = _identity(raw)
        if split in _HOLDOUT_SPLITS:
            raise HoldoutReadError(
                f"{node_id or 'run'} is on {split}; the search pruner does not read holdout metrics"
            )


def _identity(raw: Any) -> tuple[str, str]:
    if isinstance(raw, RunView):
        return raw.node_id, raw.split
    if isinstance(raw, Mapping) and _is_mlflow(raw):
        tags = _tag_map(raw)
        return str(tags.get("node_id", "")), str(tags.get("split", "search_val"))
    if isinstance(raw, Mapping):
        return str(raw.get("node_id", "")), str(raw.get("split", "search_val"))
    return str(getattr(raw, "node_id", "")), str(getattr(raw, "split", "search_val"))


def _coerce_all(raws: Sequence[Any]) -> list[RunView]:
    return [coerce_run(raw) for raw in raws]


def coerce_run(raw: Any) -> RunView:
    """Normalize a ``RunView``, a plain record, or one MLflow run payload."""

    if isinstance(raw, RunView):
        _check_split(raw.split, raw.node_id)
        return raw
    if isinstance(raw, Mapping) and _is_mlflow(raw):
        return run_from_mlflow(raw)
    if isinstance(raw, Mapping):
        view = RunView.model_validate(dict(raw))
        _check_split(view.split, view.node_id)
        return view
    return _from_attributes(raw)


def run_from_mlflow(payload: Mapping[str, Any]) -> RunView:
    """Read one MLflow run, touching only search metric names."""

    body = payload["run"] if isinstance(payload.get("run"), Mapping) else payload
    if not isinstance(body, Mapping):
        raise PrunerError("MLflow payload has no run body")
    info = body.get("info") if isinstance(body.get("info"), Mapping) else {}
    tags = _tag_map(body)
    node_id = str(tags.get("node_id") or info.get("run_name") or "run")
    split = str(tags.get("split", "search_val"))
    _check_split(split, node_id)
    data = body.get("data") if isinstance(body.get("data"), Mapping) else body
    metrics = _metrics_mapping(data if isinstance(data, Mapping) else {})
    metric = _lookup(metrics, "ndcg@10_graded")
    delta = _delta_from_metrics(metrics)
    ref_deltas = _ref_deltas(metrics, str(tags.get("ref_names", "")))
    # Per-query deltas ride beside the MLflow body, not in the metric map.
    query_source = body.get("query_deltas") or []
    query_deltas = [QueryDelta.model_validate(item) for item in query_source]
    return RunView(
        node_id=node_id,
        family=str(tags.get("family") or "candidate"),
        seed=int(tags.get("seed") or 0),
        rung=str(tags.get("rung") or "R1"),  # type: ignore[arg-type]
        split=split,
        kind=str(tags.get("kind") or "model"),  # type: ignore[arg-type]
        operator=str(tags.get("operator") or "draft"),  # type: ignore[arg-type]
        claim=str(tags.get("claim") or node_id),
        parent_id=tags.get("parent_id") or None,
        round=int(tags.get("round") or 0),
        metric=None if metric is None else float(metric),
        delta=delta,
        ref_deltas=ref_deltas,
        query_deltas=query_deltas,
        latency_p95_ms=_optional_float(_lookup(metrics, "latency_p95_ms")),
        dbu=_optional_float(_lookup(metrics, "dbu")),
        wall_min=_optional_float(_lookup(metrics, "wall_min")),
        divergent=str(tags.get("divergent", "")).lower() == "true",
        preflight=tags.get("preflight"),
        review=tags.get("review"),
        git_sha=str(tags.get("git_sha") or "unknown"),
        mlflow_run_id=str(info.get("run_id") or tags.get("mlflow_run_id") or "") or None,
        job_run_id=tags.get("job_run_id"),
    )


def _check_split(split: str, node_id: str) -> None:
    if split in _HOLDOUT_SPLITS:
        raise HoldoutReadError(
            f"{node_id} is on {split}; the search pruner does not read holdout metrics"
        )
    if split != "search_val":
        raise PrunerError(f"{node_id} split {split} is not search_val")


def _is_mlflow(payload: Mapping[str, Any]) -> bool:
    if isinstance(payload.get("run"), Mapping):
        return True
    return isinstance(payload.get("info"), Mapping) and isinstance(payload.get("data"), Mapping)


def _tag_map(payload: Mapping[str, Any]) -> dict[str, str]:
    body = payload["run"] if isinstance(payload.get("run"), Mapping) else payload
    nested = body.get("data") if isinstance(body, Mapping) else None
    data = nested if isinstance(nested, Mapping) else body
    raw = data.get("tags") if isinstance(data, Mapping) else None
    if raw is None and isinstance(body, Mapping):
        raw = body.get("tags")
    if raw is None:
        return {}
    if isinstance(raw, Mapping):
        return {str(key): str(value) for key, value in raw.items()}
    tags: dict[str, str] = {}
    for item in raw:
        tags[str(item["key"])] = str(item["value"])
    return tags


def _metrics_mapping(data: Mapping[str, Any]) -> Mapping[str, Any]:
    metrics = data.get("metrics")
    if metrics is None:
        return {}
    if isinstance(metrics, Mapping):
        return metrics
    if isinstance(metrics, list):
        found: dict[str, Any] = {}
        for item in metrics:
            key = str(item["key"])
            if "holdout" in key:
                continue
            if key in SEARCH_METRIC_KEYS or key.startswith("ref__"):
                found[key] = item["value"]
        return found
    raise PrunerError("MLflow metrics must be a mapping or a list of key/value rows")


def _lookup(metrics: Mapping[str, Any], key: str) -> Any:
    if "holdout" in key:
        raise HoldoutReadError(f"refusing to read metric {key}")
    if key not in metrics:
        return None
    return metrics[key]


def _delta_from_metrics(metrics: Mapping[str, Any]) -> DeltaVsBaseline | None:
    point = _lookup(metrics, "delta_point")
    if point is None:
        return None
    low = _lookup(metrics, "delta_ci_lo")
    high = _lookup(metrics, "delta_ci_hi")
    if low is None or high is None:
        raise PrunerError("delta_point requires delta_ci_lo and delta_ci_hi")
    return DeltaVsBaseline(point=float(point), ci_lo=float(low), ci_hi=float(high))


def _ref_deltas(metrics: Mapping[str, Any], ref_names: str) -> dict[str, DeltaVsBaseline]:
    found: dict[str, DeltaVsBaseline] = {}
    for name in [part.strip() for part in str(ref_names).split(",") if part.strip()]:
        point = _lookup(metrics, f"ref__{name}__point")
        if point is None:
            continue
        low = _lookup(metrics, f"ref__{name}__ci_lo")
        high = _lookup(metrics, f"ref__{name}__ci_hi")
        if low is None or high is None:
            raise PrunerError(f"ref {name} delta is missing a confidence bound")
        found[name] = DeltaVsBaseline(point=float(point), ci_lo=float(low), ci_hi=float(high))
    return found


def _optional_float(value: Any) -> float | None:
    if value is None or value == "":
        return None
    return float(value)


def _from_attributes(raw: Any) -> RunView:
    node_id = str(getattr(raw, "node_id", "") or "run")
    split = str(getattr(raw, "split", "search_val"))
    _check_split(split, node_id)
    payload = {
        "node_id": node_id,
        "family": getattr(raw, "family", "candidate"),
        "seed": getattr(raw, "seed", 0),
        "rung": getattr(raw, "rung", "R1"),
        "split": split,
        "kind": getattr(raw, "kind", "model"),
        "operator": getattr(raw, "operator", "draft"),
        "claim": getattr(raw, "claim", "") or node_id,
        "parent_id": getattr(raw, "parent_id", None),
        "round": getattr(raw, "round", 0),
        "metric": getattr(raw, "metric", None),
        "delta": getattr(raw, "delta", None),
        "ref_deltas": getattr(raw, "ref_deltas", {}) or {},
        "query_deltas": getattr(raw, "query_deltas", []) or [],
        "latency_p95_ms": getattr(raw, "latency_p95_ms", None),
        "dbu": getattr(raw, "dbu", None),
        "wall_min": getattr(raw, "wall_min", None),
        "divergent": bool(getattr(raw, "divergent", False)),
        "preflight": getattr(raw, "preflight", None),
        "review": getattr(raw, "review", None),
        "git_sha": getattr(raw, "git_sha", "unknown"),
        "mlflow_run_id": getattr(raw, "mlflow_run_id", None),
        "job_run_id": getattr(raw, "job_run_id", None),
    }
    return RunView.model_validate(payload)


def _group(runs: Sequence[RunView]) -> dict[str, list[RunView]]:
    grouped: dict[str, list[RunView]] = {}
    for run in runs:
        grouped.setdefault(run.node_id, []).append(run)
    for seeds in grouped.values():
        seeds.sort(key=lambda run: run.seed)
    return grouped


def _lgbm_baselines(baselines: Sequence[RunView], rung: str) -> list[RunView]:
    return [
        run
        for run in baselines
        if run.rung == rung and run.split == "search_val" and _is_lgbm(run)
    ]


def _is_lgbm(run: RunView) -> bool:
    return run.family in _LGBM_FAMILIES or run.node_id in _LGBM_FAMILIES


def _summarize(seeds: Sequence[RunView]) -> _Summary:
    deltas = [seed.delta for seed in seeds if seed.delta is not None]
    delta = None
    if deltas:
        delta = DeltaVsBaseline(
            point=sum(item.point for item in deltas) / len(deltas),
            ci_lo=sum(item.ci_lo for item in deltas) / len(deltas),
            ci_hi=sum(item.ci_hi for item in deltas) / len(deltas),
        )
    names: set[str] = set()
    for seed in seeds:
        names.update(seed.ref_deltas)
    refs: dict[str, DeltaVsBaseline] = {}
    for name in sorted(names):
        parts = [seed.ref_deltas[name] for seed in seeds if name in seed.ref_deltas]
        if len(parts) != len(seeds):
            continue
        refs[name] = DeltaVsBaseline(
            point=sum(item.point for item in parts) / len(parts),
            ci_lo=sum(item.ci_lo for item in parts) / len(parts),
            ci_hi=sum(item.ci_hi for item in parts) / len(parts),
        )
    dbus = [seed.dbu for seed in seeds if seed.dbu is not None]
    latencies = [seed.latency_p95_ms for seed in seeds if seed.latency_p95_ms is not None]
    return _Summary(
        node_id=seeds[0].node_id,
        family=seeds[0].family,
        delta=delta,
        ref_deltas=refs,
        dbu=sum(dbus) if dbus else None,
        latency_p95_ms=max(latencies) if latencies else None,
        query_deltas=_average_query_deltas(seeds),
        seeds=list(seeds),
    )


def _average_query_deltas(seeds: Sequence[RunView]) -> list[QueryDelta]:
    buckets: dict[str, list[QueryDelta]] = {}
    for seed in seeds:
        for item in seed.query_deltas:
            buckets.setdefault(item.query_id, []).append(item)
    averaged: list[QueryDelta] = []
    for query_id in sorted(buckets):
        items = buckets[query_id]
        averaged.append(
            QueryDelta(
                query_id=query_id,
                event_day=items[0].event_day,
                delta=sum(item.delta for item in items) / len(items),
            )
        )
    return averaged


def _hard_kill(seeds: Sequence[RunView], summary: _Summary, brief: Brief) -> NodeDecision | None:
    if any(str(seed.preflight or "").upper() == "FAIL" for seed in seeds):
        return _node(summary, status="failed", rule="preflight", kill_reason="preflight FAIL")
    if any(str(seed.review or "").upper() == "FAIL" for seed in seeds):
        return _node(summary, status="failed", rule="review", kill_reason="review FAIL")
    if any(not _seed_has_metrics(seed) for seed in seeds):
        return _node(summary, status="failed", rule="no_metrics", kill_reason=_NO_METRIC)
    if any(_bad_number(seed) for seed in seeds):
        return _node(summary, status="failed", rule="nan", kill_reason="nan or divergent scores")
    limit = _latency_limit(brief)
    if (
        limit is not None
        and summary.latency_p95_ms is not None
        and summary.latency_p95_ms > limit
    ):
        return _node(
            summary,
            status="killed",
            rule="latency",
            kill_reason=f"p95 latency {summary.latency_p95_ms:g} ms exceeds {limit:g} ms",
        )
    cap = brief.efficiency.train_dbu_per_candidate
    if cap is not None and summary.dbu is not None and summary.dbu > cap:
        return _node(
            summary,
            status="killed",
            rule="dbu_cap",
            kill_reason=f"training DBU {summary.dbu:g} exceeds per-candidate cap {cap:g}",
        )
    return None


def _seed_has_metrics(seed: RunView) -> bool:
    return seed.metric is not None or seed.delta is not None


def _bad_number(seed: RunView) -> bool:
    if seed.divergent:
        return True
    if seed.metric is not None and not math.isfinite(seed.metric):
        return True
    if seed.delta is None:
        return False
    bounds = (seed.delta.point, seed.delta.ci_lo, seed.delta.ci_hi)
    return not all(math.isfinite(value) for value in bounds)


def _latency_limit(brief: Brief) -> float | None:
    serving = brief.efficiency.serving
    if serving is None or serving.latency_p95_ms_per_100_candidates is None:
        return None
    return float(serving.latency_p95_ms_per_100_candidates)


def _missing_delta(summary: _Summary) -> NodeDecision | None:
    if summary.delta is not None:
        return None
    return _node(
        summary,
        status="failed",
        rule="no_delta",
        kill_reason="no paired delta versus the same-rung LightGBM reference",
    )


def _floor_kill(summary: _Summary, brief: Brief, rung: str) -> NodeDecision | None:
    threshold = _floor_threshold(rung, brief.min_effect)
    if threshold is None or summary.delta is None:
        return None
    checks = [("lgbm_reference", summary.delta), *summary.ref_deltas.items()]
    for name, delta in checks:
        if delta.ci_hi < threshold:
            return _node(
                summary,
                status="killed",
                rule="below_floor",
                kill_reason=(
                    f"below floor at {rung} vs {name}: delta ci_hi {_num(delta.ci_hi)} "
                    f"is below {_num(threshold)}"
                ),
            )
    return None


def _floor_threshold(rung: str, min_effect: float) -> float | None:
    if rung == "R1":
        return -min_effect
    if rung == "R2":
        return 0.0
    if rung == "R3":
        return min_effect
    return None


def _dominance_kills(
    nodes: Sequence[_Summary],
    *,
    n_resamples: int,
    seed: int,
) -> dict[str, NodeDecision]:
    killed: dict[str, NodeDecision] = {}
    for loser in nodes:
        if loser.delta is None:
            continue
        for winner in nodes:
            if winner.node_id == loser.node_id or winner.delta is None:
                continue
            lower = _difference_lower(loser, winner, n_resamples=n_resamples, seed=seed)
            if lower is None or not lower > 0:
                continue
            if not _cost_no_higher(winner, loser):
                continue
            killed[loser.node_id] = _node(
                loser,
                status="killed",
                rule="dominance",
                kill_reason=(
                    f"dominated by {winner.node_id}: paired lower bound of "
                    f"(delta_b - delta_a) is {_num(lower)} > 0 and training DBU "
                    "and latency are no higher"
                ),
            )
            break
    return killed


def _difference_lower(
    loser: _Summary,
    winner: _Summary,
    *,
    n_resamples: int,
    seed: int,
) -> float | None:
    if loser.delta is None or winner.delta is None:
        return None
    paired = _paired_lower(
        loser.query_deltas,
        winner.query_deltas,
        n_resamples=n_resamples,
        seed=seed,
    )
    if paired is not None:
        return paired
    half_winner = (winner.delta.ci_hi - winner.delta.ci_lo) / 2
    half_loser = (loser.delta.ci_hi - loser.delta.ci_lo) / 2
    return (winner.delta.point - loser.delta.point) - (half_winner + half_loser)


def _paired_lower(
    loser: Sequence[QueryDelta],
    winner: Sequence[QueryDelta],
    *,
    n_resamples: int,
    seed: int,
) -> float | None:
    if not loser or not winner:
        return None
    by_loser = {item.query_id: item for item in loser}
    by_day: dict[str, list[float]] = {}
    for item in winner:
        other = by_loser.get(item.query_id)
        if other is None:
            continue
        by_day.setdefault(item.event_day, []).append(item.delta - other.delta)
    if not by_day:
        return None
    return _blocked_lower(by_day, n_resamples=n_resamples, seed=seed)


def _blocked_lower(by_day: Mapping[str, Sequence[float]], *, n_resamples: int, seed: int) -> float:
    day_keys = sorted(by_day)
    rng = random.Random(seed)
    stats: list[float] = []
    for _draw in range(n_resamples):
        sample: list[float] = []
        for _day in range(len(day_keys)):
            values = by_day[day_keys[rng.randrange(len(day_keys))]]
            for _item in range(len(values)):
                sample.append(values[rng.randrange(len(values))])
        stats.append(sum(sample) / len(sample))
    return _quantile(stats, 0.025)


def _quantile(values: Sequence[float], q: float) -> float:
    ordered = sorted(values)
    pos = (len(ordered) - 1) * q
    low = math.floor(pos)
    high = math.ceil(pos)
    if low == high:
        return ordered[low]
    weight = pos - low
    return ordered[low] * (1 - weight) + ordered[high] * weight


def _cost_no_higher(winner: _Summary, loser: _Summary) -> bool:
    if winner.dbu is None or loser.dbu is None:
        return False
    if winner.latency_p95_ms is None or loser.latency_p95_ms is None:
        return False
    return winner.dbu <= loser.dbu and winner.latency_p95_ms <= loser.latency_p95_ms


def _halve(nodes: Sequence[_Summary], rung: str) -> tuple[list[_Summary], str | None]:
    ordered = sorted(nodes, key=_rank_key)
    if not ordered:
        return [], None
    keep_n = math.ceil(len(ordered) / 2)
    kept = list(ordered[:keep_n])
    cut = list(ordered[keep_n:])
    slot: str | None = None
    if rung == "R2" and cut:
        kept_families = {node.family for node in kept}
        wiped = [node for node in cut if node.family not in kept_families]
        if wiped:
            best = min(wiped, key=_rank_key)
            kept.append(best)
            slot = best.node_id
    return kept, slot


def _promote(kept: Sequence[_Summary]) -> list[str]:
    eligible = [node for node in kept if node.delta is not None and node.delta.ci_lo > 0]
    eligible.sort(key=lambda node: (-node.delta.point, node.node_id))  # type: ignore[union-attr]
    return [node.node_id for node in eligible[:3]]


def _rank_key(node: _Summary) -> tuple[float, float, float, str]:
    point = node.delta.point if node.delta is not None else float("-inf")
    dbu = node.dbu if node.dbu is not None else math.inf
    latency = node.latency_p95_ms if node.latency_p95_ms is not None else math.inf
    return (-point, dbu, latency, node.node_id)


def _node(
    summary: _Summary,
    *,
    status: str,
    rule: str,
    kill_reason: str | None,
) -> NodeDecision:
    return NodeDecision(
        node_id=summary.node_id,
        family=summary.family,
        status=status,
        rule=rule,
        kill_reason=kill_reason,
        delta=summary.delta,
        ref_deltas=summary.ref_deltas,
        seeds=[seed.seed for seed in summary.seeds],
    )


def _ledger_from(run_id: str, view: RunView, decision: NodeDecision) -> LedgerRow:
    metric = view.metric
    if metric is not None and not math.isfinite(metric):
        metric = None
    return LedgerRow(
        run_id=run_id,
        node_id=view.node_id,
        parent_id=view.parent_id,
        round=view.round,
        operator=view.operator,
        kind=view.kind,
        family=view.family,
        claim=view.claim or view.node_id,
        rung=view.rung,
        seed=view.seed,
        metric=metric,
        delta_vs_baseline=decision.delta,
        dbu=view.dbu,
        wall_min=view.wall_min,
        latency_p95_ms=view.latency_p95_ms,
        status=decision.status,  # type: ignore[arg-type]
        kill_reason=decision.kill_reason,
        git_sha=view.git_sha,
        mlflow_run_id=view.mlflow_run_id,
        job_run_id=view.job_run_id,
    )


def _num(value: float) -> str:
    return f"{value:.6g}"


def _load_run_file(path: Path) -> list[Any]:
    payload = json.loads(path.read_text())
    if isinstance(payload, dict):
        if "runs" in payload:
            return list(payload["runs"])
        return [payload]
    if isinstance(payload, list):
        return list(payload)
    raise PrunerError(f"{path} did not contain runs")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Prune one rung from logged metrics")
    parser.add_argument("--run-id", default="")
    parser.add_argument("--rung", default="R0")
    parser.add_argument("--metric", default="")
    parser.add_argument("--brief", default="")
    parser.add_argument("--runs", default="")
    parser.add_argument("--baselines", default="")
    parser.add_argument("--root", default="")
    args = parser.parse_args(argv)
    if args.runs:
        root = Path(args.root) if args.root else Path(os.environ.get("AMT_ROOT", Path.cwd()))
        if not args.brief:
            raise SystemExit("pruner --runs requires --brief")
        if not args.run_id:
            raise SystemExit("pruner --runs requires --run-id")
        brief = load_brief(Path(args.brief))
        runs = _load_run_file(Path(args.runs))
        baselines = _load_run_file(Path(args.baselines)) if args.baselines else []
        decision = decide(args.rung, runs, baselines, brief)
        commit_rung(root, args.run_id, runs, decision)
        print(decision.model_dump_json())
        return
    metric = float(args.metric) if args.metric else None
    status, reason = _legacy_status(metric)
    print(
        json.dumps(
            {
                "run_id": args.run_id,
                "rung": args.rung,
                "status": status,
                "kill_reason": reason,
            }
        )
    )


if __name__ == "__main__":
    main()
