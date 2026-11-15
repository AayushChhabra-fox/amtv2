"""Search schedule and the materialized LightGBM reference.

Node ``c00`` is reserved for ``lgbm_reference``. ``amt run`` writes its
candidate files once, and copies the round-0 R1–R3 metrics onto that node so
later rounds can improve, combine, confirm, and finalize it like any other
candidate.
"""

from __future__ import annotations

import json
from pathlib import Path

import yaml

from amt.ledger import append, read_rows
from amt.paths import candidate_dir
from amt.schemas import Brief, DeltaVsBaseline, LedgerRow, NodeWork, RungScore, RunState

C00_ID = "c00"
ARCHIVE_FAMILY = "lgbm_reference"
_SEARCH_RUNGS = ("R1", "R2", "R3")

C00_MODEL_PY = '''\
"""LightGBM lambdarank reference. Numeric feat columns only."""

from __future__ import annotations

import pickle

import lightgbm as lgb
import numpy as np


class Ranker:
    def __init__(self) -> None:
        self.n_estimators = 50
        self.num_leaves = 15
        self.learning_rate = 0.1
        self.min_child_samples = 20
        self.columns: list[str] = []
        self.model = None

    def fit(self, train, valid, ctx) -> None:
        matrix, names = train.numeric_matrix()
        if matrix.shape[1] == 0:
            raise ValueError("lgbm reference needs at least one numeric feature")
        self.columns = list(names)
        self.model = lgb.LGBMRanker(
            objective="lambdarank",
            n_estimators=self.n_estimators,
            num_leaves=self.num_leaves,
            learning_rate=self.learning_rate,
            min_child_samples=self.min_child_samples,
            random_state=int(ctx.seed),
            n_jobs=1,
            verbosity=-1,
            deterministic=True,
            force_col_wise=True,
        )
        labels = np.asarray(train.labels(), dtype=int)
        groups = train.group_sizes()
        kwargs = {}
        valid_matrix, valid_names = valid.numeric_matrix(self.columns)
        if valid_names and len(valid.groups()) >= 1:
            kwargs["eval_X"] = valid_matrix
            kwargs["eval_y"] = np.asarray(valid.labels(), dtype=int)
            kwargs["eval_group"] = [valid.group_sizes()]
        try:
            self.model.fit(matrix, labels, group=groups, **kwargs)
        except Exception:
            if not kwargs:
                raise
            self.model.fit(matrix, labels, group=groups)

    def score(self, frame) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("lgbm reference has not been fit")
        matrix, _names = frame.numeric_matrix(self.columns)
        return np.asarray(self.model.predict(matrix), dtype=float)

    def save(self, path: str) -> None:
        with open(path, "wb") as handle:
            pickle.dump({"columns": self.columns, "model": self.model}, handle)

    @classmethod
    def load(cls, path: str) -> "Ranker":
        ranker = cls()
        with open(path, "rb") as handle:
            payload = pickle.load(handle)
        ranker.columns = list(payload["columns"])
        ranker.model = payload["model"]
        return ranker
'''


def rungs_for(operator: str, *, scale: str = "search") -> list[str]:
    """Rungs one operator runs.

    Search drafts stop at R2 so a weak family does not spend R3. Calibration
    still walks one draft through R3. Improve runs R2 then R3. Combine, and any
    R2 survivor entering confirmation, runs R3 only.
    """

    if operator == "draft":
        if scale == "calibration":
            return ["R1", "R2", "R3"]
        return ["R1", "R2"]
    if operator == "combine":
        return ["R3"]
    return ["R2", "R3"]


def seed_c00(root: Path, state: RunState, brief: Brief) -> NodeWork | None:
    """Materialize ``c00`` and attach round-0 metrics.

    A legacy node that already claimed ``c00`` for another family is left alone.
    """

    existing = next((node for node in state.nodes if node.node_id == C00_ID), None)
    if existing is not None and existing.family != ARCHIVE_FAMILY:
        return existing
    scores = c00_rung_scores(root, brief)
    if existing is None:
        existing = _c00_node(scores)
        state.nodes.insert(0, existing)
    elif scores and not existing.rung_scores:
        existing.rung_scores = scores
        _apply_headline(existing, scores)
    materialize_c00(root, state.run_id)
    return existing


def materialize_c00(root: Path, run_id: str) -> Path:
    """Write the reference candidate files. Rewriting the template keeps the archive canonical."""

    directory = candidate_dir(root, run_id, C00_ID)
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "model.py").write_text(C00_MODEL_PY, encoding="utf-8")
    payload = {
        "id": C00_ID,
        "family": ARCHIVE_FAMILY,
        "operator": "draft",
        "parent_id": None,
        "claim": "LightGBM lambdarank reference on numeric features.",
        "change": None,
        "fit_argument": "Same-rung LightGBM reference from round 0.",
        "cheap_test": "Smoke scores are finite, stable, and the length of the frame.",
        "kill_if": "Scores are non-finite or p95 latency exceeds the brief limit.",
        "device": "cpu",
    }
    (directory / "candidate.yaml").write_text(
        yaml.safe_dump(payload, sort_keys=False),
        encoding="utf-8",
    )
    (directory / "spec.md").write_text(
        "# c00\n\n"
        "Family: lgbm_reference\n\n"
        "Operator: draft\n\n"
        "Parents: none\n\n"
        "Claim: LightGBM lambdarank reference on numeric features.\n\n"
        "This node is the round-0 LightGBM reference. Improve and combine may "
        "use it as a parent. Fit and score use only the frame passed in.\n",
        encoding="utf-8",
    )
    (directory / "review.md").write_text(
        "# review\n\n"
        "Materialized LightGBM reference. Static checks and the Ranker smoke "
        "test apply to model.py.\n\n"
        "VERDICT: PASS\n",
        encoding="utf-8",
    )
    return directory


def c00_rung_scores(root: Path, brief: Brief) -> dict[str, RungScore]:
    """Mean round-0 LightGBM metric, production CI, and latency at R1–R3."""

    rows = _round0_rows(root, brief)
    scores: dict[str, RungScore] = {}
    for rung in _SEARCH_RUNGS:
        cells = [
            row
            for row in rows
            if row.get("baseline_id") == ARCHIVE_FAMILY and row.get("rung") == rung
        ]
        if not cells:
            continue
        metrics = [float(row["metric"]) for row in cells if row.get("metric") is not None]
        latencies = [
            float(row["latency_p95_ms"])
            for row in cells
            if row.get("latency_p95_ms") is not None
        ]
        scores[rung] = RungScore(
            metric=(sum(metrics) / len(metrics)) if metrics else None,
            delta_vs_production=_mean_production(cells),
            delta_vs_lgbm=DeltaVsBaseline(point=0.0, ci_lo=0.0, ci_hi=0.0),
            latency_p95_ms=max(latencies) if latencies else None,
        )
    return scores


def random_metric_at(root: Path, brief: Brief, rung: str) -> float | None:
    """Mean round-0 random baseline metric at ``rung``, when round 0 recorded it."""

    rows = [
        row
        for row in _round0_rows(root, brief)
        if row.get("baseline_id") == "random"
        and row.get("rung") == rung
        and row.get("metric") is not None
    ]
    if not rows:
        return None
    return sum(float(row["metric"]) for row in rows) / len(rows)


def record_c00_promotion(root: Path, state: RunState, rung: str) -> None:
    """Ledger row for a promoted archive node. Round 0 already paid for the fit."""

    if C00_ID not in state.promoted:
        return
    if any(
        row.run_id == state.run_id and row.node_id == C00_ID and row.rung == rung
        for row in read_rows(root)
    ):
        return
    node = next((item for item in state.nodes if item.node_id == C00_ID), None)
    score = None if node is None else node.rung_scores.get(rung)
    if score is None:
        return
    append(
        root,
        LedgerRow(
            run_id=state.run_id,
            node_id=C00_ID,
            round=state.round,
            operator="draft",
            kind="model",
            family=ARCHIVE_FAMILY,
            claim="LightGBM lambdarank reference on numeric features.",
            rung=rung,  # type: ignore[arg-type]
            seed=0,
            metric=score.metric,
            delta_vs_production=score.delta_vs_production,
            delta_vs_lgbm=score.delta_vs_lgbm,
            latency_p95_ms=score.latency_p95_ms,
            status="promoted",
            git_sha=state.git_sha or "unknown",
            mlflow_run_id="m-c00-round0",
            job_run_id="fake-c00-round0",
        ),
    )


def _c00_node(scores: dict[str, RungScore]) -> NodeWork:
    node = NodeWork(
        node_id=C00_ID,
        family=ARCHIVE_FAMILY,
        operator="draft",
        claim="LightGBM lambdarank reference on numeric features.",
        fit_argument="Same-rung LightGBM reference from round 0.",
        cheap_test="Smoke scores are finite, stable, and the length of the frame.",
        kill_if="Scores are non-finite or p95 latency exceeds the brief limit.",
        est_cost=0.0,
        rung="R3",
        search_round=0,
        status="kept",
        preflight="PASS",
        review="PASS",
        rung_scores=scores,
    )
    _apply_headline(node, scores)
    return node


def _apply_headline(node: NodeWork, scores: dict[str, RungScore]) -> None:
    latest = scores.get("R3") or scores.get("R2") or scores.get("R1")
    if latest is None or latest.delta_vs_production is None:
        return
    node.delta_point = latest.delta_vs_production.point
    node.delta_lo = latest.delta_vs_production.ci_lo
    node.delta_hi = latest.delta_vs_production.ci_hi


def _round0_rows(root: Path, brief: Brief) -> list[dict]:
    spec = brief.resolved_dataset_spec() or ""
    path = root / "runs" / f"round0-{brief.name or Path(spec).name}" / "round0.json"
    if not path.is_file():
        return []
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, dict):
        payload = payload.get("observations", [])
    if not isinstance(payload, list):
        return []
    return [row for row in payload if isinstance(row, dict)]


def _mean_production(rows: list[dict]) -> DeltaVsBaseline | None:
    usable = [row for row in rows if row.get("delta_point") is not None]
    if not usable:
        return None

    def _avg(key: str) -> float:
        return sum(float(row.get(key) or 0.0) for row in usable) / len(usable)

    return DeltaVsBaseline(
        point=_avg("delta_point"),
        ci_lo=_avg("delta_ci_lo"),
        ci_hi=_avg("delta_ci_hi"),
    )
