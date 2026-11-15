"""Parameters the train, confirm, and final-holdout jobs accept.

Each builder returns only the names declared on that job. A laptop edit of one
dataset spec must not force every other spec's round-0 token to change; the
caller passes the experiment name from ``Settings.experiment_for``.
"""

from __future__ import annotations

import os
from pathlib import Path

from amt.config import Settings, load_settings
from amt.jobs_backend import FakeJobs, declared_parameters
from amt.pruner import DeltaVsBaseline, RunView
from amt.schemas import Brief
from amt.scores import scores_exist, scores_path

# Insertion order matches the train job. Jobs that do not declare a name omit it.
_PARAM_ORDER = (
    "candidate_id",
    "node_id",
    "parent_id",
    "family",
    "operator",
    "rung",
    "seed",
    "split",
    "batch",
    "spec",
    "baseline",
    "reference_score",
    "paired_reference",
    "model",
    "catalog",
    "schema",
    "dataset_dir",
    "experiment_name",
    "git_sha",
    "run_id",
    "baseline_scores",
    "write_scores",
    "confirm_token",
    "root",
    "shared_cluster",
    "matrix",
    "device",
)


def build_job_params(job: str, **values: object) -> dict[str, str]:
    """String parameters for ``job``, limited to names its bundle YAML declares."""

    text = {key: "" if values.get(key) is None else str(values.get(key)) for key in _PARAM_ORDER}
    allowed = declared_parameters(job)
    if not allowed:
        return text
    return {key: text[key] for key in _PARAM_ORDER if key in allowed}


def baselines_required(backend: object) -> bool:
    if os.environ.get("AMT_REQUIRE_BASELINES") == "1":
        return True
    return not isinstance(backend, FakeJobs)


def train_params(
    *,
    root: Path,
    brief: Brief,
    node_id: str,
    parent_id: str | None,
    family: str,
    operator: str,
    rung: str,
    seed: int,
    batch: str,
    run_id: str,
    git_sha: str,
    model: str,
    device: str = "cpu",
    settings: Settings | None = None,
) -> dict[str, str]:
    settings = settings or load_settings(root)
    spec = brief.resolved_dataset_spec() or ""
    spec_name = Path(spec).name
    baseline = scores_path(root, spec_name, "lgbm_reference", rung, seed, settings=settings)
    return build_job_params(
        "train",
        candidate_id=node_id,
        node_id=node_id,
        parent_id=parent_id or "",
        family=family,
        operator=operator,
        rung=rung,
        seed=seed,
        split="search_val",
        batch=batch,
        spec=spec,
        baseline="",
        reference_score="production",
        paired_reference="production",
        model=model,
        catalog=settings.catalog,
        schema=settings.schema_name,
        dataset_dir="",
        experiment_name=settings.experiment_for(brief.name or spec_name or run_id),
        git_sha=git_sha,
        run_id=run_id,
        baseline_scores=baseline if scores_exist(baseline, root) else "",
        write_scores="",
        root="",
        shared_cluster="true" if settings.fixed_cluster else "",
        matrix="",
        device=device,
    )


def missing_baseline_message(root: Path, brief: Brief, rung: str) -> str | None:
    spec = brief.resolved_dataset_spec() or ""
    path = scores_path(root, Path(spec).name, "lgbm_reference", rung, 0)
    if scores_exist(path, root):
        return None
    return "run amt noise --submit first"


def lgbm_baseline_view(root: Path, brief: Brief, rung: str, git_sha: str) -> RunView:
    """Same-rung LightGBM reference from round 0, or a zero delta when round 0 is absent."""

    from amt.search_policy import c00_rung_scores

    score = c00_rung_scores(root, brief).get(rung)
    zero = DeltaVsBaseline(point=0.0, ci_lo=0.0, ci_hi=0.0)
    metric = 0.4 if score is None or score.metric is None else score.metric
    production = None if score is None else score.delta_vs_production
    latency = None if score is None else score.latency_p95_ms
    return RunView(
        node_id="lgbm_reference",
        family="lgbm_reference",
        kind="baseline",
        rung=rung,  # type: ignore[arg-type]
        split="search_val",
        operator="draft",
        claim="same-rung lightgbm reference",
        metric=metric,
        delta=zero,
        delta_vs_lgbm=zero,
        delta_vs_production=production,
        latency_p95_ms=latency,
        git_sha=git_sha,
    )
