"""Parameters the search loop sends to the train and confirm jobs."""

from __future__ import annotations

import json
import os
from pathlib import Path

from amt.config import Settings, load_settings
from amt.jobs_backend import FakeJobs
from amt.pruner import DeltaVsBaseline, RunView
from amt.schemas import Brief
from amt.scores import scores_exist, scores_path


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
    params = {
        "candidate_id": node_id,
        "node_id": node_id,
        "parent_id": parent_id or "",
        "family": family,
        "operator": operator,
        "rung": rung,
        "seed": str(seed),
        "split": "search_val",
        "batch": batch,
        "spec": spec,
        "baseline": "",
        "reference_score": "production",
        "paired_reference": "production",
        "model": model,
        "catalog": settings.catalog,
        "schema": settings.schema_name,
        "dataset_dir": "",
        "experiment_name": settings.experiment_for(brief.name or spec_name or run_id),
        "git_sha": git_sha,
        "run_id": run_id,
        "baseline_scores": baseline if scores_exist(baseline, root) else "",
        "write_scores": "",
        "root": "",
        "shared_cluster": "true" if settings.fixed_cluster else "",
        "matrix": "",
        "device": device,
    }
    return params


def missing_baseline_message(root: Path, brief: Brief, rung: str) -> str | None:
    spec = brief.resolved_dataset_spec() or ""
    path = scores_path(root, Path(spec).name, "lgbm_reference", rung, 0)
    if scores_exist(path, root):
        return None
    return "run amt noise --submit first"


def lgbm_baseline_view(root: Path, brief: Brief, rung: str, git_sha: str) -> RunView:
    """Same-rung LightGBM reference from round 0, or a zero delta when round 0 is not required."""

    metric, delta = _round0_lgbm(root, brief, rung)
    return RunView(
        node_id="lgbm_reference",
        family="lgbm_reference",
        kind="baseline",
        rung=rung,  # type: ignore[arg-type]
        split="search_val",
        operator="draft",
        claim="same-rung lightgbm reference",
        metric=metric,
        delta=delta,
        git_sha=git_sha,
    )


def _round0_lgbm(root: Path, brief: Brief, rung: str) -> tuple[float, DeltaVsBaseline]:
    spec = brief.resolved_dataset_spec() or ""
    path = root / "runs" / f"round0-{brief.name or Path(spec).name}" / "round0.json"
    if path.is_file():
        rows = json.loads(path.read_text(encoding="utf-8"))
        matched = [
            row
            for row in rows
            if row.get("baseline_id") == "lgbm_reference" and row.get("rung") == rung
        ]
        if matched:
            point = sum(float(row["metric"]) for row in matched) / len(matched)
            return point, DeltaVsBaseline(point=0.0, ci_lo=0.0, ci_hi=0.0)
    return 0.4, DeltaVsBaseline(point=0.0, ci_lo=0.0, ci_hi=0.0)
