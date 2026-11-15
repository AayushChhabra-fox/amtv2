"""Local dataset checks and the remote one-hour preflight slice.

The day-count and metric-gain checks run on the laptop. The slice itself is a
``build_dataset`` job with ``--preflight-slice``.
"""

from __future__ import annotations

import traceback
from pathlib import Path

from amt.batch import fetch_job_failure, submit_dataset, wait_job
from amt.schemas import Brief
from harness.data.contract import DatasetBuildError, DatasetSpec, load_spec
from harness.data.preflight import search_val_day_count


def run_local_checks(spec: DatasetSpec, brief: Brief | None = None) -> dict:
    """Day span, and brief ``metric.gains`` when a brief is supplied.

    ``search_val`` must cover ``preflight.min_search_val_days`` calendar days.
    The default is five, which is what the day-blocked bootstrap needs. A demo
    spec sets a lower number on purpose.
    """

    days = search_val_day_count(spec)
    required = spec.preflight.min_search_val_days
    if days < required:
        raise DatasetBuildError(
            "split",
            (
                f"search_val spans {days} calendar days; "
                f"day-blocked bootstrap needs {required}"
            ),
            [{"search_val": spec.split.search_val, "days": days, "required": required}],
        )
    grades: list[int] = []
    if brief is not None:
        _check_metric(spec, brief)
        grades = positive_grades(brief)
    return {"search_val_days": days, "positive_grades": grades}


def positive_grades(brief: Brief) -> list[int]:
    metric = brief.problem.metric
    if metric is None or not metric.gains:
        raise DatasetBuildError(
            "label_support",
            "brief.problem.metric.gains is required for dataset preflight",
        )
    found: list[int] = []
    for name, value in metric.gains.items():
        if isinstance(value, bool) or not isinstance(value, int):
            raise DatasetBuildError("label_support", f"gain {name} must be an int")
        if value < 0:
            raise DatasetBuildError("label_support", f"gain {name} is negative")
        if value > 0:
            found.append(value)
    if not found:
        raise DatasetBuildError("label_support", "metric.gains has no positive grade")
    return sorted(set(found))


def _check_metric(spec: DatasetSpec, brief: Brief) -> None:
    metric = brief.problem.metric
    if metric is None:
        raise DatasetBuildError("label_support", "brief.problem.metric is required")
    if spec.primary_metric and spec.primary_metric != metric.name:
        raise DatasetBuildError(
            "contract",
            (
                f"dataset.yaml primary_metric {spec.primary_metric} "
                f"does not match brief metric {metric.name}"
            ),
        )


def failure_from_exception(exc: BaseException, *, fragment: str = "") -> dict:
    return {
        "exception_type": type(exc).__name__,
        "message": str(exc),
        "traceback": "".join(traceback.format_exception(type(exc), exc, exc.__traceback__)),
        "kind": str(getattr(exc, "kind", "") or ""),
        "examples": list(getattr(exc, "examples", None) or []),
        "fragment": fragment,
        "fragment_path": "",
        "source_schemas": {},
    }


def run_preflight(
    root: Path,
    spec_path: str,
    brief: Brief | None = None,
    *,
    remote: bool = True,
    slice_window: str = "1h",
) -> dict:
    """Local checks, then the remote slice unless ``remote`` is false."""

    path = Path(spec_path)
    if not path.is_absolute():
        path = root / path
    spec = load_spec(path)
    try:
        local = run_local_checks(spec, brief)
    except DatasetBuildError as exc:
        fragment = "labels" if exc.kind == "label_support" else "search_val"
        return {"status": "fail", "failure": failure_from_exception(exc, fragment=fragment)}
    if not remote:
        return {"status": "pass", **local}
    grades = ",".join(str(grade) for grade in local["positive_grades"])
    run_id = submit_dataset(
        root,
        spec_path,
        preflight_slice=slice_window,
        positive_grades=grades,
    )
    result = wait_job(root, run_id)
    if result.get("result_state") == "SUCCESS":
        return {"status": "pass", "run_id": run_id, **local}
    failure = fetch_job_failure(root, run_id)
    return {"status": "fail", "run_id": run_id, "failure": failure, **local}
