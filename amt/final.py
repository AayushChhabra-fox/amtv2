"""Gate G5: score the winner once on the final holdout, and rerun a ledger row."""

from __future__ import annotations

import json
from pathlib import Path

from amt.config import load_settings
from amt.jobs_backend import make_backend
from amt.ledger import read_rows
from amt.signoff import _RESERVED_SIGNERS
from amt.snapshot import upload_candidate
from amt.tokens import issue_token

_WIN = {"kept", "promoted"}


class FinalError(RuntimeError):
    """The final holdout cannot be opened."""


def open_final(root: Path, run_id: str, signer: str) -> Path:
    """Submit the winner, the runner-up, and the baselines once. A second call is refused."""

    person = signer.strip()
    if not person or person.casefold() in _RESERVED_SIGNERS:
        raise FinalError("signer must be a person's name")
    report = root / "runs" / run_id / "final_holdout.md"
    if report.is_file():
        raise FinalError("final holdout was already opened for this run")
    rows = [row for row in read_rows(root) if row.run_id == run_id and row.rung == "R3"]
    ranked = [row for row in rows if row.status in _WIN and row.kind != "baseline"]
    ranked.sort(key=lambda row: (-(row.metric or 0.0), row.node_id))
    if not ranked:
        raise FinalError("no R3 survivor to score on the final holdout")
    chosen = ranked[:2]
    settings = load_settings(root)
    backend = make_backend(root)
    token = issue_token(root, settings=settings)
    results: list[str] = []
    for row in chosen:
        model = upload_candidate(root, run_id, row.node_id, row.git_sha, settings=settings)
        params = _holdout_params(root, row, model=model, token=token, baseline="")
        run = backend.submit(
            job="final_holdout",
            job_key=f"final/{row.node_id}",
            params=params,
            idempotency_token=f"{run_id}:final:{row.node_id}",
        )
        backend.wait(run)
        results.append(f"- {row.node_id} job {run}")
    baselines = (("production", "ref.production"), ("lgbm_reference", "lgbm_reference"))
    for baseline, family in baselines:
        params = _holdout_params(
            root,
            chosen[0],
            model="",
            token=issue_token(root, settings=settings),
            baseline="reference_score" if baseline == "production" else "lgbm_reference",
            candidate_id=baseline,
            family=family,
        )
        run = backend.submit(
            job="final_holdout",
            job_key=f"final/{baseline}",
            params=params,
            idempotency_token=f"{run_id}:final:{baseline}",
        )
        backend.wait(run)
        results.append(f"- {baseline} job {run}")
    winner = chosen[0]
    select = _select_interval(rows, winner.node_id)
    final_found = backend.find_mlflow_run(  # type: ignore[attr-defined]
        tags={"run_id": run_id, "node_id": winner.node_id, "split": "final_holdout"}
    )
    flag = _overfit_line(select, final_found)
    body = "\n".join(
        [
            f"# Final holdout {run_id}",
            "",
            f"Signed by {person}.",
            "",
            flag,
            "",
            *results,
            "",
        ]
    )
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text(body, encoding="utf-8")
    return report


def reproduce(root: Path, run_id: str, node_id: str) -> str:
    """Rerun the recorded R3 seed and compare it with the ledger interval."""

    match = [
        row
        for row in read_rows(root)
        if row.run_id == run_id and row.node_id == node_id and row.rung == "R3"
    ]
    if not match:
        raise FinalError(f"no R3 ledger row for {node_id}")
    row = match[0]
    settings = load_settings(root)
    model = upload_candidate(root, run_id, node_id, row.git_sha, settings=settings)
    params = {
        "candidate_id": node_id,
        "node_id": node_id,
        "parent_id": row.parent_id or "",
        "family": row.family,
        "operator": row.operator,
        "rung": "R3",
        "seed": str(row.seed),
        "split": "search_val",
        "batch": f"reproduce-{run_id}",
        "spec": "",
        "baseline": "",
        "reference_score": "production",
        "paired_reference": "production",
        "model": model,
        "catalog": settings.catalog,
        "schema": settings.schema_name,
        "dataset_dir": "",
        "experiment_name": settings.experiment_for(run_id),
        "git_sha": row.git_sha,
        "run_id": f"reproduce-{run_id}",
        "baseline_scores": "",
        "write_scores": "",
        "root": "",
        "shared_cluster": "true" if settings.fixed_cluster else "",
        "matrix": "",
        "device": "cpu",
    }
    state_path = root / "runs" / run_id / "state.json"
    if state_path.is_file():
        payload = json.loads(state_path.read_text(encoding="utf-8"))
        brief_path = payload.get("brief_path")
        if brief_path:
            from amt.schemas import load_brief

            params["spec"] = load_brief(Path(brief_path)).resolved_dataset_spec() or ""
    backend = make_backend(root)
    job_run = backend.submit(
        job="train",
        job_key=f"reproduce/{node_id}",
        params=params,
        idempotency_token=f"{run_id}:reproduce:{node_id}:{row.seed}",
    )
    backend.wait(job_run)
    found = backend.find_mlflow_run(  # type: ignore[attr-defined]
        tags={"run_id": f"reproduce-{run_id}", "node_id": node_id, "seed": str(row.seed)}
    )
    metric = None if found is None else found.get("metric")
    if metric is None or row.metric is None:
        return "fail: no metric"
    half = 0.0
    if row.delta_vs_baseline is not None:
        half = (row.delta_vs_baseline.ci_hi - row.delta_vs_baseline.ci_lo) / 2
    if abs(float(metric) - float(row.metric)) <= max(half, 1e-9):
        return f"pass: {metric} within the R3 interval of {row.metric}"
    return f"fail: {metric} outside the R3 interval of {row.metric}"


def _holdout_params(
    root,
    row,
    *,
    model: str,
    token: str,
    baseline: str,
    candidate_id: str = "",
    family: str = "",
) -> dict[str, str]:
    settings = load_settings(root)
    return {
        "candidate_id": candidate_id or row.node_id,
        "seed": str(row.seed),
        "batch": f"final-{row.run_id}",
        "spec": "",
        "baseline": baseline,
        "reference_score": "production",
        "model": model,
        "catalog": settings.catalog,
        "schema": settings.schema_name,
        "dataset_dir": "",
        "experiment_name": settings.experiment_for(row.run_id),
        "git_sha": row.git_sha,
        "family": family or row.family,
        "run_id": row.run_id,
        "node_id": candidate_id or row.node_id,
        "parent_id": "",
        "operator": "draft",
        "paired_reference": "production",
        "baseline_scores": "",
        "confirm_token": token,
        "root": "",
        "shared_cluster": "true" if settings.fixed_cluster else "",
        "device": "cpu",
    }


def _select_interval(rows, node_id: str) -> tuple[float, float] | None:
    select = [row for row in rows if row.node_id == node_id and row.delta_vs_baseline is not None]
    if not select or select[0].delta_vs_baseline is None:
        return None
    delta = select[0].delta_vs_baseline
    return delta.ci_lo, delta.ci_hi


def _overfit_line(interval: tuple[float, float] | None, found: dict | None) -> str:
    if interval is None or not found or found.get("delta_point") is None:
        return "Overfit check: not enough intervals to compare."
    point = float(found["delta_point"])
    lo, hi = interval
    if lo <= point <= hi:
        return f"Overfit check: final delta {point} is inside the select interval [{lo}, {hi}]."
    return (
        f"Overfit check: final delta {point} is outside the select interval [{lo}, {hi}]. "
        "Flagged as overfit."
    )
