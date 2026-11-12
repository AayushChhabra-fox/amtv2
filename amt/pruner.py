"""Pruner stub for the walking skeleton.

The deterministic rules from the plan (section 5.3) are Phase 1. This stub
only checks that a same-rung MLflow metric exists. It never receives holdout
frames or holdout metric names.
"""

from __future__ import annotations

from amt.schemas import LedgerRow


def decide(metric: float | None) -> tuple[str, str | None]:
    if metric is None:
        return "failed", "run produced no metric"
    return "kept", None


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
    status, reason = decide(metric)
    return LedgerRow(
        run_id=run_id,
        node_id=node_id,
        round=0,
        operator="draft",
        family=family,
        claim=claim,
        rung=rung,
        seed=seed,
        metric=metric,
        dbu=dbu,
        wall_min=wall_min,
        status=status,
        kill_reason=reason,
        git_sha=git_sha,
        mlflow_run_id=mlflow_run_id,
        job_run_id=job_run_id,
    )
