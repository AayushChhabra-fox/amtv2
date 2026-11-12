"""Pruner entry point.

Section 5.3 rules are the control-layer milestone. This job entry only
reports whether a metric exists, and it never reads holdout frames.
"""

from __future__ import annotations

import argparse
import json

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


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Prune one rung from logged metrics")
    parser.add_argument("--run-id", default="")
    parser.add_argument("--rung", default="R0")
    parser.add_argument("--metric", default="")
    args = parser.parse_args(argv)
    metric = float(args.metric) if args.metric else None
    status, reason = decide(metric)
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
