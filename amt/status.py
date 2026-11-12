"""Human-readable run status.

The report is a table of live nodes, spend versus the pool, and every kill
with its reason. ``--audit`` checks that each ledger row links to a job run,
an MLflow run, and a git SHA.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from amt.jobs_backend import databricks_bin
from amt.ledger import read_rows
from amt.schemas import LedgerRow


def format_status(root: Path, run_id: str, *, audit: bool = False) -> str:
    state_path = root / "runs" / run_id / "state.json"
    lines = [f"run {run_id}"]
    state: dict | None = None
    if state_path.is_file():
        state = json.loads(state_path.read_text())
        lines.append(
            f"stage {state.get('stage')}  round {state.get('round', 0)}  "
            f"spend {_num(state.get('spent_dbu', 0))} / {_num(state.get('budget_dbu', 0))} DBU"
        )
        for key, record in (state.get("submitted_jobs") or {}).items():
            lines.append(
                f"job {key}  run_id={record.get('job_run_id')}  "
                f"mlflow={record.get('mlflow_run_id')}"
            )
    else:
        lines.append("no state file")
    rows = [row for row in read_rows(root) if row.run_id == run_id]
    live = list((state or {}).get("live_nodes") or [])
    lines.extend(_node_table(rows, live))
    lines.extend(_kill_lines(rows))
    if audit:
        problems = audit_rows(root, run_id)
        if problems:
            lines.append("audit FAIL")
            lines.extend(f"  {problem}" for problem in problems)
        else:
            lines.append("audit PASS")
    return "\n".join(lines)


def _node_table(rows: list[LedgerRow], live: list[str]) -> list[str]:
    latest: dict[str, LedgerRow] = {}
    for row in rows:
        latest[row.node_id] = row
    order = list(dict.fromkeys([*live, *latest]))
    lines = [
        "",
        "live nodes",
        f"{'node':<16} {'family':<16} {'rung':<6} {'status':<10} {'delta':>10}",
    ]
    if not order:
        lines.append("(none)")
        return lines
    for node_id in order:
        row = latest.get(node_id)
        if row is None:
            lines.append(f"{node_id:<16} {'-':<16} {'-':<6} {'pending':<10} {'-':>10}")
            continue
        delta = "-" if row.delta_vs_baseline is None else f"{row.delta_vs_baseline.point:.4f}"
        lines.append(
            f"{row.node_id:<16} {row.family:<16} {row.rung:<6} {row.status:<10} {delta:>10}"
        )
    return lines


def _kill_lines(rows: list[LedgerRow]) -> list[str]:
    lines = ["", "kills"]
    kills = [row for row in rows if row.status in {"killed", "failed"}]
    if not kills:
        lines.append("(none)")
        return lines
    for row in kills:
        lines.append(
            f"{row.node_id} {row.rung} seed={row.seed} {row.status}: {row.kill_reason}"
        )
    return lines


def _num(value: object) -> str:
    try:
        return f"{float(value):.3f}"  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return "-"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Print AMT run status")
    parser.add_argument("--run-id", default="toy")
    parser.add_argument("--audit", action="store_true")
    parser.add_argument("--root", type=Path, default=Path.cwd())
    args = parser.parse_args(argv)
    print(format_status(args.root, args.run_id, audit=args.audit))
    if args.audit and audit_rows(args.root, args.run_id):
        return 1
    return 0


def audit_rows(root: Path, run_id: str) -> list[str]:
    problems: list[str] = []
    rows = [row for row in read_rows(root) if row.run_id == run_id]
    if not rows:
        return ["no ledger rows"]
    for row in rows:
        if not row.job_run_id:
            problems.append(f"{row.node_id} missing job_run_id")
        if not row.mlflow_run_id:
            problems.append(f"{row.node_id} missing mlflow_run_id")
        if not row.git_sha or row.git_sha == "unknown":
            problems.append(f"{row.node_id} missing git_sha")
        if row.job_run_id and not str(row.job_run_id).startswith("fake-"):
            problems.extend(_remote_trace(root, row.job_run_id, row.mlflow_run_id))
    return problems


def _remote_trace(root: Path, job_run_id: str, mlflow_run_id: str | None) -> list[str]:
    import subprocess

    problems: list[str] = []
    job = subprocess.run(
        [databricks_bin(), "api", "get", f"/api/2.1/jobs/runs/get?run_id={job_run_id}"],
        cwd=root,
        check=False,
        capture_output=True,
        text=True,
    )
    if job.returncode != 0:
        problems.append(f"jobs get-run {job_run_id} failed")
    if mlflow_run_id:
        mlflow = subprocess.run(
            [databricks_bin(), "api", "get", f"/api/2.0/mlflow/runs/get?run_id={mlflow_run_id}"],
            cwd=root,
            check=False,
            capture_output=True,
            text=True,
        )
        if mlflow.returncode != 0:
            problems.append(f"mlflow get-run {mlflow_run_id} failed")
    return problems


if __name__ == "__main__":
    raise SystemExit(main())
