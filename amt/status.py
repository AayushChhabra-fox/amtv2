"""Human-readable run status. `--audit` checks the ledger trace."""

from __future__ import annotations

import json
from pathlib import Path

from amt.jobs_backend import databricks_bin
from amt.ledger import read_rows


def format_status(root: Path, run_id: str, *, audit: bool = False) -> str:
    state_path = root / "runs" / run_id / "state.json"
    lines = [f"run {run_id}"]
    if state_path.is_file():
        state = json.loads(state_path.read_text())
        lines.append(f"stage {state.get('stage')}  spent_dbu {state.get('spent_dbu')}")
        for key, record in (state.get("submitted_jobs") or {}).items():
            lines.append(
                f"job {key}  run_id={record.get('job_run_id')}  "
                f"mlflow={record.get('mlflow_run_id')}"
            )
    else:
        lines.append("no state file")
    rows = [row for row in read_rows(root) if row.run_id == run_id]
    if not rows:
        lines.append("ledger empty")
    for row in rows:
        lines.append(
            f"ledger {row.node_id} {row.rung} seed={row.seed} "
            f"metric={row.metric} status={row.status} "
            f"job={row.job_run_id} mlflow={row.mlflow_run_id} git={row.git_sha}"
        )
    if audit:
        problems = audit_rows(root, run_id)
        if problems:
            lines.append("audit FAIL")
            lines.extend(f"  {problem}" for problem in problems)
        else:
            lines.append("audit PASS")
    return "\n".join(lines)


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
