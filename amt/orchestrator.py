"""Resumable toy loop: proposal, critic, logistic candidate, smoke, hello_train, pruner."""

from __future__ import annotations

import json
import os
import subprocess
import time
from pathlib import Path
from typing import Any

from amt.agents import run_role
from amt.budget import (
    BudgetExceeded,
    assert_can_spend,
    estimate_dbu,
    inflight_dbu,
    node_committed_dbu,
)
from amt.confirm import ConfirmationSubmission, confirmation_submissions
from amt.jobs_backend import make_backend
from amt.ledger import append, read_rows
from amt.preflight import smoke_candidate
from amt.pruner import ledger_row
from amt.schemas import AgentRevision, Brief, RunState, load_brief, parse_agent_output


def run_brief(
    brief_path: Path,
    *,
    root: Path | None = None,
    fresh: bool = False,
    run_id: str | None = None,
    approval_mode: str | None = None,
) -> RunState:
    root = root or Path.cwd()
    brief = load_brief(brief_path)
    if approval_mode:
        brief.approval_mode = approval_mode  # type: ignore[assignment]
    resolved_id = run_id or brief.run_id or brief.name or brief_path.stem
    state_path = root / "runs" / resolved_id / "state.json"
    if fresh and state_path.is_file():
        state_path.unlink()
    if state_path.is_file():
        state = _load_state(state_path)
    else:
        state = _new_state(brief, resolved_id, brief_path, root)
    if state.stage in {"REPORT", "ESCALATE"}:
        return state
    backend = make_backend(root)
    while state.stage not in {"REPORT", "ESCALATE"}:
        if _stop_requested(root, state.run_id):
            state.stop_requested = True
            state.stage = "ESCALATE"
            _save(state_path, state)
            _write_escalate(root, state, "stop file present")
            return state
        finished = state.stage
        state = _step(state, brief, root, backend)
        _save(state_path, state)
        if os.environ.get("AMT_STOP_AFTER") == finished:
            return state
    return state


def _step(state: RunState, brief: Brief, root: Path, backend: object) -> RunState:
    stage = state.stage
    if stage == "PLAN":
        output = run_role("planner", brief, root, state.run_id)
        rejected = _reject_invalid_output(state, "planner", output, root)
        if rejected is not None:
            return rejected
        state.stage = "CRITIQUE"
    elif stage == "CRITIQUE":
        output = run_role("critic", brief, root, state.run_id)
        rejected = _reject_invalid_output(state, "critic", output, root)
        if rejected is not None:
            return rejected
        state.stage = "IMPLEMENT"
    elif stage == "IMPLEMENT":
        run_role("implementer", brief, root, state.run_id)
        state.stage = "SMOKE"
    elif stage == "SMOKE":
        smoke_candidate(root / "candidates" / state.candidate_id)
        state.stage = "SUBMIT"
    elif stage == "SUBMIT":
        if _submit(state, brief, root, backend):
            state.stage = "WAIT"
        else:
            state.stage = "ESCALATE"
    elif stage == "WAIT":
        _wait(state, root, backend)
        state.stage = "PRUNE"
    elif stage == "PRUNE":
        _prune(state, brief, root, backend)
        state.stage = "REPORT"
        _write_digest(root, state)
    else:
        raise RuntimeError(f"unknown stage {stage}")
    return state


def request_confirmation(
    promoted: list[str],
    *,
    batch: str,
    spec: str,
    run_id: str,
    families: dict[str, str] | None = None,
    git_sha: str = "unknown",
    catalog: str = "product_api_dev",
    schema: str = "amt",
) -> list[ConfirmationSubmission]:
    """Open select_holdout scoring for the top 3 promoted nodes, 3 seeds each.

    This is the only supported trigger. The split stays ``select_holdout``.
    """

    return confirmation_submissions(
        promoted,
        batch=batch,
        spec=spec,
        run_id=run_id,
        families=families,
        git_sha=git_sha,
        catalog=catalog,
        schema=schema,
    )


def apply_agent_output(
    state: RunState, role: str, payload: Any
) -> tuple[RunState, AgentRevision | None]:
    """Validate role output. A failure is one revision to hand back to the agent."""
    parsed = parse_agent_output(role, payload)
    if parsed.revision is None:
        return state, None
    updated = state.model_copy(
        update={"revisions": state.revisions + parsed.revision.revision_count}
    )
    return updated, parsed.revision


def _structured_role_output(role: str, output: Any) -> bool:
    if role not in {"planner", "critic"} or not isinstance(output, dict):
        return False
    if role == "planner":
        return "candidates" in output or "proposals" in output
    return "verdicts" in output or "decision" in output


def _reject_invalid_output(state: RunState, role: str, output: Any, root: Path) -> RunState | None:
    if not _structured_role_output(role, output):
        return None
    state, revision = apply_agent_output(state, role, output)
    if revision is None:
        return None
    _write_feedback(root, state, revision)
    state.stage = "ESCALATE"
    _write_escalate(root, state, revision.message)
    return state


def _write_feedback(root: Path, state: RunState, revision: AgentRevision) -> None:
    path = root / "runs" / state.run_id / "agent_feedback.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(revision.message)


def _submit(state: RunState, brief: Brief, root: Path, backend: object) -> bool:
    key = f"{state.candidate_id}/{state.rung}/{state.seed}"
    record = dict(state.submitted_jobs.get(key) or {})
    if record.get("job_run_id"):
        state.submitted_jobs[key] = record
        return True
    pinned = brief.candidate
    estimate = float(pinned.est_dbu if pinned else 0.1)
    committed = node_committed_dbu(read_rows(root), state.run_id, state.candidate_id)
    try:
        assert_can_spend(
            brief,
            state.spent_dbu + inflight_dbu(state.submitted_jobs),
            estimate,
            candidate_spent_dbu=committed + inflight_dbu(state.submitted_jobs, state.candidate_id),
        )
    except BudgetExceeded as exc:
        _write_escalate(root, state, str(exc))
        return False
    params = {
        "candidate_id": state.candidate_id,
        "rung": state.rung,
        "seed": str(state.seed),
        "batch": state.batch,
        "family": state.family,
        "git_sha": state.git_sha,
        "dbu_est": str(estimate),
        "amt_run_id": state.run_id,
        "experiment_name": "/Shared/amt/nfy_article_rerank",
    }
    token = f"{state.run_id}:{key}"
    job_run_id = backend.submit(job_key=key, params=params, idempotency_token=token)  # type: ignore[attr-defined]
    record.update({"job_run_id": job_run_id, "submitted_at": time.time(), "dbu_est": estimate})
    state.submitted_jobs[key] = record
    return True


def _wait(state: RunState, root: Path, backend: object) -> None:
    del root
    key = f"{state.candidate_id}/{state.rung}/{state.seed}"
    record = state.submitted_jobs[key]
    result = backend.wait(record["job_run_id"])  # type: ignore[attr-defined]
    record["result_state"] = result.get("result_state")
    record["finished_at"] = time.time()
    if result.get("result_state") != "SUCCESS":
        raise RuntimeError(f"hello_train ended {result}")
    state.submitted_jobs[key] = record


def _prune(state: RunState, brief: Brief, root: Path, backend: object) -> None:
    key = f"{state.candidate_id}/{state.rung}/{state.seed}"
    record = state.submitted_jobs[key]
    found = backend.find_mlflow_run(  # type: ignore[attr-defined]
        tags={
            "candidate_id": state.candidate_id,
            "rung": state.rung,
            "seed": str(state.seed),
        }
    )
    metric = None if found is None else found.get("metric")
    mlflow_run_id = None if found is None else found.get("mlflow_run_id")
    wall_min = None
    if record.get("submitted_at") and record.get("finished_at"):
        wall_min = round((record["finished_at"] - record["submitted_at"]) / 60, 3)
    if wall_min is None:
        charged = float(record.get("dbu_est") or 0.0)
    else:
        charged = estimate_dbu(wall_min)
    if not record.get("pruned"):
        if not _ledger_has(root, state):
            claim = brief.candidate.claim if brief.candidate else "logistic regression"
            append(
                root,
                ledger_row(
                    run_id=state.run_id,
                    node_id=state.candidate_id,
                    family=state.family,
                    claim=claim,
                    rung=state.rung,
                    seed=state.seed,
                    metric=float(metric) if metric is not None else None,
                    dbu=charged,
                    wall_min=wall_min,
                    git_sha=state.git_sha,
                    mlflow_run_id=mlflow_run_id,
                    job_run_id=record.get("job_run_id"),
                ),
            )
            state.spent_dbu += charged
        record["pruned"] = True
    record["mlflow_run_id"] = mlflow_run_id or record.get("mlflow_run_id")
    state.submitted_jobs[key] = record


def _ledger_has(root: Path, state: RunState) -> bool:
    return any(
        row.run_id == state.run_id
        and row.node_id == state.candidate_id
        and row.rung == state.rung
        and row.seed == state.seed
        for row in read_rows(root)
    )


def _new_state(brief: Brief, run_id: str, brief_path: Path, root: Path) -> RunState:
    pinned = brief.candidate
    return RunState(
        run_id=run_id,
        brief_path=str(brief_path),
        stage="PLAN",
        round=0,
        live_nodes=[] if pinned is None else [pinned.id],
        approval_mode=brief.approval_mode,
        k=brief.k,
        budget_dbu=brief.budget.total_dbu,
        candidate_id=pinned.id if pinned else "c00",
        family=pinned.family if pinned else "linear",
        rung=pinned.rung if pinned else "R0",
        seed=pinned.seed if pinned else 0,
        batch=pinned.batch if pinned else "toy",
        git_sha=_git_sha(root),
    )


def _git_sha(root: Path) -> str:
    override = os.environ.get("AMT_GIT_SHA")
    if override:
        return override
    for candidate in (root, *root.parents):
        if (candidate / ".git").exists():
            completed = subprocess.run(
                ["git", "rev-parse", "HEAD"],
                cwd=candidate,
                check=False,
                capture_output=True,
                text=True,
            )
            if completed.returncode == 0:
                return completed.stdout.strip()
    return "unknown"


def _load_state(path: Path) -> RunState:
    return RunState.model_validate(json.loads(path.read_text()))


def _save(path: Path, state: RunState) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(state.model_dump_json(indent=2) + "\n")
    temporary.replace(path)


def _stop_requested(root: Path, run_id: str) -> bool:
    return (root / "runs" / run_id / "STOP").is_file()


def _write_escalate(root: Path, state: RunState, reason: str) -> None:
    path = root / "runs" / state.run_id / "ESCALATE.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(f"# Escalate\n\n{reason}\n")


def _write_digest(root: Path, state: RunState) -> None:
    path = root / "runs" / state.run_id / "round_0.md"
    key = f"{state.candidate_id}/{state.rung}/{state.seed}"
    record = state.submitted_jobs.get(key, {})
    path.write_text(
        "# Round 0\n\n"
        f"Candidate `{state.candidate_id}` family `{state.family}` "
        f"job `{record.get('job_run_id')}` mlflow `{record.get('mlflow_run_id')}`.\n\n"
        f"Spent {state.spent_dbu} DBU (estimated).\n"
    )
