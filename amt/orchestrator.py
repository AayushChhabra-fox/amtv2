"""Resumable toy loop: proposal, critic, logistic candidate, smoke, hello_train, pruner."""

from __future__ import annotations

import json
import os
import subprocess
import time
from pathlib import Path
from typing import Any

import yaml

from amt.agents import coerce_role_output, run_role
from amt.budget import (
    BudgetExceeded,
    assert_can_spend,
    estimate_dbu,
    inflight_dbu,
    node_committed_dbu,
)
from amt.config import effective_concurrency, load_settings
from amt.confirm import ConfirmationSubmission, confirmation_submissions
from amt.jobs_backend import FakeJobs, make_backend
from amt.ledger import append, read_rows
from amt.preflight import preflight_candidate
from amt.pruner import RunView, commit_rung, decide, ledger_row
from amt.scale import (
    finish_run_report,
    full_budget_block_reason,
    passing_small_receipt,
    render_phase2_entry,
    scale_block_reason,
    stats_summary,
    write_agent_stats,
    write_calibration_claim,
)
from amt.schemas import (
    AgentRevision,
    Brief,
    Candidate,
    DeltaVsBaseline,
    NodeWork,
    ProposalBatch,
    RunState,
    VerdictBatch,
    load_brief,
    parse_agent_output,
)
from amt.signoff import SignoffError, brief_key, search_block_reason
from amt.snapshot import SnapshotError, upload_candidate
from amt.submit import (
    baselines_required,
    lgbm_baseline_view,
    missing_baseline_message,
    train_params,
)

# Caps from section 3.4. A schema failure is returned once; the next one escalates.
MAX_DEBUG_ATTEMPTS = 3
MAX_OUTER_RETRIES = 2
MAX_ROUNDS = 4


def run_brief(
    brief_path: Path,
    *,
    root: Path | None = None,
    fresh: bool = False,
    run_id: str | None = None,
    approval_mode: str | None = None,
    scale: str | None = None,
) -> RunState:
    root = root or Path.cwd()
    brief = load_brief(brief_path)
    if scale:
        brief.scale = scale  # type: ignore[assignment]
    conflict = scale == "calibration" and approval_mode not in {None, "every_stage"}
    if approval_mode and not conflict and brief.scale != "calibration":
        brief.approval_mode = approval_mode  # type: ignore[assignment]
    if brief.scale == "calibration":
        brief.k = 1
        brief.approval_mode = "every_stage"
    resolved_id = run_id or brief.run_id or brief.name or brief_path.stem
    state_path = root / "runs" / resolved_id / "state.json"
    if fresh and state_path.is_file():
        state_path.unlink()
    if state_path.is_file():
        state = _load_state(state_path)
        if approval_mode and state.scale != "calibration" and not conflict:
            state.approval_mode = brief.approval_mode
    else:
        state = _new_state(brief, resolved_id, brief_path, root)
    _apply_scale(brief, state)
    if state.stage in {"REPORT", "ESCALATE"}:
        return state
    if conflict:
        return _stop(state_path, state, root, "G3 calibration requires approval_mode every_stage.")
    if brief.candidate is not None and state.scale != "search":
        return _stop(
            state_path,
            state,
            root,
            "Scale runs require a search brief with no pinned candidate.",
        )
    blocked = search_block_reason(root, brief, brief_path)
    if blocked:
        return _stop(state_path, state, root, blocked)
    scaled = scale_block_reason(
        root,
        scale=state.scale,
        run_id=state.run_id,
        k=state.k,
        approval_mode=state.approval_mode,
    )
    if scaled:
        return _stop(state_path, state, root, scaled)
    if state.scale == "calibration":
        try:
            write_calibration_claim(root, state.run_id, brief_key(brief, brief_path))
        except SignoffError as exc:
            return _stop(state_path, state, root, str(exc))
    if state.scale == "full":
        state = _enter_phase2(state, root)
        _save(state_path, state)
        return state
    from amt.doctor import cheap

    problems = cheap()
    if problems:
        return _stop(state_path, state, root, "; ".join(problems))
    backend = make_backend(root)
    while state.stage not in {"REPORT", "ESCALATE"}:
        if _stop_requested(root, state.run_id):
            state.stop_requested = True
            state.stage = "ESCALATE"
            _save(state_path, state)
            _write_escalate(root, state, "stop file present")
            return state
        if state.awaiting_approval:
            if not _approved(root, state.run_id):
                return state
            state.awaiting_approval = False
            state.approval_gate = None
            _consume_approval(root, state.run_id)
            _save(state_path, state)
        finished = state.stage
        if brief.candidate is not None:
            state = _step(state, brief, root, backend)
        else:
            state = _search_step(state, brief, root, backend)
        _save(state_path, state)
        if os.environ.get("AMT_STOP_AFTER") == finished:
            return state
        if state.awaiting_approval:
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
        preflight_candidate(root / "candidates" / state.candidate_id)
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


def _agent_payload(output: Any) -> Any:
    """Drop the token envelope so planner and critic objects match their schemas."""

    payload = coerce_role_output(output)
    if isinstance(payload, dict) and "usage" in payload:
        return {key: value for key, value in payload.items() if key != "usage"}
    return payload


def _structured_role_output(role: str, output: Any) -> bool:
    if role not in {"planner", "critic"}:
        return False
    if isinstance(output, list):
        return True
    if not isinstance(output, dict):
        return False
    if role == "planner":
        return "candidates" in output or "proposals" in output
    return "verdicts" in output or "decision" in output


def _reject_invalid_output(state: RunState, role: str, output: Any, root: Path) -> RunState | None:
    output = _agent_payload(output)
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


def _others_inflight(root: Path, run_id: str) -> bool:
    runs = root / "runs"
    if not runs.is_dir():
        return False
    for path in runs.glob("*/state.json"):
        if path.parent.name == run_id:
            continue
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            continue
        if payload.get("stage") not in {"REPORT", "ESCALATE"}:
            return True
    return False


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
    job_run_id = backend.submit(  # type: ignore[attr-defined]
        job="hello_train",
        job_key=key,
        params=params,
        idempotency_token=token,
    )
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
    if pinned is None:
        return RunState(
            run_id=run_id,
            brief_path=str(brief_path),
            stage="PLAN",
            round=1,
            live_nodes=[],
            approval_mode=brief.approval_mode,
            scale=brief.scale,
            k=brief.k,
            budget_dbu=brief.budget.total_dbu,
            candidate_id="pending",
            family="search",
            rung="R1",
            seed=0,
            batch=brief.name or brief_path.stem,
            git_sha=_git_sha(root),
            round_operator="draft",
        )
    return RunState(
        run_id=run_id,
        brief_path=str(brief_path),
        stage="PLAN",
        round=0,
        live_nodes=[] if pinned is None else [pinned.id],
        approval_mode=brief.approval_mode,
        scale=brief.scale,
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


def _stop(state_path: Path, state: RunState, root: Path, reason: str) -> RunState:
    state.stage = "ESCALATE"
    _save(state_path, state)
    _write_escalate(root, state, reason)
    return state


def _apply_scale(brief: Brief, state: RunState) -> None:
    """Keep the in-memory brief aligned with the run that is already on disk."""

    brief.scale = state.scale  # type: ignore[assignment]
    if state.scale == "calibration":
        brief.k = 1
        brief.approval_mode = "every_stage"
        state.k = 1
        state.approval_mode = "every_stage"
    elif state.scale == "small":
        brief.k = state.k
        brief.approval_mode = state.approval_mode


def _enter_phase2(state: RunState, root: Path) -> RunState:
    reason = full_budget_block_reason(root)
    if reason:
        state.stage = "ESCALATE"
        _write_escalate(root, state, reason)
        return state
    receipt = passing_small_receipt(root)
    if receipt is None:
        state.stage = "ESCALATE"
        _write_escalate(root, state, "Full budget is not ready. No passing K=4 run is recorded.")
        return state
    path = root / "runs" / state.run_id / "phase2_entry.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(render_phase2_entry(state.run_id, receipt), encoding="utf-8")
    state.stage = "REPORT"
    return state


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


def operator_for_round(round_number: int, *, plateau: bool, lineage_count: int) -> str:
    """Operator for a round. Round 4 combines two lineages; a plateau forks to a new draft."""

    if round_number <= 1:
        return "draft"
    if round_number > MAX_ROUNDS:
        return "confirm"
    if round_number == MAX_ROUNDS:
        return "combine" if lineage_count >= 2 else "confirm"
    if plateau:
        return "draft"
    return "improve"


def fork_on_plateau(gains: list[float], min_effect: float) -> bool:
    """True when the last two improvements each sit below the noise floor."""

    if len(gains) < 2:
        return False
    return gains[-1] < min_effect and gains[-2] < min_effect


def _search_step(state: RunState, brief: Brief, root: Path, backend: object) -> RunState:
    stage = state.stage
    if stage == "PLAN":
        return _search_plan(state, brief, root)
    if stage == "CRITIQUE":
        return _search_critique(state, brief, root)
    if stage == "SPEC":
        return _search_spec(state, root)
    if stage == "IMPLEMENT":
        return _search_implement(state, brief, root)
    if stage == "PREFLIGHT":
        return _search_preflight(state, brief, root)
    if stage == "DEBUG":
        return _search_debug(state, brief, root)
    if stage == "SUBMIT":
        return _search_submit(state, brief, root, backend)
    if stage == "WAIT":
        return _search_wait(state, root, backend)
    if stage == "PRUNE":
        return _search_prune(state, brief, root, backend)
    if stage == "CONFIRM":
        return _search_confirm(state, brief, root, backend)
    if stage == "REPORT":
        _write_report(root, state, brief)
        return state
    raise RuntimeError(f"unknown stage {stage}")


def _search_plan(state: RunState, brief: Brief, root: Path) -> RunState:
    feedback = _feedback_text(root, state)
    parents = _allowed_parent_list(state)
    output_path = _proposals_path(root, state)
    output = run_role(
        "planner",
        brief,
        root,
        state.run_id,
        operator=state.round_operator,
        parent_ids=parents,
        output_path=output_path,
        note=_plan_note(state, feedback),
    )
    payload = _agent_payload(output)
    if not _structured_role_output("planner", payload):
        return _schema_retry(state, root, "planner output had no candidates")
    state, revision = apply_agent_output(state, "planner", payload)
    if revision is not None:
        return _schema_retry(state, root, revision.message, already_counted=True)
    parsed = parse_agent_output("planner", payload)
    batch = parsed.value
    if not isinstance(batch, ProposalBatch):
        return _schema_retry(state, root, "planner output did not parse")
    problems = _proposal_problems(state, batch)
    if problems:
        return _schema_retry(state, root, "\n".join(problems))
    _write_proposals(output_path, batch)
    _clear_feedback(root, state)
    state.stage = "CRITIQUE"
    return _maybe_pause(state, root, "PLAN")


def _search_critique(state: RunState, brief: Brief, root: Path) -> RunState:
    output_path = root / "runs" / state.run_id / f"round_{state.round}_verdicts.json"
    output = run_role(
        "critic",
        brief,
        root,
        state.run_id,
        operator=state.round_operator,
        parent_ids=_allowed_parent_list(state),
        output_path=output_path,
        inputs=[str(_proposals_path(root, state))],
    )
    payload = _agent_payload(output)
    if not _structured_role_output("critic", payload):
        return _schema_retry(state, root, "critic output had no verdicts")
    state, revision = apply_agent_output(state, "critic", payload)
    if revision is not None:
        return _schema_retry(state, root, revision.message, already_counted=True)
    parsed = parse_agent_output("critic", payload)
    verdicts = parsed.value
    if not isinstance(verdicts, VerdictBatch):
        return _schema_retry(state, root, "critic output did not parse")
    output_path.write_text(verdicts.model_dump_json(indent=2) + "\n", encoding="utf-8")
    revises = [item for item in verdicts.verdicts if item.decision == "REVISE"]
    if revises:
        state.critic_revisions += 1
        message = "\n".join(f"{item.candidate_id}: {item.reason}" for item in revises)
        if state.critic_revisions > 1:
            return _escalate(state, root, f"critic revision cap\n{message}")
        _write_note(root, state, message)
        state.stage = "PLAN"
        return _maybe_pause(state, root, "CRITIQUE")
    batch = _load_proposals(root, state)
    by_id = {item.candidate_id: item for item in verdicts.verdicts}
    passed = []
    for candidate in batch.candidates:
        verdict = by_id.get(candidate.id)
        if verdict is None or verdict.decision != "PASS":
            continue
        if _duplicate(candidate, passed, root):
            continue
        passed.append(candidate)
    passed = _shortlist(state, passed)
    if not passed:
        return _escalate(state, root, "critic dropped every proposal")
    needed = min(3, state.k)
    families = {candidate.family for candidate in passed}
    if state.round == 1 and state.round_operator == "draft" and len(families) < needed:
        state.critic_revisions += 1
        message = f"shortlist has {len(families)} families; need at least {needed}"
        if state.critic_revisions > 1:
            return _escalate(state, root, f"critic revision cap\n{message}")
        _write_note(root, state, message)
        state.stage = "PLAN"
        return _maybe_pause(state, root, "CRITIQUE")
    _clear_feedback(root, state)
    rung = _rungs_for(state.round_operator)[0]
    state.rung_index = 0
    state.nodes = [
        node
        for node in state.nodes
        if not (node.search_round == state.round and node.status == "pending")
    ]
    combine_parents = _combine_parents(state) if state.round_operator == "combine" else []
    for candidate in passed:
        parents = (
            list(combine_parents)
            if combine_parents
            else ([candidate.parent_id] if candidate.parent_id else [])
        )
        state.nodes.append(
            NodeWork(
                node_id=candidate.id,
                family=candidate.family,
                operator=candidate.operator,
                parent_id=candidate.parent_id,
                parents=parents,
                claim=candidate.claim,
                change=candidate.change,
                fit_argument=candidate.fit_argument,
                cheap_test=candidate.cheap_test,
                kill_if=candidate.kill_if,
                est_cost=candidate.est_cost,
                device=candidate.device,
                rung=rung,  # type: ignore[arg-type]
                search_round=state.round,
                status="pending",
            )
        )
    state.live_nodes = [candidate.id for candidate in passed]
    state.candidate_id = passed[0].id
    state.family = passed[0].family
    state.rung = rung  # type: ignore[assignment]
    state.stage = "SPEC"
    return _maybe_pause(state, root, "CRITIQUE")


def _search_spec(state: RunState, root: Path) -> RunState:
    for node in _round_nodes(state):
        _write_node_spec(root, node)
    state.stage = "IMPLEMENT"
    return _maybe_pause(state, root, "SPEC")


def _search_implement(state: RunState, brief: Brief, root: Path) -> RunState:
    node = next((item for item in _round_nodes(state) if item.status == "pending"), None)
    if node is None:
        state.stage = "PREFLIGHT"
        return _maybe_pause(state, root, "IMPLEMENT")
    model_path = root / "candidates" / node.node_id / "model.py"
    run_role(
        "implementer",
        brief,
        root,
        state.run_id,
        node_id=node.node_id,
        operator=node.operator,
        parent_ids=node.parents,
        output_path=model_path,
    )
    if not model_path.is_file():
        return _fail_preflight(state, root, node, "implementer did not write model.py")
    node.status = "implemented"
    if any(item.status == "pending" for item in _round_nodes(state)):
        return state
    state.stage = "PREFLIGHT"
    return _maybe_pause(state, root, "IMPLEMENT")


def _search_preflight(state: RunState, brief: Brief, root: Path) -> RunState:
    node = next(
        (item for item in _round_nodes(state) if item.status in {"implemented", "debugging"}),
        None,
    )
    if node is None:
        if any(item.status == "ready" for item in _round_nodes(state)):
            state.stage = "SUBMIT"
            return _maybe_pause(state, root, "PREFLIGHT")
        return _escalate(state, root, "no candidate passed preflight")
    directory = root / "candidates" / node.node_id
    if node.device == "gpu" and not load_settings(root).gpu_available:
        return _fail_preflight(
            state,
            root,
            node,
            "gpu proposals are dropped: no GPU cluster or node type",
        )
    try:
        preflight_candidate(directory)
    except Exception as exc:
        return _fail_preflight(state, root, node, str(exc))
    review_path = directory / "review.md"
    run_role(
        "reviewer",
        brief,
        root,
        state.run_id,
        node_id=node.node_id,
        operator=node.operator,
        output_path=review_path,
        inputs=[str(directory / "spec.md"), str(directory / "model.py")],
    )
    verdict = _review_verdict(review_path)
    if verdict != "PASS":
        return _fail_preflight(state, root, node, f"review {verdict or 'missing'}")
    node.status = "ready"
    node.preflight = "PASS"
    node.review = "PASS"
    if any(item.status in {"implemented", "debugging", "pending"} for item in _round_nodes(state)):
        return state
    state.stage = "SUBMIT"
    return _maybe_pause(state, root, "PREFLIGHT")


def _search_debug(state: RunState, brief: Brief, root: Path) -> RunState:
    node = next((item for item in _round_nodes(state) if item.status == "debugging"), None)
    if node is None:
        state.stage = "PREFLIGHT"
        return state
    model_path = root / "candidates" / node.node_id / "model.py"
    run_role(
        "implementer",
        brief,
        root,
        state.run_id,
        node_id=node.node_id,
        operator="debug",
        parent_ids=node.parents,
        output_path=model_path,
        note="Fix the preflight failure recorded in the fix chain.",
    )
    node.status = "implemented"
    state.stage = "PREFLIGHT"
    return _maybe_pause(state, root, "DEBUG")


def _search_submit(state: RunState, brief: Brief, root: Path, backend: object) -> RunState:
    ready = [node for node in _round_nodes(state) if node.status == "ready"]
    if not ready:
        if any(node.status == "submitted" for node in _round_nodes(state)):
            state.stage = "WAIT"
            return _maybe_pause(state, root, "SUBMIT")
        return _escalate(state, root, "nothing to submit")
    cap = effective_concurrency(brief)
    if cap is not None and _inflight_unfinished(state) >= cap:
        state.stage = "WAIT"
        return state
    node = ready[0]
    key = _job_key(node)
    record = dict(state.submitted_jobs.get(key) or {})
    if not record.get("job_run_id"):
        estimate = float(node.est_cost)
        committed = node_committed_dbu(read_rows(root), state.run_id, node.node_id)
        try:
            assert_can_spend(
                brief,
                state.spent_dbu + inflight_dbu(state.submitted_jobs),
                estimate,
                candidate_spent_dbu=committed + inflight_dbu(state.submitted_jobs, node.node_id),
            )
        except BudgetExceeded as exc:
            return _escalate(state, root, str(exc))
        if baselines_required(backend):
            missing = missing_baseline_message(root, brief, node.rung)
            if missing:
                return _escalate(state, root, missing)
        try:
            model = upload_candidate(root, state.run_id, node.node_id, state.git_sha)
        except SnapshotError as exc:
            return _escalate(state, root, str(exc))
        params = train_params(
            root=root,
            brief=brief,
            node_id=node.node_id,
            parent_id=node.parent_id,
            family=node.family,
            operator=node.operator,
            rung=node.rung,
            seed=node.seed,
            batch=state.batch,
            run_id=state.run_id,
            git_sha=state.git_sha,
            model=model,
            device=node.device,
        )
        job_run_id = backend.submit(  # type: ignore[attr-defined]
            job="train",
            job_key=key,
            params=params,
            idempotency_token=f"{state.run_id}:{key}",
        )
        record.update({"job_run_id": job_run_id, "submitted_at": time.time(), "dbu_est": estimate})
        state.submitted_jobs[key] = record
    node.status = "submitted"
    state.candidate_id = node.node_id
    state.family = node.family
    state.rung = node.rung
    if any(item.status == "ready" for item in _round_nodes(state)):
        if cap is None or _inflight_unfinished(state) < cap:
            return state
    state.stage = "WAIT"
    return _maybe_pause(state, root, "SUBMIT")


def _search_wait(state: RunState, root: Path, backend: object) -> RunState:
    pending: list[tuple[NodeWork, str, dict[str, Any]]] = []
    for node in _round_nodes(state):
        if node.status != "submitted":
            continue
        key = _job_key(node)
        record = dict(state.submitted_jobs.get(key) or {})
        if record.get("result_state"):
            continue
        pending.append((node, key, record))
    if not pending:
        if any(node.status == "ready" for node in _round_nodes(state)):
            state.stage = "SUBMIT"
            return state
        state.stage = "PRUNE"
        return _maybe_pause(state, root, "WAIT")
    node, key, record = pending[0]
    result = backend.wait(record["job_run_id"])  # type: ignore[attr-defined]
    record["result_state"] = result.get("result_state")
    record["finished_at"] = time.time()
    state.submitted_jobs[key] = record
    if result.get("result_state") != "SUCCESS":
        node.status = "failed"
    return state


def _search_prune(state: RunState, brief: Brief, root: Path, backend: object) -> RunState:
    schedule = _rungs_for(state.round_operator)
    rung = schedule[state.rung_index]
    views: list[RunView] = []
    charges: dict[str, float] = {}
    for node in _round_nodes(state):
        if node.rung != rung or node.status not in {"submitted", "failed"}:
            continue
        key = _job_key(node)
        record = dict(state.submitted_jobs.get(key) or {})
        found = backend.find_mlflow_run(  # type: ignore[attr-defined]
            tags={
                "run_id": state.run_id,
                "node_id": node.node_id,
                "rung": node.rung,
                "seed": str(node.seed),
            }
        )
        charged = _job_charge(record)
        charges[node.node_id] = charged
        delta = _delta_from_found(found)
        metric = None if found is None else found.get("metric")
        views.append(
            RunView(
                node_id=node.node_id,
                family=node.family,
                seed=node.seed,
                rung=rung,  # type: ignore[arg-type]
                split="search_val",
                operator=node.operator,
                claim=node.claim or node.node_id,
                parent_id=node.parent_id,
                round=state.round,
                metric=None if metric is None else float(metric),
                delta=delta,
                preflight="FAIL" if node.status == "failed" else (node.preflight or "PASS"),
                review="FAIL" if node.status == "failed" else (node.review or "PASS"),
                dbu=charged,
                git_sha=state.git_sha,
                mlflow_run_id=None if found is None else found.get("mlflow_run_id"),
                job_run_id=record.get("job_run_id"),
            )
        )
    if not views:
        return _escalate(state, root, f"no runs to prune at {rung}")
    baseline = lgbm_baseline_view(root, brief, rung, state.git_sha)
    try:
        decision = decide(
            rung,
            views,
            [baseline],
            brief,
            shared_cluster=load_settings(root).fixed_cluster,
        )
    except Exception as exc:
        return _escalate(state, root, f"pruner failed: {exc}")
    commit_rung(root, state.run_id, views, decision)
    by_id = {item.node_id: item for item in decision.decisions}
    for node in _round_nodes(state):
        item = by_id.get(node.node_id)
        if item is None:
            continue
        if item.delta is not None:
            node.delta_point = item.delta.point
            node.delta_lo = item.delta.ci_lo
            node.delta_hi = item.delta.ci_hi
        node.status = "kept" if item.status in {"kept", "promoted"} else item.status  # type: ignore[assignment]
        _mark_pruned(state, node, charges.get(node.node_id, 0.0))
    if decision.promoted:
        state.promoted = list(decision.promoted)
    state.live_nodes = list(decision.survivors)
    if decision.survivors and state.rung_index + 1 < len(schedule):
        state.rung_index += 1
        next_rung = schedule[state.rung_index]
        for node in _round_nodes(state):
            if node.node_id in decision.survivors:
                node.status = "ready"
                node.rung = next_rung  # type: ignore[assignment]
        state.stage = "SUBMIT"
        return _maybe_pause(state, root, "PRUNE")
    _record_gains(state)
    _write_round_digest(root, state, decision)
    return _advance_round(state, brief, root, bool(decision.survivors))


def _search_confirm(state: RunState, brief: Brief, root: Path, backend: object) -> RunState:
    if not state.promoted:
        state.stage = "REPORT"
        _write_report(root, state, brief)
        return state
    if not isinstance(backend, FakeJobs) and _others_inflight(root, state.run_id):
        return _escalate(state, root, "confirmation waits until other AMT runs finish")
    families = {node.node_id: node.family for node in state.nodes}
    submissions = request_confirmation(
        state.promoted,
        batch=state.batch,
        spec=brief.resolved_dataset_spec() or "",
        run_id=state.run_id,
        families=families,
        git_sha=state.git_sha,
    )
    for submission in submissions:
        key = f"confirm/{submission.candidate_id}/{submission.seed}"
        record = dict(state.submitted_jobs.get(key) or {})
        if not record.get("job_run_id"):
            estimate = 0.1
            try:
                assert_can_spend(
                    brief,
                    state.spent_dbu + inflight_dbu(state.submitted_jobs),
                    estimate,
                )
            except BudgetExceeded as exc:
                return _escalate(state, root, str(exc))
            params = submission.job_params()
            if submission.candidate_id != "lgbm_reference":
                try:
                    params["model"] = upload_candidate(
                        root, state.run_id, submission.candidate_id, state.git_sha
                    )
                except SnapshotError as exc:
                    return _escalate(state, root, str(exc))
            from amt.tokens import issue_token

            params["confirm_token"] = issue_token(root)
            job_run_id = backend.submit(  # type: ignore[attr-defined]
                job="confirm",
                job_key=key,
                params=params,
                idempotency_token=f"{state.run_id}:{key}",
            )
            record.update(
                {"job_run_id": job_run_id, "submitted_at": time.time(), "dbu_est": estimate}
            )
            state.submitted_jobs[key] = record
            return state
        if not record.get("result_state"):
            result = backend.wait(record["job_run_id"])  # type: ignore[attr-defined]
            record["result_state"] = result.get("result_state")
            record["finished_at"] = time.time()
            if not record.get("pruned"):
                state.spent_dbu += float(record.get("dbu_est") or 0.0)
                record["pruned"] = True
            state.submitted_jobs[key] = record
            if result.get("result_state") != "SUCCESS":
                return _escalate(state, root, f"confirm job failed: {result}")
            return state
    state.stage = "REPORT"
    _write_report(root, state, brief)
    return state


def _advance_round(state: RunState, brief: Brief, root: Path, survived: bool) -> RunState:
    if state.scale == "calibration":
        state.stage = "REPORT"
        _write_report(root, state, brief)
        return state
    if not survived:
        state.outer_retries += 1
        if state.outer_retries > MAX_OUTER_RETRIES:
            return _escalate(state, root, "outer retry cap")
        state.nodes = [node for node in state.nodes if node.search_round != state.round]
        state.rung_index = 0
        state.stage = "PLAN"
        return _maybe_pause(state, root, "PRUNE")
    state.outer_retries = 0
    kept = [node.node_id for node in state.nodes if node.status == "kept"]
    nxt = operator_for_round(
        state.round + 1,
        plateau=fork_on_plateau(list(state.improvement_gains), brief.min_effect),
        lineage_count=len(set(kept)),
    )
    if brief.max_rounds is not None and state.round >= brief.max_rounds:
        nxt = "confirm"
    if nxt == "confirm":
        if not state.promoted:
            state.stage = "REPORT"
            _write_report(root, state, brief)
            return state
        state.stage = "CONFIRM"
        return _maybe_pause(state, root, "PRUNE")
    state.round += 1
    state.round_operator = nxt  # type: ignore[assignment]
    state.rung_index = 0
    state.stage = "PLAN"
    return _maybe_pause(state, root, "PRUNE")


def _record_gains(state: RunState) -> None:
    if state.round_operator != "improve":
        return
    for node in _round_nodes(state):
        if node.status != "kept" or node.delta_point is None or not node.parent_id:
            continue
        parent = _find_node(state, node.parent_id)
        parent_point = 0.0 if parent is None or parent.delta_point is None else parent.delta_point
        state.improvement_gains.append(node.delta_point - parent_point)


def _fail_preflight(state: RunState, root: Path, node: NodeWork, reason: str) -> RunState:
    _append_debug(root, node, reason)
    if node.debug_attempts >= MAX_DEBUG_ATTEMPTS:
        node.status = "failed"
        node.preflight = "FAIL"
        active = {"pending", "implemented", "ready", "debugging"}
        others = [
            item
            for item in _round_nodes(state)
            if item.node_id != node.node_id and item.status in active
        ]
        if not others:
            return _escalate(
                state,
                root,
                f"debug cap after {MAX_DEBUG_ATTEMPTS} attempts\n{reason}",
            )
        if any(item.status != "pending" for item in others):
            state.stage = "PREFLIGHT"
        else:
            state.stage = "IMPLEMENT"
        return state
    node.debug_attempts += 1
    node.status = "debugging"
    state.candidate_id = node.node_id
    state.stage = "DEBUG"
    return _maybe_pause(state, root, "PREFLIGHT")


def _maybe_pause(state: RunState, root: Path, completed: str) -> RunState:
    pause = state.approval_mode == "every_stage" or (
        state.approval_mode == "round_shortlist" and completed == "CRITIQUE"
    )
    if not pause:
        return state
    state.awaiting_approval = True
    state.approval_gate = completed
    path = root / "runs" / state.run_id / "APPROVAL.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        f"# Approval\n\nStage `{completed}` finished. Mode `{state.approval_mode}`. "
        f"Next stage is `{state.stage}`.\n\n"
        f"Resume with `amt approve --run-id {state.run_id}` and then `amt run`.\n",
        encoding="utf-8",
    )
    return state


def _schema_retry(
    state: RunState,
    root: Path,
    message: str,
    *,
    already_counted: bool = False,
) -> RunState:
    if not already_counted:
        state.revisions += 1
    if "counts as 1 revision" in message:
        text = message
    else:
        text = f"This counts as 1 revision.\n{message}"
    _write_note(root, state, text)
    if state.revisions > 1:
        return _escalate(state, root, text)
    return state


def _escalate(state: RunState, root: Path, reason: str) -> RunState:
    state.stage = "ESCALATE"
    _write_escalate(root, state, reason)
    return state


def _proposal_problems(state: RunState, batch: ProposalBatch) -> list[str]:
    allowed = set(_allowed_parent_list(state))
    problems: list[str] = []
    seen: set[tuple[str, str, str]] = set()
    for candidate in batch.candidates:
        if candidate.operator != state.round_operator:
            problems.append(
                f"{candidate.id} operator is {candidate.operator}; "
                f"this round requires {state.round_operator}"
            )
        if state.round_operator != "draft" and candidate.parent_id not in allowed:
            problems.append(f"{candidate.id} parent is not one of {sorted(allowed)}")
        key = _feature_key(candidate.family, candidate.claim, candidate.change)
        if key in seen:
            problems.append(f"{candidate.id} repeats a proposal already in this batch")
        seen.add(key)
    if state.scale == "small" and state.round == 1 and state.round_operator == "draft":
        if len(batch.candidates) < 8:
            problems.append(
                f"round 1 on a K=4 run must propose 8 drafts; got {len(batch.candidates)}"
            )
    if state.scale == "small" and state.round_operator == "improve":
        for parent in _allowed_parent_list(state):
            count = sum(1 for candidate in batch.candidates if candidate.parent_id == parent)
            if count < 3:
                problems.append(f"propose 3 improvements of {parent}; found {count}")
    return problems


def _duplicate(candidate: Any, accepted: list[Any], root: Path) -> bool:
    key = _feature_key(candidate.family, candidate.claim, candidate.change)
    for other in accepted:
        if _feature_key(other.family, other.claim, other.change) == key:
            return True
    claim = candidate.claim.strip().lower()
    for row in read_rows(root):
        if row.family == candidate.family and row.claim.strip().lower() == claim:
            return True
    return False


def _feature_key(family: str, claim: str, change: str | None) -> tuple[str, str, str]:
    return (family.strip().lower(), claim.strip().lower(), (change or "").strip().lower())


def _allowed_parent_list(state: RunState) -> list[str]:
    if state.round_operator == "draft":
        return []
    if state.round_operator == "combine":
        return _combine_parents(state)
    previous = state.round - 1
    kept = [node for node in state.nodes if node.status == "kept" and node.search_round == previous]
    if state.scale == "small":
        return _top_lineages(kept, limit=2)
    return [node.node_id for node in kept]


def _combine_parents(state: RunState) -> list[str]:
    kept = [node for node in state.nodes if node.status == "kept"]
    kept.sort(key=lambda node: (-node.search_round, -(node.delta_point or 0.0), node.node_id))
    found: list[str] = []
    for node in kept:
        if node.node_id not in found:
            found.append(node.node_id)
        if len(found) == 2:
            break
    return found


def _round_nodes(state: RunState) -> list[NodeWork]:
    return [node for node in state.nodes if node.search_round == state.round]


def _find_node(state: RunState, node_id: str) -> NodeWork | None:
    for node in state.nodes:
        if node.node_id == node_id:
            return node
    return None


def _rungs_for(operator: str) -> list[str]:
    if operator == "draft":
        return ["R1", "R2", "R3"]
    if operator == "combine":
        return ["R3"]
    return ["R2", "R3"]


def _job_key(node: NodeWork) -> str:
    return f"{node.node_id}/{node.rung}/{node.seed}"


def _inflight_unfinished(state: RunState) -> int:
    count = 0
    for record in state.submitted_jobs.values():
        if record.get("pruned") or record.get("result_state") or not record.get("job_run_id"):
            continue
        count += 1
    return count


def _job_charge(record: dict[str, Any]) -> float:
    start = record.get("submitted_at")
    end = record.get("finished_at")
    if start is not None and end is not None:
        wall = (float(end) - float(start)) / 60.0
        if wall >= 1e-3:
            return estimate_dbu(wall)
    return float(record.get("dbu_est") or 0.0)


def _delta_from_found(found: dict[str, Any] | None) -> DeltaVsBaseline | None:
    if not found:
        return None
    if found.get("delta_lgbm_point") is not None:
        return DeltaVsBaseline(
            point=float(found["delta_lgbm_point"]),
            ci_lo=float(found["delta_lgbm_ci_lo"]),
            ci_hi=float(found["delta_lgbm_ci_hi"]),
        )
    if found.get("delta_point") is None:
        return None
    return DeltaVsBaseline(
        point=float(found["delta_point"]),
        ci_lo=float(found["delta_ci_lo"]),
        ci_hi=float(found["delta_ci_hi"]),
    )


def _mark_pruned(state: RunState, node: NodeWork, charged: float) -> None:
    key = _job_key(node)
    record = dict(state.submitted_jobs.get(key) or {})
    if record.get("pruned"):
        return
    state.spent_dbu += charged
    record["pruned"] = True
    state.submitted_jobs[key] = record


def _proposals_path(root: Path, state: RunState) -> Path:
    return root / "runs" / state.run_id / f"round_{state.round}_proposals.yaml"


def _write_proposals(path: Path, batch: ProposalBatch) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"candidates": [candidate.model_dump() for candidate in batch.candidates]}
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")


def _load_proposals(root: Path, state: RunState) -> ProposalBatch:
    return ProposalBatch.model_validate(yaml.safe_load(_proposals_path(root, state).read_text()))


def _write_node_spec(root: Path, node: NodeWork) -> None:
    directory = root / "candidates" / node.node_id
    directory.mkdir(parents=True, exist_ok=True)
    payload = {
        "id": node.node_id,
        "family": node.family,
        "operator": node.operator,
        "parent_id": node.parent_id,
        "claim": node.claim,
        "change": node.change,
        "fit_argument": node.fit_argument,
        "cheap_test": node.cheap_test,
        "kill_if": node.kill_if,
        "device": node.device,
    }
    (directory / "candidate.yaml").write_text(
        yaml.safe_dump(payload, sort_keys=False), encoding="utf-8"
    )
    parents = ", ".join(node.parents) if node.parents else "none"
    (directory / "spec.md").write_text(
        f"# {node.node_id}\n\n"
        f"Family: {node.family}\n\n"
        f"Operator: {node.operator}\n\n"
        f"Parents: {parents}\n\n"
        f"Claim: {node.claim}\n\n"
        f"Change: {node.change or 'new design'}\n\n"
        f"Fit argument: {node.fit_argument}\n\n"
        f"Cheap test: {node.cheap_test}\n\n"
        f"Kill if: {node.kill_if}\n\n"
        "Acceptance: fit and score use only the frame passed in. "
        "Scores are finite and match the frame length. "
        "Do not read holdout splits, tables, or files outside this directory.\n",
        encoding="utf-8",
    )


def _review_verdict(path: Path) -> str | None:
    if not path.is_file():
        return None
    verdict = None
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip().upper()
        if stripped.startswith("VERDICT:"):
            verdict = stripped.split(":", 1)[1].strip()
    return verdict


def _append_debug(root: Path, node: NodeWork, reason: str) -> None:
    path = root / "candidates" / node.node_id / "debug_log.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    existing = path.read_text(encoding="utf-8") if path.is_file() else "# debug log\n"
    attempt = node.debug_attempts + 1
    path.write_text(existing.rstrip() + f"\n\n## attempt {attempt}\n\n{reason}\n", encoding="utf-8")


def _plan_note(state: RunState, feedback: str) -> str:
    lines = [f"This is round {state.round}. Propose {state.round_operator} candidates."]
    parents = _allowed_parent_list(state)
    if parents:
        lines.append("Allowed parents: " + ", ".join(parents))
    if state.round_operator == "improve":
        lines.append("Each improvement is one atomic change. Set parent_id and change.")
    if state.round_operator == "combine":
        lines.append("Propose one combine of the two parent nodes.")
    if state.round_operator == "draft" and state.round == 1:
        lines.append("Prefer a simple design. No ensembles and no heavy tuning.")
    if state.scale == "calibration":
        lines.append("This is the one-time G3 calibration. Propose one draft.")
    if state.scale == "small" and state.round == 1 and state.round_operator == "draft":
        lines.append("Propose 8 drafts across families. The critic keeps 4, at least 3 families.")
    if state.scale == "small" and state.round_operator == "improve":
        lines.append(
            "Propose 3 atomic improvements for each allowed parent. The critic keeps 2 per parent."
        )
    if state.round_operator == "draft" and state.round > 1:
        lines.append("The lineage plateaued. Draft one new family.")
    if feedback:
        lines.append(feedback)
    return "\n".join(lines)


def _feedback_text(root: Path, state: RunState) -> str:
    path = root / "runs" / state.run_id / "agent_feedback.md"
    if not path.is_file():
        return ""
    return path.read_text(encoding="utf-8")


def _write_note(root: Path, state: RunState, message: str) -> None:
    path = root / "runs" / state.run_id / "agent_feedback.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(message, encoding="utf-8")


def _clear_feedback(root: Path, state: RunState) -> None:
    path = root / "runs" / state.run_id / "agent_feedback.md"
    path.unlink(missing_ok=True)


def _write_round_digest(root: Path, state: RunState, decision: Any) -> None:
    path = root / "runs" / state.run_id / f"round_{state.round}.md"
    rows = [
        row for row in read_rows(root) if row.run_id == state.run_id and row.round == state.round
    ]
    lines = [
        f"# Round {state.round}",
        "",
        f"Operator: {state.round_operator}",
        "",
        "## Proposals and verdicts",
    ]
    verdict_path = root / "runs" / state.run_id / f"round_{state.round}_verdicts.json"
    if verdict_path.is_file():
        lines.append(verdict_path.read_text(encoding="utf-8").strip())
    else:
        lines.append("(none)")
    lines.extend(["", "## Kills"])
    kills = [row for row in rows if row.status in {"killed", "failed"}]
    if not kills:
        lines.append("(none)")
    else:
        for row in kills:
            lines.append(f"- {row.node_id} {row.rung}: {row.kill_reason}")
    lines.extend(["", "## Survivors"])
    survivors = list(getattr(decision, "survivors", []) or [])
    if not survivors:
        lines.append("(none)")
    by_id = {item.node_id: item for item in getattr(decision, "decisions", [])}
    for node_id in survivors:
        item = by_id.get(node_id)
        delta = None if item is None else item.delta
        if delta is None:
            lines.append(f"- {node_id}")
        else:
            lines.append(
                f"- {node_id} delta {delta.point:.4f} CI [{delta.ci_lo:.4f}, {delta.ci_hi:.4f}]"
            )
    lines.extend(["", "## Spend", f"Spent {state.spent_dbu:.3f} / {state.budget_dbu:.3f} DBU", ""])
    lines.append(stats_summary(write_agent_stats(root, state)))
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def _write_report(root: Path, state: RunState, brief: Brief) -> None:
    path = root / "runs" / state.run_id / "report.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(finish_run_report(root, state, brief), encoding="utf-8")


def _shortlist(state: RunState, passed: list[Candidate]) -> list[Candidate]:
    """K=4 improvement rounds keep two children of each parent. Other rounds keep K."""

    if state.scale == "small" and state.round_operator == "improve":
        kept: list[Candidate] = []
        per_parent: dict[str, int] = {}
        for candidate in passed:
            parent = candidate.parent_id or ""
            if per_parent.get(parent, 0) >= 2:
                continue
            per_parent[parent] = per_parent.get(parent, 0) + 1
            kept.append(candidate)
            if len(kept) >= state.k:
                break
        return kept
    return passed[: state.k]


def _top_lineages(nodes: list[NodeWork], limit: int) -> list[str]:
    """Best kept node in each family, then the strongest families."""

    best: dict[str, NodeWork] = {}
    for node in nodes:
        current = best.get(node.family)
        if current is None or _lineage_rank(node) < _lineage_rank(current):
            best[node.family] = node
    ordered = sorted(best.values(), key=_lineage_rank)
    return [node.node_id for node in ordered[:limit]]


def _lineage_rank(node: NodeWork) -> tuple[float, str]:
    return (-(node.delta_point or 0.0), node.node_id)


def _approved(root: Path, run_id: str) -> bool:
    return (root / "runs" / run_id / "APPROVE").is_file()


def _consume_approval(root: Path, run_id: str) -> None:
    (root / "runs" / run_id / "APPROVE").unlink(missing_ok=True)
