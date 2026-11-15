"""Resumable search loop: plan, critique, implement, smoke, train, prune."""

from __future__ import annotations

import json
import os
import subprocess
import time
from pathlib import Path
from typing import Any, Literal

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
from amt.gates import (
    SignoffError,
    _spec_path,
    apply_locked_min_effect,
    brief_key,
    search_block_reason,
)
from amt.job_params import (
    baselines_required,
    lgbm_baseline_view,
    missing_baseline_message,
    train_params,
)
from amt.jobs_backend import FakeJobs, make_backend
from amt.ledger import read_rows
from amt.paths import candidate_dir
from amt.preflight import preflight_candidate
from amt.pruner import RunView, commit_rung, decide
from amt.reporting import log_progress, write_report, write_report_json, write_round_digest
from amt.scale import scale_block_reason, write_calibration_claim
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
from amt.search_policy import (
    C00_ID,
    allowed_parents,
    combine_parents,
    duplicate_candidate,
    fork_on_plateau,
    operator_for_round,
    random_metric_at,
    record_c00_promotion,
    rungs_for,
    seed_c00,
    shortlist,
    unscored_final_r3,
)
from amt.search_policy import (
    proposal_problems as _proposal_problems,
)
from amt.snapshot import SnapshotError, upload_candidate
from harness.data.contract import load_spec

# Caps from section 3.4. A schema failure is returned once; the next one escalates.
MAX_DEBUG_ATTEMPTS = 3
MAX_OUTER_RETRIES = 2
MAX_INFRA_RETRIES = 1


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
    apply_locked_min_effect(root, brief, brief_path)
    if state.scale == "calibration":
        try:
            write_calibration_claim(root, state.run_id, brief_key(brief, brief_path))
        except SignoffError as exc:
            return _stop(state_path, state, root, str(exc))
    from amt.doctor import cheap

    problems = cheap()
    if problems:
        return _stop(state_path, state, root, "; ".join(problems))
    backend = make_backend(root)
    if brief.candidate is None and state.scale != "calibration":
        seed_c00(root, state, brief)
        _save(state_path, state)
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
        log_progress(state)
        state = _search_step(state, brief, root, backend)
        _save(state_path, state)
        if os.environ.get("AMT_STOP_AFTER") == finished:
            return state
        if state.awaiting_approval:
            return state
    return state


def resume_brief(run_id: str, *, root: Path | None = None) -> RunState:
    """Continue a saved run. An escalated run restarts at the stage that stopped it."""

    root = root or Path.cwd()
    state_path = root / "runs" / run_id / "state.json"
    if not state_path.is_file():
        raise FileNotFoundError(f"no state for run {run_id}")
    state = _load_state(state_path)
    if state.stage == "ESCALATE":
        state.stage = state.escalated_from_stage or "PLAN"
        state.stop_requested = False
        _save(state_path, state)
    brief_path = Path(state.brief_path)
    if not brief_path.is_absolute():
        brief_path = root / brief_path
    return run_brief(brief_path, root=root, run_id=run_id)


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
        round=1,
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
    _remember_escalation(state)
    state.stage = "ESCALATE"
    _save(state_path, state)
    _write_escalate(root, state, reason)
    write_report_json(root, state)
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
        write_report(root, state, brief)
        return state
    raise RuntimeError(f"unknown stage {stage}")


def _search_plan(state: RunState, brief: Brief, root: Path) -> RunState:
    feedback = _feedback_text(root, state)
    parents = allowed_parents(state)
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
        parent_ids=allowed_parents(state),
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
    batch = _load_proposals(root, state)
    passed = _accepted(state, root, batch, verdicts)
    if revises and _needs_critic_revision(state, passed):
        state.critic_revisions += 1
        message = "\n".join(f"{item.candidate_id}: {item.reason}" for item in revises)
        if not passed and state.critic_revisions > 1:
            return _escalate(state, root, f"critic revision cap\n{message}")
        _write_note(root, state, message)
        state.stage = "PLAN"
        return _maybe_pause(state, root, "CRITIQUE")
    if not passed:
        return _escalate(state, root, "critic dropped every proposal")
    needed = min(3, state.k)
    families = {candidate.family for candidate in passed}
    if (
        not revises
        and state.round == 1
        and state.round_operator == "draft"
        and len(families) < needed
    ):
        state.critic_revisions += 1
        message = f"shortlist has {len(families)} families; need at least {needed}"
        if state.critic_revisions > 1:
            return _escalate(state, root, f"critic revision cap\n{message}")
        _write_note(root, state, message)
        state.stage = "PLAN"
        return _maybe_pause(state, root, "CRITIQUE")
    _clear_feedback(root, state)
    rung = _rungs_for(state)[0]
    state.rung_index = 0
    state.nodes = [
        node
        for node in state.nodes
        if not (node.search_round == state.round and node.status == "pending")
    ]
    parents_for_combine = combine_parents(state) if state.round_operator == "combine" else []
    for candidate in passed:
        parents = (
            list(parents_for_combine)
            if parents_for_combine
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
        _write_node_spec(root, state.run_id, node)
    state.stage = "IMPLEMENT"
    return _maybe_pause(state, root, "SPEC")


def _search_implement(state: RunState, brief: Brief, root: Path) -> RunState:
    node = next((item for item in _round_nodes(state) if item.status == "pending"), None)
    if node is None:
        state.stage = "PREFLIGHT"
        return _maybe_pause(state, root, "IMPLEMENT")
    model_path = candidate_dir(root, state.run_id, node.node_id) / "model.py"
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
    directory = candidate_dir(root, state.run_id, node.node_id)
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
    model_path = candidate_dir(root, state.run_id, node.node_id) / "model.py"
    run_role(
        "implementer",
        brief,
        root,
        state.run_id,
        node_id=node.node_id,
        operator="debug",
        parent_ids=node.parents,
        output_path=model_path,
        note="Fix the failure recorded in debug_log.md. Change only what that traceback requires.",
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
    node = _next_submittable(state, root, ready, cap)
    if node is None:
        state.stage = "WAIT"
        return state
    key = _job_key(node)
    record = dict(state.submitted_jobs.get(key) or {})
    if not record.get("job_run_id"):
        estimate = _next_job_estimate(root, brief, state, node)
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
        retries = int(record.get("infra_retries") or 0)
        runtime = int(record.get("runtime_retries") or 0)
        suffix = ""
        if retries:
            suffix += f":retry{retries}"
        if runtime:
            suffix += f":debug{runtime}"
        token = f"{state.run_id}:{key}" + suffix
        job_run_id = backend.submit(  # type: ignore[attr-defined]
            job="train",
            job_key=key,
            params=params,
            idempotency_token=token,
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
        if any(node.status == "debugging" for node in _round_nodes(state)):
            state.stage = "DEBUG"
            return _maybe_pause(state, root, "WAIT")
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
    if result.get("life_cycle_state") == "INTERNAL_ERROR":
        return _retry_infra_failure(state, root, node, key, record, result)
    if result.get("result_state") != "SUCCESS":
        failure = _load_candidate_failure(root, state, node, result, record)
        record["failure_kind"] = _failure_kind(result, failure)
        if failure.get("traceback"):
            record["traceback"] = failure.get("traceback")
        state.submitted_jobs[key] = record
        if node.rung == "R1" and _is_candidate_runtime(result, failure):
            return _fail_runtime(state, root, node, key, record, failure)
        node.status = "failed"
    return state


def _retry_infra_failure(
    state: RunState,
    root: Path,
    node: NodeWork,
    key: str,
    record: dict[str, Any],
    result: dict[str, Any],
) -> RunState:
    """A job the platform killed (driver crash, lost cluster) says nothing about
    the candidate. Charge the attempt, resubmit once, then escalate."""

    message = str(result.get("state_message") or "internal error")
    state.spent_dbu += _job_charge(record)
    attempts = list(record.get("failed_attempts") or [])
    attempts.append({"job_run_id": record.get("job_run_id"), "state_message": message})
    retries = int(record.get("infra_retries") or 0)
    if retries >= MAX_INFRA_RETRIES:
        record.update(
            {"pruned": True, "failed_attempts": attempts, "failure_kind": "infra_crash"}
        )
        state.submitted_jobs[key] = record
        node.status = "failed"
        return _escalate(
            state,
            root,
            f"infrastructure failure: infra_crash: {key} was killed "
            f"{retries + 1} times; last: {message}",
        )
    state.submitted_jobs[key] = {"infra_retries": retries + 1, "failed_attempts": attempts}
    node.status = "ready"
    state.stage = "SUBMIT"
    return state


def _search_prune(state: RunState, brief: Brief, root: Path, backend: object) -> RunState:
    schedule = _rungs_for(state)
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
        production, lgbm = _deltas_from_found(found)
        metric = None if found is None else found.get("metric")
        latency = None if found is None else found.get("latency_p95_ms")
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
                delta=lgbm,
                delta_vs_lgbm=lgbm,
                delta_vs_production=production,
                latency_p95_ms=None if latency is None else float(latency),
                preflight=node.preflight or "PASS",
                review=node.review or "PASS",
                failure_kind=_view_failure_kind(node, record),
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
            random_metric=random_metric_at(root, brief, rung),
        )
    except Exception as exc:
        return _escalate(state, root, f"pruner failed: {exc}")
    commit_rung(root, state.run_id, views, decision)
    if decision.promoted:
        state.promoted = list(decision.promoted)
        record_c00_promotion(root, state, rung)
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
    state.live_nodes = list(decision.survivors)
    if state.scoring_final_r3:
        state.scoring_final_r3 = False
        state.promoted = list(decision.promoted)
        if decision.promoted:
            record_c00_promotion(root, state, rung)
        write_round_digest(root, state, decision)
        if not state.promoted:
            state.stage = "REPORT"
            write_report(root, state, brief)
            return state
        state.stage = "CONFIRM"
        return _maybe_pause(state, root, "PRUNE")
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
    write_round_digest(root, state, decision)
    return _advance_round(state, brief, root, bool(decision.survivors))


def _search_confirm(state: RunState, brief: Brief, root: Path, backend: object) -> RunState:
    if not state.promoted:
        state.stage = "REPORT"
        write_report(root, state, brief)
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
    write_report(root, state, brief)
    return state


def _advance_round(state: RunState, brief: Brief, root: Path, survived: bool) -> RunState:
    if state.scale == "calibration":
        state.stage = "REPORT"
        write_report(root, state, brief)
        return state
    if not survived:
        state.outer_retries += 1
        if state.outer_retries > MAX_OUTER_RETRIES:
            return _escalate(state, root, "outer retry cap")
        state.nodes = [node for node in state.nodes if node.search_round != state.round]
        state.rung_index = 0
        _reset_revision_counters(state)
        state.stage = "PLAN"
        return _maybe_pause(state, root, "PRUNE")
    state.outer_retries = 0
    kept = [
        node.node_id
        for node in state.nodes
        if node.status == "kept" and node.node_id != C00_ID
    ]
    nxt = operator_for_round(
        state.round + 1,
        plateau=fork_on_plateau(list(state.improvement_gains), brief.effect_floor()),
        lineage_count=len(set(kept)),
    )
    if brief.max_rounds is not None and state.round >= brief.max_rounds:
        nxt = "confirm"
    if nxt == "confirm":
        pending = unscored_final_r3(state)
        if pending:
            for node in pending:
                node.rung = "R3"
                node.status = "ready"
            state.scoring_final_r3 = True
            state.rung_index = 0
            state.stage = "SUBMIT"
            return _maybe_pause(state, root, "PRUNE")
        if not state.promoted:
            state.stage = "REPORT"
            write_report(root, state, brief)
            return state
        state.stage = "CONFIRM"
        return _maybe_pause(state, root, "PRUNE")
    state.round += 1
    state.round_operator = nxt  # type: ignore[assignment]
    state.rung_index = 0
    _reset_revision_counters(state)
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
    _append_debug(root, state.run_id, node, reason)
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
    _remember_escalation(state)
    state.stage = "ESCALATE"
    _write_escalate(root, state, reason)
    write_report_json(root, state)
    return state


def _remember_escalation(state: RunState) -> None:
    if state.escalated_from_stage is None and state.stage != "ESCALATE":
        state.escalated_from_stage = state.stage


def _reset_revision_counters(state: RunState) -> None:
    state.revisions = 0
    state.critic_revisions = 0


def _round_nodes(state: RunState) -> list[NodeWork]:
    return [node for node in state.nodes if node.search_round == state.round]


def _find_node(state: RunState, node_id: str) -> NodeWork | None:
    for node in state.nodes:
        if node.node_id == node_id:
            return node
    return None


def _rungs_for(state: RunState) -> list[str]:
    if state.scoring_final_r3:
        return ["R3"]
    return rungs_for(state.round_operator, scale=state.scale)


def _job_key(node: NodeWork) -> str:
    return f"{node.node_id}/{node.rung}/{node.seed}"


def _inflight_unfinished(state: RunState) -> int:
    count = 0
    for record in state.submitted_jobs.values():
        if record.get("pruned") or record.get("result_state") or not record.get("job_run_id"):
            continue
        count += 1
    return count


def _accepted(
    state: RunState, root: Path, batch: ProposalBatch, verdicts: VerdictBatch
) -> list[Candidate]:
    by_id = {item.candidate_id: item for item in verdicts.verdicts}
    passed: list[Candidate] = []
    for candidate in batch.candidates:
        verdict = by_id.get(candidate.id)
        if verdict is None or verdict.decision != "PASS":
            continue
        if duplicate_candidate(candidate, passed, root):
            continue
        passed.append(candidate)
    return shortlist(state, passed)


def _needs_critic_revision(state: RunState, passed: list[Candidate]) -> bool:
    """Revise once when nothing passed, or when the shortlist is still short of K."""

    if not passed:
        return True
    return len(passed) < state.k and state.critic_revisions < 1


def _next_submittable(
    state: RunState, root: Path, ready: list[NodeWork], cap: int
) -> NodeWork | None:
    """Next ready node. Shared clusters run one R3 job at a time."""

    if _inflight_unfinished(state) >= cap:
        return None
    fixed = load_settings(root).fixed_cluster
    r3_inflight = _inflight_rung(state, "R3")
    for node in ready:
        if fixed and node.rung == "R3" and r3_inflight >= 1:
            continue
        return node
    return None


def _inflight_rung(state: RunState, rung: str) -> int:
    count = 0
    for key, record in state.submitted_jobs.items():
        if record.get("pruned") or record.get("result_state") or not record.get("job_run_id"):
            continue
        parts = key.split("/")
        if len(parts) >= 2 and parts[1] == rung:
            count += 1
    return count


def _view_failure_kind(
    node: NodeWork, record: dict[str, Any]
) -> Literal["infra_crash", "runtime_error"] | None:
    kind = record.get("failure_kind")
    if kind in {"infra_crash", "runtime_error"}:
        return str(kind)
    preflight_failed = (node.preflight or "PASS") == "FAIL"
    review_failed = (node.review or "PASS") == "FAIL"
    if node.status == "failed" and not preflight_failed and not review_failed:
        return "runtime_error"
    return None


def _failure_kind(result: dict[str, Any], failure: dict[str, Any]) -> str:
    if result.get("life_cycle_state") == "INTERNAL_ERROR":
        return "infra_crash"
    if failure.get("traceback"):
        return "runtime_error"
    message = str(result.get("state_message") or failure.get("message") or "").lower()
    markers = ("driver", "cluster", "internal error", "spot", "shutdown")
    if any(token in message for token in markers):
        return "infra_crash"
    return "runtime_error"


def _is_candidate_runtime(result: dict[str, Any], failure: dict[str, Any]) -> bool:
    if result.get("life_cycle_state") == "INTERNAL_ERROR":
        return False
    if result.get("result_state") == "SUCCESS":
        return False
    return bool(str(failure.get("traceback") or "").strip())


def _fail_runtime(
    state: RunState,
    root: Path,
    node: NodeWork,
    key: str,
    record: dict[str, Any],
    failure: dict[str, Any],
) -> RunState:
    """Send one R1 candidate exception to DEBUG. Siblings keep their jobs."""

    reason = str(failure.get("traceback") or failure.get("message") or "candidate runtime error")
    _append_debug(root, state.run_id, node, reason)
    if node.debug_attempts >= MAX_DEBUG_ATTEMPTS:
        node.status = "failed"
        record["failure_kind"] = "runtime_error"
        state.submitted_jobs[key] = record
        return state
    attempts = list(record.get("failed_attempts") or [])
    attempts.append({"job_run_id": record.get("job_run_id"), "traceback": reason})
    state.submitted_jobs[key] = {
        "runtime_retries": int(record.get("runtime_retries") or 0) + 1,
        "failed_attempts": attempts,
        "failure_kind": "runtime_error",
    }
    node.debug_attempts += 1
    node.status = "debugging"
    state.candidate_id = node.node_id
    if _inflight_unfinished(state):
        state.stage = "WAIT"
        return state
    state.stage = "DEBUG"
    return _maybe_pause(state, root, "WAIT")


def _load_candidate_failure(
    root: Path,
    state: RunState,
    node: NodeWork,
    result: dict[str, Any],
    record: dict[str, Any],
) -> dict[str, Any]:
    embedded = result.get("failure")
    if isinstance(embedded, dict):
        return embedded
    if result.get("traceback"):
        return {
            "traceback": str(result.get("traceback")),
            "message": str(result.get("state_message") or ""),
            "exception_type": "CandidateError",
        }
    local = candidate_dir(root, state.run_id, node.node_id) / "failure.json"
    loaded = _read_json_file(local)
    if loaded:
        return loaded
    job_run_id = str(record.get("job_run_id") or "")
    if job_run_id:
        loaded = _read_json_file(root / "runs" / "jobs" / f"{job_run_id}-failure.json")
        if loaded:
            return loaded
    if os.environ.get("AMT_JOBS_BACKEND", "").lower() == "fake":
        return {}
    try:
        from amt.volume import VolumeError
        from amt.volume import get as volume_get
        from harness.train import candidate_failure_path

        settings = load_settings(root)
        remote = candidate_failure_path(
            settings.catalog,
            settings.schema_name,
            state.run_id,
            node.node_id,
            node.rung,
            node.seed,
        )
        if not str(remote).startswith("/Volumes"):
            return _read_json_file(remote) or {}
        downloaded = root / "runs" / "jobs" / f"{node.node_id}-{node.rung}-failure.json"
        try:
            volume_get(root, str(remote), downloaded)
        except VolumeError:
            return {}
        return _read_json_file(downloaded) or {}
    except Exception:
        return {}


def _read_json_file(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return payload if isinstance(payload, dict) else None


def _next_job_estimate(root: Path, brief: Brief, state: RunState, node: NodeWork) -> float:
    """The planner's est_cost prices a node's first job. A later rung scales the
    measured charge of the node's last finished job by the train-fraction ratio."""

    finished = [
        (key, record)
        for key, record in state.submitted_jobs.items()
        if key.startswith(f"{node.node_id}/") and record.get("finished_at")
    ]
    if not finished:
        return float(node.est_cost)
    key, record = max(finished, key=lambda item: float(item[1]["finished_at"]))
    try:
        rungs = load_spec(_spec_path(root, brief)).sampling.rungs
        ratio = float(rungs[node.rung].train_fraction) / float(
            rungs[key.split("/")[1]].train_fraction
        )
    except (KeyError, OSError, ValueError, ZeroDivisionError):
        return float(node.est_cost)
    return _job_charge(record) * max(ratio, 1.0)


def _job_charge(record: dict[str, Any]) -> float:
    start = record.get("submitted_at")
    end = record.get("finished_at")
    if start is not None and end is not None:
        wall = (float(end) - float(start)) / 60.0
        if wall >= 1e-3:
            return estimate_dbu(wall)
    return float(record.get("dbu_est") or 0.0)


def _deltas_from_found(
    found: dict[str, Any] | None,
) -> tuple[DeltaVsBaseline | None, DeltaVsBaseline | None]:
    """Production paired delta, then the same-rung LightGBM paired delta."""

    if not found:
        return None, None
    production = _delta_keys(found, "delta_point", "delta_ci_lo", "delta_ci_hi")
    lgbm = _delta_keys(found, "delta_lgbm_point", "delta_lgbm_ci_lo", "delta_lgbm_ci_hi")
    if lgbm is None:
        lgbm = production
    return production, lgbm


def _delta_keys(
    found: dict[str, Any], point: str, low: str, high: str
) -> DeltaVsBaseline | None:
    if found.get(point) is None:
        return None
    return DeltaVsBaseline(
        point=float(found[point]),
        ci_lo=float(found.get(low) or 0.0),
        ci_hi=float(found.get(high) or 0.0),
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


def _write_node_spec(root: Path, run_id: str, node: NodeWork) -> None:
    directory = candidate_dir(root, run_id, node.node_id)
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


def _append_debug(root: Path, run_id: str, node: NodeWork, reason: str) -> None:
    path = candidate_dir(root, run_id, node.node_id) / "debug_log.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    existing = path.read_text(encoding="utf-8") if path.is_file() else "# debug log\n"
    attempt = node.debug_attempts + 1
    path.write_text(existing.rstrip() + f"\n\n## attempt {attempt}\n\n{reason}\n", encoding="utf-8")


def _plan_note(state: RunState, feedback: str) -> str:
    lines = [f"This is round {state.round}. Propose {state.round_operator} candidates."]
    parents = allowed_parents(state)
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


def _approved(root: Path, run_id: str) -> bool:
    return (root / "runs" / run_id / "APPROVE").is_file()


def _consume_approval(root: Path, run_id: str) -> None:
    (root / "runs" / run_id / "APPROVE").unlink(missing_ok=True)
