"""G3 calibration, the K=4 unattended run, and Phase 2 entry.

Calibration is one K=1 lineage with ``approval_mode: every_stage``. A human
signs it once. Later briefs do not repeat it. The small run keeps K=4 with
``approval_mode: none`` and records bug rate, revisions, tokens, and
wall-clock. Full budget only checks that those two steps passed.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from amt.ledger import read_rows
from amt.schemas import Brief, RunState, load_brief
from amt.signoff import (
    SignoffError,
    _read_envelope,
    _relative,
    _signer,
    _stamp,
    _write_envelope,
    brief_key,
)
from amt.status import audit_rows

PROMPT_ROLES = ("planner", "critic", "implementer", "reviewer")


class NodeAgentStats(BaseModel):
    """Tokens and wall-clock for one node. Planner calls with no node stay on the run."""

    model_config = ConfigDict(extra="forbid")

    node_id: str
    tokens: int = Field(default=0, ge=0)
    wall_clock_s: float = Field(default=0.0, ge=0)
    calls: int = Field(default=0, ge=0)
    failed_calls: int = Field(default=0, ge=0)
    debug_attempts: int = Field(default=0, ge=0)
    implementer_calls: int = Field(default=0, ge=0)


class AgentStats(BaseModel):
    """Run-level agent statistics plus one row per node."""

    model_config = ConfigDict(extra="forbid")

    bug_rate: float = Field(default=0.0, ge=0)
    revisions: int = Field(default=0, ge=0)
    tokens: int = Field(default=0, ge=0)
    wall_clock_s: float = Field(default=0.0, ge=0)
    calls: int = Field(default=0, ge=0)
    failed_calls: int = Field(default=0, ge=0)
    debug_attempts: int = Field(default=0, ge=0)
    implementer_calls: int = Field(default=0, ge=0)
    nodes: list[NodeAgentStats] = Field(default_factory=list)


class PromptRecord(BaseModel):
    model_config = ConfigDict(extra="forbid")

    role: str
    path: str
    sha256: str


class ReviewRecord(BaseModel):
    model_config = ConfigDict(extra="forbid")

    node_id: str
    path: str
    verdict: str | None = None


class CalibrationReport(BaseModel):
    """What a human reads before signing G3."""

    model_config = ConfigDict(extra="forbid")

    status: Literal["pass", "fail"]
    run_id: str
    brief_name: str
    k: int
    approval_mode: str
    git_sha: str
    reasons: list[str] = Field(default_factory=list)
    audit: list[str] = Field(default_factory=list)
    prompts: list[PromptRecord] = Field(default_factory=list)
    specs: list[str] = Field(default_factory=list)
    reviews: list[ReviewRecord] = Field(default_factory=list)
    reached_r3: bool = False
    agent_stats: AgentStats = Field(default_factory=AgentStats)


class G3Signoff(BaseModel):
    """One human acceptance of the calibration packet. It is not repeated."""

    model_config = ConfigDict(extra="forbid")

    gate: Literal["G3"] = "G3"
    signer: str
    signed_at: str
    run_id: str
    brief_name: str
    report_sha256: str
    k: Literal[1] = 1
    approval_mode: Literal["every_stage"] = "every_stage"


class SmallRunReceipt(BaseModel):
    """Automatic pass/fail for the K=4 unattended run."""

    model_config = ConfigDict(extra="forbid")

    status: Literal["pass", "fail"]
    run_id: str
    brief_name: str
    k: int
    approval_mode: str
    families: list[str] = Field(default_factory=list)
    family_count: int = Field(ge=0)
    kills_without_reason: list[str] = Field(default_factory=list)
    spent_dbu: float = Field(ge=0)
    budget_dbu: float = Field(ge=0)
    rounds_with_digest: list[int] = Field(default_factory=list)
    bug_rate: float = Field(ge=0)
    revisions: int = Field(ge=0)
    tokens: int = Field(ge=0)
    wall_clock_s: float = Field(ge=0)
    reasons: list[str] = Field(default_factory=list)


def stats_path(root: Path, run_id: str) -> Path:
    return root / "runs" / run_id / "agent_stats.json"


def g3_dir(root: Path) -> Path:
    return root / "gates" / "g3"


def calibration_report_path(root: Path, brief_name: str) -> Path:
    return g3_dir(root) / brief_name / "calibration_report.json"


def g3_signoff_path(root: Path) -> Path:
    return g3_dir(root) / "g3.json"


def small_receipt_path(root: Path, run_id: str) -> Path:
    return root / "gates" / "small" / f"{run_id}.json"


def collect_agent_stats(root: Path, state: RunState) -> AgentStats:
    """Bug rate is debug attempts divided by implementer calls.

    Revisions are schema failures plus critic REVISE cycles. Tokens and
    wall-clock are summed per node from ``agent_calls.jsonl``.
    """

    records = _call_records(root, state.run_id)
    buckets: dict[str, dict[str, float]] = {}
    tokens = 0
    wall = 0.0
    calls = 0
    failed = 0
    implementer_calls = 0
    for record in records:
        calls += 1
        latency = float(record.get("latency_s") or 0.0)
        wall += latency
        token_n = _tokens(record.get("tokens"))
        tokens += token_n
        ok = bool(record.get("ok", True))
        if not ok:
            failed += 1
        node_id = record.get("node_id")
        role = str(record.get("role") or "")
        if role == "implementer":
            implementer_calls += 1
        if isinstance(node_id, str) and node_id:
            bucket = buckets.setdefault(
                node_id,
                {
                    "tokens": 0.0,
                    "wall": 0.0,
                    "calls": 0.0,
                    "failed": 0.0,
                    "implementer_calls": 0.0,
                },
            )
            bucket["tokens"] += token_n
            bucket["wall"] += latency
            bucket["calls"] += 1
            if not ok:
                bucket["failed"] += 1
            if role == "implementer":
                bucket["implementer_calls"] += 1
    debug_by_node = {node.node_id: node.debug_attempts for node in state.nodes}
    for node_id in debug_by_node:
        buckets.setdefault(
            node_id,
            {"tokens": 0.0, "wall": 0.0, "calls": 0.0, "failed": 0.0, "implementer_calls": 0.0},
        )
    debug_total = sum(debug_by_node.values())
    nodes = [
        NodeAgentStats(
            node_id=node_id,
            tokens=int(bucket["tokens"]),
            wall_clock_s=round(bucket["wall"], 4),
            calls=int(bucket["calls"]),
            failed_calls=int(bucket["failed"]),
            debug_attempts=debug_by_node.get(node_id, 0),
            implementer_calls=int(bucket["implementer_calls"]),
        )
        for node_id, bucket in sorted(buckets.items())
    ]
    bug_rate = 0.0 if implementer_calls == 0 else debug_total / implementer_calls
    return AgentStats(
        bug_rate=bug_rate,
        revisions=state.revisions + state.critic_revisions,
        tokens=tokens,
        wall_clock_s=round(wall, 4),
        calls=calls,
        failed_calls=failed,
        debug_attempts=debug_total,
        implementer_calls=implementer_calls,
        nodes=nodes,
    )


def write_agent_stats(root: Path, state: RunState) -> AgentStats:
    stats = collect_agent_stats(root, state)
    path = stats_path(root, state.run_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(stats.model_dump_json(indent=2) + "\n", encoding="utf-8")
    return stats


def stats_summary(stats: AgentStats) -> str:
    return (
        f"Agent statistics: bug rate {stats.bug_rate:.3f}, revisions {stats.revisions}, "
        f"tokens {stats.tokens}, wall-clock {stats.wall_clock_s:.3f}s."
    )


def scale_block_reason(
    root: Path,
    *,
    scale: str,
    run_id: str,
    k: int,
    approval_mode: str,
) -> str | None:
    """Why this scale setting must not start. ``search`` is always allowed here."""

    if scale == "search" or scale == "full":
        return None
    if scale == "calibration":
        return calibration_block_reason(root, run_id)
    if scale == "small":
        if k != 4:
            return "The small unattended run requires K=4."
        if approval_mode != "none":
            return "The small unattended run requires approval_mode none."
        if not g3_is_current(root):
            return (
                "G3 is not signed. Run one K=1 calibration with approval_mode "
                "every_stage and sign it before the unattended run."
            )
        return None
    return f"Unknown scale {scale}."


def calibration_block_reason(root: Path, run_id: str) -> str | None:
    """G3 runs once. An escalated attempt can be replaced. A finished one cannot."""

    if g3_is_current(root):
        return "G3 is already signed. The calibration is not repeated."
    existing = _find_passing_report(root)
    if existing is not None:
        if existing.run_id == run_id:
            return "This calibration already finished. Sign it with amt sign g3."
        return (
            f"G3 calibration already ran as {existing.run_id} on {existing.brief_name}. "
            "Sign it, or leave it. It is not repeated."
        )
    claim = _read_claim(root)
    if claim and claim.get("run_id") not in {None, run_id}:
        other = str(claim.get("run_id"))
        if _run_stage(root, other) != "ESCALATE":
            return f"G3 calibration is already started as {other}."
    return None


def write_calibration_claim(root: Path, run_id: str, brief_name: str) -> None:
    path = g3_dir(root) / "started.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"run_id": run_id, "brief_name": brief_name}
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def build_calibration_report(root: Path, state: RunState, brief: Brief) -> CalibrationReport:
    name = brief_key(brief, Path(state.brief_path))
    stats = collect_agent_stats(root, state)
    audit = audit_rows(root, state.run_id)
    prompts, prompt_gaps = _prompts(root)
    reviews = _reviews(root, state)
    specs = _specs(root, state)
    reached = _reached_r3(root, state.run_id)
    reasons: list[str] = []
    if state.k != 1:
        reasons.append(f"K is {state.k}; calibration requires K=1.")
    if state.approval_mode != "every_stage":
        reasons.append(f"approval_mode is {state.approval_mode}; calibration requires every_stage.")
    reasons.extend(prompt_gaps)
    if not specs:
        reasons.append("No candidate spec was written.")
    if not reviews or any(item.verdict != "PASS" for item in reviews):
        reasons.append("Every calibration review must end with VERDICT: PASS.")
    if not reached:
        reasons.append("The lineage did not finish R3 as kept or promoted.")
    if audit:
        reasons.append("Audit failed: " + "; ".join(audit))
    status: Literal["pass", "fail"] = "pass" if not reasons else "fail"
    return CalibrationReport(
        status=status,
        run_id=state.run_id,
        brief_name=name,
        k=state.k,
        approval_mode=state.approval_mode,
        git_sha=state.git_sha,
        reasons=reasons,
        audit=audit,
        prompts=prompts,
        specs=specs,
        reviews=reviews,
        reached_r3=reached,
        agent_stats=stats,
    )


def write_calibration_report(root: Path, state: RunState, brief: Brief) -> CalibrationReport:
    report = build_calibration_report(root, state, brief)
    path = calibration_report_path(root, report.brief_name)
    _write_envelope(path, report.model_dump(mode="json"))
    (path.parent / "calibration_report.md").write_text(
        _calibration_markdown(report), encoding="utf-8"
    )
    return report


def sign_g3(
    root: Path,
    brief_path: Path,
    signer: str,
    run_id: str,
    *,
    signed_at: datetime | None = None,
) -> G3Signoff:
    """Record the one human acceptance of a passing calibration packet."""

    person = _signer(signer)
    brief = load_brief(brief_path)
    name = brief_key(brief, brief_path)
    report, digest = load_calibration_report(root, name)
    if report.run_id != run_id:
        raise SignoffError(f"G3 report for {name} is run {report.run_id}, not {run_id}.")
    if report.status != "pass":
        detail = " ".join(report.reasons) or report.status
        raise SignoffError(f"G3 is not ready to sign ({report.status}): {detail}")
    if report.k != 1 or report.approval_mode != "every_stage":
        raise SignoffError("G3 only signs a K=1 every_stage calibration.")
    audit = audit_rows(root, run_id)
    if audit:
        raise SignoffError("G3 audit failed: " + "; ".join(audit))
    record = G3Signoff(
        signer=person,
        signed_at=_stamp(signed_at),
        run_id=run_id,
        brief_name=name,
        report_sha256=digest,
    )
    path = g3_signoff_path(root)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(record.model_dump_json(indent=2) + "\n", encoding="utf-8")
    return record


def load_g3_signoff(root: Path) -> G3Signoff | None:
    path = g3_signoff_path(root)
    if not path.is_file():
        return None
    return G3Signoff.model_validate(json.loads(path.read_text(encoding="utf-8")))


def load_calibration_report(root: Path, brief_name: str) -> tuple[CalibrationReport, str]:
    body, digest = _read_envelope(calibration_report_path(root, brief_name))
    return CalibrationReport.model_validate(body), digest


def g3_is_current(root: Path) -> bool:
    """True when the signed packet still matches the report and the audit is clean."""

    record = load_g3_signoff(root)
    if record is None:
        return False
    try:
        report, digest = load_calibration_report(root, record.brief_name)
    except (OSError, SignoffError, ValueError):
        return False
    if digest != record.report_sha256 or report.status != "pass":
        return False
    if report.run_id != record.run_id or report.k != 1:
        return False
    if report.approval_mode != "every_stage":
        return False
    return audit_rows(root, report.run_id) == []


def evaluate_small_run(
    root: Path,
    state: RunState,
    brief: Brief,
    stats: AgentStats,
) -> SmallRunReceipt:
    """Pass when 3 families ran, every kill has a reason, and spend stays in the pool."""

    name = brief_key(brief, Path(state.brief_path))
    rows = [row for row in read_rows(root) if row.run_id == state.run_id and row.kind == "model"]
    families = sorted({row.family for row in rows})
    missing = [
        f"{row.node_id} {row.rung}"
        for row in rows
        if row.status in {"killed", "failed"} and not (row.kill_reason and row.kill_reason.strip())
    ]
    digests = [
        number
        for number in range(1, 5)
        if (root / "runs" / state.run_id / f"round_{number}.md").is_file()
    ]
    reasons: list[str] = []
    if state.k != 4:
        reasons.append(f"K is {state.k}; the unattended run requires K=4.")
    if state.approval_mode != "none":
        reasons.append(f"approval_mode is {state.approval_mode}; the unattended run requires none.")
    if state.stage != "REPORT":
        reasons.append(f"Stage is {state.stage}; the unattended run must finish.")
    if len(families) < 3:
        reasons.append(f"{len(families)} families were evaluated; at least 3 are required.")
    if missing:
        reasons.append("Kills without a reason: " + ", ".join(missing))
    if state.spent_dbu > state.budget_dbu + 1e-9:
        reasons.append(f"Spend {state.spent_dbu:.3f} exceeds the pool {state.budget_dbu:.3f}.")
    if digests != [1, 2, 3, 4]:
        reasons.append(f"Round digests are {digests}; rounds 1 to 4 are required.")
    if not stats_path(root, state.run_id).is_file():
        reasons.append("Agent statistics were not recorded.")
    status: Literal["pass", "fail"] = "pass" if not reasons else "fail"
    return SmallRunReceipt(
        status=status,
        run_id=state.run_id,
        brief_name=name,
        k=state.k,
        approval_mode=state.approval_mode,
        families=families,
        family_count=len(families),
        kills_without_reason=missing,
        spent_dbu=state.spent_dbu,
        budget_dbu=state.budget_dbu,
        rounds_with_digest=digests,
        bug_rate=stats.bug_rate,
        revisions=stats.revisions,
        tokens=stats.tokens,
        wall_clock_s=stats.wall_clock_s,
        reasons=reasons,
    )


def write_small_receipt(
    root: Path,
    state: RunState,
    brief: Brief,
    stats: AgentStats,
) -> SmallRunReceipt:
    receipt = evaluate_small_run(root, state, brief, stats)
    _write_envelope(small_receipt_path(root, state.run_id), receipt.model_dump(mode="json"))
    return receipt


def passing_small_receipt(root: Path) -> SmallRunReceipt | None:
    directory = root / "gates" / "small"
    if not directory.is_dir():
        return None
    for path in sorted(directory.glob("*.json")):
        try:
            body, _digest = _read_envelope(path)
            receipt = SmallRunReceipt.model_validate(body)
        except (OSError, SignoffError, ValueError):
            continue
        if receipt.status == "pass":
            return receipt
    return None


def full_budget_block_reason(root: Path) -> str | None:
    """Phase 2 may start only after G3 is signed and a small run has passed."""

    if not g3_is_current(root):
        return (
            "Full budget is not ready. G3 is not signed. Run one K=1 calibration "
            "with approval_mode every_stage and sign it."
        )
    if passing_small_receipt(root) is None:
        return "Full budget is not ready. No passing K=4 small-budget run is recorded."
    return None


def render_phase2_entry(run_id: str, receipt: SmallRunReceipt) -> str:
    return (
        "# Phase 2 entry\n\n"
        f"G3 is signed. Small run `{receipt.run_id}` passed with "
        f"{receipt.family_count} families, spend {receipt.spent_dbu:.3f} / "
        f"{receipt.budget_dbu:.3f} DBU, bug rate {receipt.bug_rate:.3f}, "
        f"revisions {receipt.revisions}, tokens {receipt.tokens}, "
        f"wall-clock {receipt.wall_clock_s:.3f}s.\n\n"
        f"Run `{run_id}` does not start training. Phase 2 uses the signed brief budget.\n"
    )


def finish_run_report(root: Path, state: RunState, brief: Brief) -> str:
    """Write agent stats, the calibration packet or the small-run receipt, and the report."""

    stats = write_agent_stats(root, state)
    if state.scale == "calibration":
        write_calibration_report(root, state, brief)
    receipt = write_small_receipt(root, state, brief, stats) if state.scale == "small" else None
    return render_run_report(root, state, stats, receipt)


def render_run_report(
    root: Path,
    state: RunState,
    stats: AgentStats,
    receipt: SmallRunReceipt | None,
) -> str:
    rows = [row for row in read_rows(root) if row.run_id == state.run_id]
    families = sorted({row.family for row in rows if row.kind == "model"})
    kills = [row for row in rows if row.status in {"killed", "failed"}]
    promoted = ", ".join(state.promoted) if state.promoted else "none"
    lines = [
        f"# Report {state.run_id}",
        "",
        f"Scale: {state.scale}. K={state.k}. Approval: {state.approval_mode}.",
        f"Promoted: {promoted}",
        f"Spent {state.spent_dbu:.3f} / {state.budget_dbu:.3f} DBU",
        "",
        "Live nodes: " + (", ".join(state.live_nodes) or "(none)"),
        "",
        "## Families",
        "",
        ", ".join(families) if families else "(none)",
        "",
        "## Kills",
        "",
    ]
    if not kills:
        lines.append("(none)")
    else:
        for row in kills:
            lines.append(f"- {row.node_id} {row.rung}: {row.kill_reason}")
    lines.extend(["", "## Agent statistics", "", stats_summary(stats), ""])
    if not stats.nodes:
        lines.append("(no node calls)")
    else:
        for node in stats.nodes:
            lines.append(
                f"- {node.node_id}: tokens {node.tokens}, "
                f"wall-clock {node.wall_clock_s:.3f}s, calls {node.calls}, "
                f"debug attempts {node.debug_attempts}"
            )
    lines.append("")
    if state.scale == "calibration":
        lines.append("Calibration stops before confirmation. Holdouts stay closed.")
    elif not state.promoted:
        lines.append("No candidate cleared the promotion rule.")
    else:
        lines.append("Confirmation scored the promoted nodes on select_holdout.")
    if receipt is not None:
        lines.extend(["", f"Small-run check: {receipt.status}."])
        for reason in receipt.reasons:
            lines.append(f"- {reason}")
    lines.append("")
    return "\n".join(lines)


def _tokens(value: object) -> int:
    if isinstance(value, bool) or value is None:
        return 0
    if isinstance(value, int):
        return max(0, value)
    if isinstance(value, float):
        return max(0, int(value))
    return 0


def _call_records(root: Path, run_id: str) -> list[dict]:
    path = root / "runs" / run_id / "agent_calls.jsonl"
    if not path.is_file():
        return []
    records: list[dict] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        payload = json.loads(line)
        if isinstance(payload, dict):
            records.append(payload)
    return records


def _prompts(root: Path) -> tuple[list[PromptRecord], list[str]]:
    import hashlib

    records: list[PromptRecord] = []
    gaps: list[str] = []
    for role in PROMPT_ROLES:
        path = root / ".cursor" / "agents" / f"{role}.md"
        if not path.is_file():
            gaps.append(f"Role prompt {role}.md is missing.")
            continue
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        records.append(PromptRecord(role=role, path=_relative(root, path), sha256=digest))
    return records, gaps


def _specs(root: Path, state: RunState) -> list[str]:
    found: list[str] = []
    for node in state.nodes:
        path = root / "candidates" / node.node_id / "spec.md"
        if path.is_file():
            found.append(_relative(root, path))
    return found


def _reviews(root: Path, state: RunState) -> list[ReviewRecord]:
    records: list[ReviewRecord] = []
    for node in state.nodes:
        path = root / "candidates" / node.node_id / "review.md"
        if not path.is_file():
            records.append(ReviewRecord(node_id=node.node_id, path="", verdict=None))
            continue
        verdict = None
        for line in path.read_text(encoding="utf-8").splitlines():
            stripped = line.strip()
            if stripped.upper().startswith("VERDICT:"):
                verdict = stripped.split(":", 1)[1].strip().upper()
        records.append(
            ReviewRecord(node_id=node.node_id, path=_relative(root, path), verdict=verdict)
        )
    return records


def _reached_r3(root: Path, run_id: str) -> bool:
    for row in read_rows(root):
        if row.run_id != run_id or row.rung != "R3":
            continue
        if row.status in {"kept", "promoted"}:
            return True
    return False


def _find_passing_report(root: Path) -> CalibrationReport | None:
    directory = g3_dir(root)
    if not directory.is_dir():
        return None
    for path in sorted(directory.glob("*/calibration_report.json")):
        try:
            body, _digest = _read_envelope(path)
            report = CalibrationReport.model_validate(body)
        except (OSError, SignoffError, ValueError):
            continue
        if report.status == "pass":
            return report
    return None


def _read_claim(root: Path) -> dict | None:
    path = g3_dir(root) / "started.json"
    if not path.is_file():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, dict):
        return payload
    return None


def _run_stage(root: Path, run_id: str) -> str | None:
    path = root / "runs" / run_id / "state.json"
    if not path.is_file():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None
    if isinstance(payload, dict):
        stage = payload.get("stage")
        if isinstance(stage, str):
            return stage
    return None


def _calibration_markdown(report: CalibrationReport) -> str:
    lines = [
        f"# G3 calibration {report.run_id}",
        "",
        f"Status: {report.status}",
        f"Brief: {report.brief_name}",
        f"K={report.k}. Approval: {report.approval_mode}. Git: {report.git_sha}.",
        "",
        "## Why",
        "",
    ]
    if report.reasons:
        lines.extend(f"- {reason}" for reason in report.reasons)
    else:
        lines.append("The lineage reached R3, the reviews passed, and the audit is clean.")
    lines.extend(["", "## Role prompts", ""])
    if not report.prompts:
        lines.append("(none)")
    else:
        for prompt in report.prompts:
            lines.append(f"- {prompt.role}: `{prompt.path}`")
    lines.extend(["", "## Specs", ""])
    if report.specs:
        lines.extend(f"- `{path}`" for path in report.specs)
    else:
        lines.append("(none)")
    lines.extend(["", "## Reviews", ""])
    if not report.reviews:
        lines.append("(none)")
    else:
        for review in report.reviews:
            lines.append(f"- {review.node_id}: {review.verdict or 'missing'}")
    lines.extend(
        [
            "",
            "## Agent statistics",
            "",
            stats_summary(report.agent_stats),
            "",
            "Sign a passing report once with "
            f'`amt sign g3 --brief <brief> --run-id {report.run_id} --signer "Your Name"`.',
            "",
        ]
    )
    return "\n".join(lines)
