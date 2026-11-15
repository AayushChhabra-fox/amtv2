"""Live progress, round digests, and the machine-readable run report.

Progress lines go to stderr and are flushed on every stage so a long search
stays visible. ``report.json`` is the scriptable outcome; ``report.md`` is the
human summary.
"""

from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

from amt.ledger import read_rows
from amt.scale import finish_run_report, stats_summary, write_agent_stats
from amt.schemas import Brief, RunState
from amt.search_policy import C00_ID


def log_progress(state: RunState) -> None:
    """Unbuffered ``[HH:MM:SS] [ROUND r | STAGE]`` line on stderr."""

    stamp = datetime.now().strftime("%H:%M:%S")
    print(
        f"[{stamp}] [ROUND {state.round} | {state.stage}]",
        file=sys.stderr,
        flush=True,
    )


def assign_outcome(state: RunState) -> None:
    """Set ``outcome`` from the nodes that reached confirmation."""

    promoted = [node_id for node_id in state.promoted if node_id]
    if not promoted:
        state.outcome = "no_winner"
        return
    if all(node_id == C00_ID for node_id in promoted):
        state.outcome = "baseline_c00_promoted"
        return
    state.outcome = "winner_promoted"


def write_round_digest(root: Path, state: RunState, decision: Any) -> None:
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


def write_report(root: Path, state: RunState, brief: Brief) -> None:
    """Write ``report.md`` and ``report.json`` for a finished search."""

    if state.stage == "REPORT":
        assign_outcome(state)
    directory = root / "runs" / state.run_id
    directory.mkdir(parents=True, exist_ok=True)
    if state.stage == "REPORT":
        body = finish_run_report(root, state, brief).rstrip()
        outcome = state.outcome or "no_winner"
        (directory / "report.md").write_text(f"{body}\n\nOutcome: {outcome}\n", encoding="utf-8")
    write_report_json(root, state)


def write_report_json(root: Path, state: RunState) -> None:
    directory = root / "runs" / state.run_id
    directory.mkdir(parents=True, exist_ok=True)
    payload = {
        "run_id": state.run_id,
        "stage": state.stage,
        "outcome": state.outcome,
        "escalated_from_stage": state.escalated_from_stage,
        "promoted": list(state.promoted),
        "round": state.round,
        "spent_dbu": state.spent_dbu,
        "budget_dbu": state.budget_dbu,
    }
    (directory / "report.json").write_text(
        json.dumps(payload, indent=2) + "\n",
        encoding="utf-8",
    )
