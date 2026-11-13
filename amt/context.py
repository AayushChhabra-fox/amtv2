"""Scoped context for one agent call.

Each role sees file paths plus short ledger lines: the 12 newest nodes, the
best node in each family, sibling one-liners for a draft or an improvement,
and the full debug fix chain. The prompt does not paste harness source.
"""

from __future__ import annotations

import json
from pathlib import Path

from amt.ledger import read_rows
from amt.schemas import Brief, LedgerRow, load_brief

_NEWEST = 12


def build_context(
    role: str,
    brief: Brief,
    root: Path,
    *,
    run_id: str | None = None,
    node_id: str | None = None,
    operator: str | None = None,
    parent_ids: list[str] | None = None,
    output_path: str | None = None,
    inputs: list[str] | None = None,
    note: str = "",
) -> str:
    metric = "ndcg@10_graded"
    if brief.problem.metric is not None:
        metric = brief.problem.metric.name
    lines = [
        f"Role: {role}",
        f"Brief: {brief.name or brief.resolved_dataset_spec() or 'untitled'}",
        f"Metric: {metric}",
        f"Approval mode: {brief.approval_mode}",
        f"Scale: {brief.scale}",
        f"Keep at most K={brief.k} candidates.",
        "",
        "Task:",
        f"  query_unit: {brief.task.query_unit or '(unset)'}",
        f"  candidates: {brief.task.candidates or '(unset)'}",
        f"  serve_time_inputs: {', '.join(brief.task.serve_time_inputs) or '(unset)'}",
        f"  forbidden: {', '.join(brief.task.forbidden) or '(none)'}",
        "",
        "Write scope: candidates/<id>/ only.",
        "Do not edit harness/, amt/, briefs/, datasets/, databricks.yml, or resources/.",
        "",
        _ledger_block(root, brief, run_id),
    ]
    if operator in {"draft", "improve"}:
        lines.extend(["", _sibling_block(root, run_id, node_id, operator)])
    if operator == "debug":
        lines.extend(["", _fix_chain(root, node_id)])
    if operator:
        lines.extend(["", f"Operator for this call: {operator}"])
    if parent_ids:
        lines.append("Parent nodes: " + ", ".join(parent_ids))
    if brief.design_space and brief.design_space.families:
        lines.append("Allowed families: " + "; ".join(brief.design_space.families))
    if note:
        lines.extend(["", "Note:", note.strip()])
    if operator == "draft":
        lines.extend(
            [
                "",
                "Complexity cue: prefer a simple, well-understood design. "
                "No ensembles and no heavy tuning on a draft.",
            ]
        )
    if role == "planner":
        used = sorted(path.name for path in (root / "candidates").glob("*") if path.is_dir())
        if used:
            lines.extend(["", "Candidate ids already taken (do not reuse): " + ", ".join(used)])
        lines.extend(
            [
                "",
                "Web search and page fetch are available for ranking methods, papers, "
                "and library docs. Do not put brief data, table names, or metric values "
                "in a query. List every page you used in references.",
            ]
        )
    lines.extend(["", "Files:"])
    for path in inputs or _default_inputs(role, root, brief, node_id):
        lines.append(f"  read {path}")
    if output_path:
        lines.append(f"  write {output_path}")
    lines.append("")
    lines.append(_output_contract(role))
    return "\n".join(lines).rstrip() + "\n"


def _ledger_block(root: Path, brief: Brief, run_id: str | None) -> str:
    rows = _comparable_rows(root, brief, run_id)
    newest = rows[-_NEWEST:]
    lines = ["Ledger (12 newest nodes on this dataset, one line each):"]
    if not newest:
        lines.append("  (empty)")
    else:
        lines.extend(f"  {_one_line(row)}" for row in newest)
    lines.append("Best node per family:")
    best = _best_by_family(rows)
    if not best:
        lines.append("  (none)")
    else:
        lines.extend(f"  {_one_line(row)}" for row in best)
    return "\n".join(lines)


def _comparable_rows(root: Path, brief: Brief, run_id: str | None) -> list[LedgerRow]:
    """This run's rows plus rows from other runs on the same dataset.

    Deltas measured on another dataset's splits and baselines are not comparable.
    """

    dataset = _normalized(brief.resolved_dataset_spec())
    datasets: dict[str, str | None] = {}
    kept: list[LedgerRow] = []
    for row in read_rows(root):
        if row.run_id == run_id:
            kept.append(row)
            continue
        if row.run_id not in datasets:
            datasets[row.run_id] = _run_dataset(root, row.run_id)
        if dataset is not None and datasets[row.run_id] == dataset:
            kept.append(row)
    return kept


def _run_dataset(root: Path, run_id: str) -> str | None:
    state = root / "runs" / run_id / "state.json"
    if not state.is_file():
        return None
    brief_path = json.loads(state.read_text(encoding="utf-8")).get("brief_path")
    if not brief_path:
        return None
    path = root / brief_path
    if not path.is_file():
        return None
    try:
        return _normalized(load_brief(path).resolved_dataset_spec())
    except (ValueError, OSError):
        return None


def _normalized(spec: str | None) -> str | None:
    return spec.rstrip("/") if spec else None


def _one_line(row: LedgerRow) -> str:
    delta = "-"
    if row.delta_vs_baseline is not None:
        delta = f"{row.delta_vs_baseline.point:.4f}"
    elif row.metric is not None:
        delta = f"metric={row.metric:.4f}"
    parent = row.parent_id or "-"
    return (
        f"{row.node_id} {row.operator} {row.family} {row.rung} "
        f"{row.status} delta={delta} parent={parent} {row.claim}"
    )


def _best_by_family(rows: list[LedgerRow]) -> list[LedgerRow]:
    best: dict[str, LedgerRow] = {}
    for row in rows:
        current = best.get(row.family)
        if current is None or _score(row) > _score(current):
            best[row.family] = row
    return [best[family] for family in sorted(best)]


def _score(row: LedgerRow) -> float:
    if row.delta_vs_baseline is not None:
        return row.delta_vs_baseline.point
    if row.metric is not None:
        return row.metric
    return float("-inf")


def _sibling_block(root: Path, run_id: str | None, node_id: str | None, operator: str) -> str:
    lines = [f"Siblings for this {operator} (one line each):"]
    siblings = _siblings(root, run_id, node_id)
    if not siblings:
        lines.append("  (none)")
    else:
        lines.extend(f"  {line}" for line in siblings)
    return "\n".join(lines)


def _siblings(root: Path, run_id: str | None, node_id: str | None) -> list[str]:
    if not run_id:
        return []
    path = root / "runs" / run_id / "state.json"
    if not path.is_file():
        return []
    payload = json.loads(path.read_text(encoding="utf-8"))
    current_round = payload.get("round")
    found: list[str] = []
    for node in payload.get("nodes") or []:
        if node.get("node_id") == node_id:
            continue
        if current_round is not None and node.get("search_round") not in {None, current_round}:
            continue
        claim = node.get("claim") or ""
        found.append(
            f"{node.get('node_id')} {node.get('operator')} {node.get('family')} "
            f"{node.get('status')} {claim}"
        )
    return found


def _debug_log(root: Path, node_id: str | None) -> Path:
    name = node_id or "unknown"
    return root / "candidates" / name / "debug_log.md"


def _fix_chain(root: Path, node_id: str | None) -> str:
    path = _debug_log(root, node_id)
    header = f"Fix chain for {node_id or 'this node'} ({path.as_posix()}):"
    if not path.is_file():
        return header + "\n  (no attempts yet)"
    body = path.read_text(encoding="utf-8").strip()
    return header + "\n" + body


def _default_inputs(role: str, root: Path, brief: Brief, node_id: str | None) -> list[str]:
    schema = root / "amt" / "json_schemas"
    paths = []
    if role == "planner":
        paths.append(str(schema / "proposal_batch.schema.json"))
    elif role == "critic":
        paths.append(str(schema / "verdict_batch.schema.json"))
    elif role == "implementer":
        paths.append(str(root / "harness" / "interface.py"))
        if node_id:
            paths.append(str(root / "candidates" / node_id / "spec.md"))
    elif role == "reviewer" and node_id:
        candidate = root / "candidates" / node_id
        paths.append(str(candidate / "spec.md"))
        paths.append(str(candidate / "model.py"))
    spec = brief.resolved_dataset_spec()
    if spec:
        paths.append(spec)
    return paths


def _output_contract(role: str) -> str:
    if role == "planner":
        return (
            "Return one JSON object with a candidates list. Each candidate needs "
            "id, operator, family, claim, fit_argument, cheap_test, kill_if, and est_cost. "
            "improve also needs parent_id and change. draft must not set parent_id. "
            "Web search is available. When an idea came from a page you opened, "
            "list that http(s) URL in references (at most 5)."
        )
    if role == "critic":
        return (
            "Return one JSON object with a verdicts list. Each verdict needs "
            "candidate_id, decision (PASS, REVISE, or DROP), and reason."
        )
    if role == "reviewer":
        return "Write review.md ending with a line VERDICT: PASS or VERDICT: FAIL."
    if role == "implementer":
        return (
            "Write model.py implementing fit, score, save, and load. Stay inside candidates/<id>/."
        )
    return "Follow the role prompt."
