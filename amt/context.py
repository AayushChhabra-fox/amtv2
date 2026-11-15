"""Scoped context for one agent call.

Each role sees file paths plus short ledger lines: the 12 newest nodes, the
best node in each family, sibling one-liners for a draft or an improvement,
and the full debug fix chain. The prompt does not paste harness source.
"""

from __future__ import annotations

import json
from pathlib import Path

from amt.ledger import read_annotations, read_rows
from amt.paths import candidate_dir
from amt.schemas import Brief, LedgerRow, load_brief

_NEWEST = 12
_REPLY_ONLY = (
    "You run read-only: do not write or edit any file. Your reply is the output, "
    "and it must contain only the JSON object. "
)


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
        *_write_scope(role, brief, run_id),
        "",
        _ledger_block(root, brief, run_id),
    ]
    if role in {"planner", "critic"}:
        lines.extend(["", _constraint_block(brief), "", _modality_block(root, brief)])
    if operator in {"draft", "improve"}:
        lines.extend(["", _sibling_block(root, run_id, node_id, operator)])
    if operator == "debug":
        lines.extend(["", _fix_chain(root, run_id, node_id)])
    if operator:
        lines.extend(["", f"Operator for this call: {operator}"])
    if parent_ids:
        lines.append("Parent nodes: " + ", ".join(parent_ids))
    if brief.design_space and brief.design_space.families:
        lines.append("Family hints (optional): " + "; ".join(brief.design_space.families))
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
    if role == "planner" and run_id:
        directory = root / "runs" / run_id / "candidates"
        used = []
        if directory.is_dir():
            used = sorted(path.name for path in directory.glob("*") if path.is_dir())
        if used:
            lines.extend(["", "Candidate ids already used in this run: " + ", ".join(used)])
        lines.extend(
            [
                "",
                "Web search and page fetch are available for ranking methods, papers, "
                "and library docs. Do not put brief data, table names, or metric values "
                "in a query. List every page you used in references.",
            ]
        )
    lines.extend(["", "Files:"])
    for path in inputs or _default_inputs(role, root, brief, node_id, run_id):
        lines.append(f"  read {path}")
    if output_path and role in {"planner", "critic"}:
        lines.append(f"  (the orchestrator saves your reply to {output_path})")
    elif output_path:
        lines.append(f"  write {output_path}")
    lines.append("")
    lines.append(_output_contract(role))
    return "\n".join(lines).rstrip() + "\n"


def _constraint_block(brief: Brief) -> str:
    constraints = brief.constraints
    latency = None
    if brief.efficiency.serving is not None:
        latency = brief.efficiency.serving.latency_p95_ms_per_100_candidates
    shown = "(unset)" if latency is None else str(latency)
    return "\n".join(
        [
            "Constraints:",
            f"  allow_gpu_serving: {constraints.allow_gpu_serving}",
            f"  allow_external_weights: {constraints.allow_external_weights}",
            f"  license_policy: {constraints.license_policy}",
            f"  latency_p95_ms_per_100_candidates: {shown}",
        ]
    )


def _modality_block(root: Path, brief: Brief) -> str:
    """Viable families from the ready-gate manifest, plus the c00 parent rule."""

    fallback = (
        "Modality profile: (unavailable). Choose a family the constraints allow. "
        "c00 is the LightGBM reference; improve it only when it is an allowed parent."
    )
    if not brief.name:
        return fallback
    path = root / "gates" / "ready" / brief.name / "manifest.json"
    if not path.is_file():
        return fallback
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return fallback
    from amt.modality import paradigm_decisions, profile_from_manifest

    profile = profile_from_manifest(payload) if isinstance(payload, dict) else None
    if profile is None:
        return fallback
    decisions = paradigm_decisions(profile, brief)
    viable = [item.name for item in decisions if item.viable]
    blocked = [f"{item.name} ({item.reason})" for item in decisions if not item.viable]
    lines = [
        "Modality profile: " + json.dumps(profile.model_dump(mode="json"), sort_keys=True),
        "Viable families: " + (", ".join(viable) or "(none)"),
    ]
    if blocked:
        lines.append("Blocked families: " + "; ".join(blocked))
    lines.append(
        "Choose families from the viable list. Brief family hints are optional. "
        "c00 is the materialized LightGBM reference; improve it only when it is an allowed parent."
    )
    return "\n".join(lines)


def _ledger_block(root: Path, brief: Brief, run_id: str | None) -> str:
    rows = _comparable_rows(root, brief, run_id)
    notes = read_annotations(root)
    newest = rows[-_NEWEST:]
    lines = ["Ledger (12 newest nodes on this dataset, one line each):"]
    if not newest:
        lines.append("  (empty)")
    else:
        lines.extend(f"  {_one_line(row, notes)}" for row in newest)
    lines.append("Best node per family:")
    best = _best_by_family(rows)
    if not best:
        lines.append("  (none)")
    else:
        lines.extend(f"  {_one_line(row, notes)}" for row in best)
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


def _one_line(row: LedgerRow, notes: dict[str, str] | None = None) -> str:
    delta = "-"
    shown = row.delta_vs_lgbm or row.delta_vs_production
    if shown is not None:
        delta = f"{shown.point:.4f}"
    elif row.metric is not None:
        delta = f"metric={row.metric:.4f}"
    parent = row.parent_id or "-"
    status = row.status
    note = (notes or {}).get(row.job_run_id or "")
    if note:
        status = f"not_measured ({note})"
    return (
        f"{row.node_id} {row.operator} {row.family} {row.rung} "
        f"{status} delta={delta} parent={parent} {row.claim}"
    )


def _best_by_family(rows: list[LedgerRow]) -> list[LedgerRow]:
    best: dict[str, LedgerRow] = {}
    for row in rows:
        current = best.get(row.family)
        if current is None or _score(row) > _score(current):
            best[row.family] = row
    return [best[family] for family in sorted(best)]


def _score(row: LedgerRow) -> float:
    shown = row.delta_vs_lgbm or row.delta_vs_production
    if shown is not None:
        return shown.point
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


def _debug_log(root: Path, run_id: str | None, node_id: str | None) -> Path:
    name = node_id or "unknown"
    return candidate_dir(root, run_id or "unknown", name) / "debug_log.md"


def _fix_chain(root: Path, run_id: str | None, node_id: str | None) -> str:
    path = _debug_log(root, run_id, node_id)
    header = f"Fix chain for {node_id or 'this node'} ({path.as_posix()}):"
    if not path.is_file():
        return header + "\n  (no attempts yet)"
    body = path.read_text(encoding="utf-8").strip()
    return header + "\n" + body


def _default_inputs(
    role: str, root: Path, brief: Brief, node_id: str | None, run_id: str | None
) -> list[str]:
    schema = root / "amt" / "json_schemas"
    paths = []
    if role == "planner":
        paths.append(str(schema / "proposal_batch.schema.json"))
    elif role == "critic":
        paths.append(str(schema / "verdict_batch.schema.json"))
    elif role == "implementer":
        paths.append(str(root / "harness" / "interface.py"))
        if node_id:
            paths.append(str(candidate_dir(root, run_id or "unknown", node_id) / "spec.md"))
    elif role == "dataset_fixer":
        spec_text = brief.resolved_dataset_spec()
        if spec_text:
            features = root / spec_text / "features"
            if features.is_dir():
                paths.extend(str(path) for path in sorted(features.glob("*.sql")))
    elif role == "reviewer" and node_id:
        candidate = candidate_dir(root, run_id or "unknown", node_id)
        paths.append(str(candidate / "spec.md"))
        paths.append(str(candidate / "model.py"))
    spec = brief.resolved_dataset_spec()
    if spec:
        paths.append(spec)
    return paths


def _output_contract(role: str) -> str:
    if role == "planner":
        return (
            _REPLY_ONLY + "Return one JSON object with a candidates list. Each candidate needs "
            "id, operator, family, claim, fit_argument, cheap_test, kill_if, and est_cost. "
            "improve also needs parent_id and change. draft must not set parent_id. "
            "Web search is available. When an idea came from a page you opened, "
            "list that http(s) URL in references (at most 5)."
        )
    if role == "critic":
        return (
            _REPLY_ONLY + "Return one JSON object with a verdicts list. Each verdict needs "
            "candidate_id, decision (PASS, REVISE, or DROP), and reason."
        )
    if role == "reviewer":
        return "Write review.md ending with a line VERDICT: PASS or VERDICT: FAIL."
    if role == "implementer":
        return (
            "Write model.py implementing fit, score, save, and load. "
            "Stay inside runs/<run_id>/candidates/<id>/."
        )
    if role == "dataset_fixer":
        return (
            "Fix the failing feature SQL. Return one JSON object with an edits list. "
            "Each edit needs path (features/<name>.sql) and sql (the full file). "
            "Keep the same output column aliases. Do not add forbidden columns or sources."
        )
    return "Follow the role prompt."


def _write_scope(role: str, brief: Brief, run_id: str | None) -> list[str]:
    if role in {"planner", "critic"}:
        scope = "Write scope: none; reply only."
        banned = "Do not edit harness/, amt/, briefs/, datasets/, databricks.yml, or resources/."
    elif role == "dataset_fixer":
        spec = Path(brief.resolved_dataset_spec() or "").name or "<spec>"
        scope = f"Write scope: datasets/{spec}/features/*.sql only."
        banned = (
            "Do not edit dataset.yaml, queries.sql, candidates.sql, labels.sql, "
            "checks/, harness/, amt/, or any other spec."
        )
    else:
        scope = f"Write scope: runs/{run_id or '<run_id>'}/candidates/<id>/ only."
        banned = "Do not edit harness/, amt/, briefs/, datasets/, databricks.yml, or resources/."
    return [scope, banned]
