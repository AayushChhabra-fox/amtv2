"""Repair ``datasets/<spec>/features/*.sql`` after a preflight failure.

SQL analysis, type, cast, and contract errors inside a feature file can be
repaired. Errors in queries, candidates, labels, point-in-time or leakage
checks, split or label-support failures, and harness bugs are escalated.
Each attempt is capped by ``AMT_DATASET_REPAIR_MAX_ATTEMPTS`` (default 3)
and ``AMT_DATASET_REPAIR_MAX_DBU`` (default 10).
"""

from __future__ import annotations

import difflib
import json
import os
import re
from pathlib import Path

from amt.agents import run_role
from amt.config import load_settings
from amt.dataset_preflight import run_preflight
from amt.deploy import deploy
from amt.schemas import Brief
from harness.data.assemble import _PLACEHOLDER
from harness.data.contract import DatasetSpec, forbidden_hit, load_spec

_CORE_FILES = {"queries.sql", "candidates.sql", "labels.sql", "dataset.yaml"}
_CORE_FRAGMENTS = {"queries", "candidates", "labels", "search_val"}
_ESCALATE_KINDS = {
    "point_in_time",
    "label_window",
    "split",
    "label_support",
    "check",
    "forbidden_column",
}
_REPAIRABLE_KINDS = {"contract", "explain"}
_PLATFORM_TYPES = {
    "TypeError",
    "AttributeError",
    "KeyError",
    "NameError",
    "ImportError",
    "RuntimeError",
    "AssertionError",
}
_SQL_MARKERS = (
    "analysisexception",
    "parseexception",
    "unresolved_column",
    "cannot resolve",
    "datatype_mismatch",
    "data type mismatch",
    "cannot cast",
    "type mismatch",
    "incompatible",
    "analysis",
)
_COMMENT = re.compile(r"--[^\n]*")
_STRING = re.compile(r"'(?:''|[^'])*'")
_CAST = re.compile(
    r"\bCAST\s*\((?:[^()]|\([^()]*\))*\bAS\s+[A-Za-z_][A-Za-z0-9_]*\s*\)",
    re.IGNORECASE,
)
_ALIAS = re.compile(r"\bAS\s+([A-Za-z_][A-Za-z0-9_]*)\b", re.IGNORECASE)
_CTE = re.compile(r"(?:WITH|,)\s+([A-Za-z_][A-Za-z0-9_]*)\s+AS\s*\(", re.IGNORECASE)
_FROM = re.compile(
    r"\b(?:FROM|JOIN)\s+(\{[A-Za-z_][A-Za-z0-9_]*\}|`?[A-Za-z_][\w.]*`?|\()",
    re.IGNORECASE,
)
_IDENT = re.compile(r"\b([A-Za-z_][A-Za-z0-9_]*)\b")
_ALLOWED_PLACEHOLDERS = {
    "start",
    "end",
    "start_ts",
    "end_ts",
    "scan_start",
    "scan_end",
    "entity_type",
    "date_col",
}


class DatasetRepairError(RuntimeError):
    """The failure is not safe to repair, or a repair broke the local checks."""

    def __init__(self, action: str, failure: dict) -> None:
        self.action = action
        self.failure = failure
        super().__init__(str(failure.get("message") or action))


def repair_caps() -> tuple[int, float]:
    attempts_text = os.environ.get("AMT_DATASET_REPAIR_MAX_ATTEMPTS", "3").strip() or "3"
    dbu_text = os.environ.get("AMT_DATASET_REPAIR_MAX_DBU", "10").strip() or "10"
    return max(0, int(attempts_text)), float(dbu_text)


def slice_dbu() -> float:
    """Nominal DBU for one 90-second slice. A missing rate counts as 1 DBU/hour."""

    rate = load_settings().dbu_per_hour
    if rate <= 0:
        rate = 1.0
    return rate * 90.0 / 3600.0


def classify_failure(failure: dict) -> str:
    """``repairable`` for feature-SQL errors, otherwise ``escalate``."""

    kind = str(failure.get("kind") or "")
    path = str(failure.get("fragment_path") or "").replace("\\", "/")
    fragment = str(failure.get("fragment") or "")
    exc = str(failure.get("exception_type") or "")
    message = str(failure.get("message") or "")
    trace = str(failure.get("traceback") or "")
    if kind in _ESCALATE_KINDS or "leak" in message.lower():
        return "escalate"
    if _core_fragment(path, fragment):
        return "escalate"
    if _platform_bug(trace, exc, kind):
        return "escalate"
    if _feature_sql(path) and (kind in _REPAIRABLE_KINDS or _sql_error(kind, exc, message)):
        return "repairable"
    return "escalate"


def _core_fragment(path: str, fragment: str) -> bool:
    name = path.rsplit("/", 1)[-1]
    if fragment in _CORE_FRAGMENTS or fragment.startswith("source:"):
        return True
    if name in _CORE_FILES or path.startswith("checks/"):
        return True
    return False


def _feature_sql(path: str) -> bool:
    return path.startswith("features/") and path.endswith(".sql") and path.count("/") == 1


def _sql_error(kind: str, exc: str, message: str) -> bool:
    blob = f"{kind} {exc} {message}".lower()
    return any(marker in blob for marker in _SQL_MARKERS)


def _platform_bug(trace: str, exc: str, kind: str) -> bool:
    if kind:
        return False
    if exc in {"AnalysisException", "ParseException", "Py4JJavaError", "DatasetBuildError"}:
        return False
    text = trace.replace("\\", "/")
    return exc in _PLATFORM_TYPES and "harness/" in text


def _strip_sql(sql: str) -> str:
    text = _STRING.sub(" ", _COMMENT.sub(" ", sql))
    previous = None
    while previous != text:
        previous = text
        text = _CAST.sub(" ", text)
    return text


def projected_aliases(sql: str) -> set[str]:
    return set(_ALIAS.findall(_strip_sql(sql)))


def feature_columns(spec: DatasetSpec) -> dict[str, set[str]]:
    """``feat__<group>__<alias>`` columns each feature file projects."""

    found: dict[str, set[str]] = {}
    for name, group in spec.feature_groups.items():
        aliases = projected_aliases(spec.sql_text(group.sql))
        skip = {key.lower() for key in group.keys} | {group.available_at.lower()}
        found[name] = {
            f"feat__{name}__{alias}" for alias in aliases if alias.lower() not in skip
        }
    return found


def _read_features(spec: DatasetSpec) -> dict[str, str]:
    texts: dict[str, str] = {}
    for group in spec.feature_groups.values():
        path = spec.root / group.sql
        texts[group.sql] = path.read_text(encoding="utf-8") if path.is_file() else ""
    return texts


def _restore_features(spec_dir: Path, before: dict[str, str]) -> None:
    for relative, text in before.items():
        path = spec_dir / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")


def _safe_feature_rel(raw: str) -> str:
    text = raw.replace("\\", "/").lstrip("/")
    marker = "/features/"
    if marker in text:
        text = "features/" + text.split(marker, 1)[1]
    parts = [part for part in text.split("/") if part not in ("", ".")]
    if ".." in parts or len(parts) != 2 or parts[0] != "features":
        raise DatasetRepairError(
            "escalate",
            {"kind": "contract", "message": f"refusing to write {raw}"},
        )
    if not parts[1].endswith(".sql") or parts[1].startswith("."):
        raise DatasetRepairError(
            "escalate",
            {"kind": "contract", "message": f"refusing to write {raw}"},
        )
    return "features/" + parts[1]


def apply_fixer_output(spec: DatasetSpec, output: dict) -> list[str]:
    """Write ``edits`` from the fixer. An empty list means the files are already on disk."""

    edits = output.get("edits") if isinstance(output, dict) else None
    if not edits:
        return []
    if not isinstance(edits, list):
        raise DatasetRepairError(
            "escalate",
            {"kind": "contract", "message": "dataset_fixer edits must be a list"},
        )
    written: list[str] = []
    for edit in edits:
        if not isinstance(edit, dict):
            raise DatasetRepairError(
                "escalate",
                {"kind": "contract", "message": "dataset_fixer edit must be an object"},
            )
        relative = _safe_feature_rel(str(edit.get("path") or ""))
        sql = edit.get("sql")
        if not isinstance(sql, str) or not sql.strip():
            raise DatasetRepairError(
                "escalate",
                {"kind": "contract", "message": f"{relative} edit has no sql"},
            )
        if spec.root is None:
            raise DatasetRepairError(
                "escalate",
                {"kind": "contract", "message": "spec root is not set"},
            )
        (spec.root / relative).write_text(sql, encoding="utf-8")
        written.append(relative)
    return written


def _reject_forbidden(sql: str, spec: DatasetSpec) -> None:
    cleaned = _strip_sql(sql)
    for token in set(_IDENT.findall(cleaned)):
        pattern = forbidden_hit(token, spec.forbidden_columns)
        if pattern is not None:
            raise DatasetRepairError(
                "escalate",
                {
                    "kind": "forbidden_column",
                    "message": f"feature SQL uses forbidden column {token}",
                    "examples": [{"column": token, "pattern": pattern}],
                },
            )
    allowed_placeholders = set(spec.sources) | _ALLOWED_PLACEHOLDERS
    for name in _PLACEHOLDER.findall(cleaned):
        if name not in allowed_placeholders:
            raise DatasetRepairError(
                "escalate",
                {"kind": "contract", "message": f"feature SQL uses unknown source {{{name}}}"},
            )
    ctes = set(_CTE.findall(cleaned))
    allowed_tables = {source.table for source in spec.sources.values()}
    allowed_tables |= {f"amt_src_{name}" for name in spec.sources}
    for token in _FROM.findall(cleaned):
        if token.startswith("(") or token.startswith("{"):
            continue
        name = token.strip("`")
        if name in allowed_tables or name in ctes:
            continue
        raise DatasetRepairError(
            "escalate",
            {"kind": "contract", "message": f"feature SQL reads undeclared source {name}"},
        )


def validate_repair(
    spec_dir: Path,
    yaml_bytes: bytes,
    original_columns: dict[str, set[str]],
) -> DatasetSpec:
    """``dataset.yaml`` is unchanged, ``feat__*`` columns match, and SQL stays in bounds."""

    current = (spec_dir / "dataset.yaml").read_bytes()
    if current != yaml_bytes:
        raise DatasetRepairError(
            "escalate",
            {"kind": "contract", "message": "dataset.yaml changed"},
        )
    spec = load_spec(spec_dir)
    found = feature_columns(spec)
    if found != original_columns:
        raise DatasetRepairError(
            "escalate",
            {
                "kind": "contract",
                "message": "feat__ column set changed",
                "examples": [
                    {
                        "group": name,
                        "before": sorted(original_columns.get(name, set())),
                        "after": sorted(found.get(name, set())),
                    }
                    for name in sorted(set(original_columns) | set(found))
                    if original_columns.get(name, set()) != found.get(name, set())
                ],
            },
        )
    for group in spec.feature_groups.values():
        _reject_forbidden(spec.sql_text(group.sql), spec)
    return spec


def record_attempt(
    root: Path,
    spec_name: str,
    attempt: int,
    before: dict[str, str],
    after: dict[str, str],
    failure: dict,
) -> Path:
    """Store the before/after feature SQL and a unified diff."""

    directory = root / "runs" / "dataset-repair" / spec_name / str(attempt)
    (directory / "before").mkdir(parents=True, exist_ok=True)
    (directory / "after").mkdir(parents=True, exist_ok=True)
    diffs: list[str] = []
    names = sorted(set(before) | set(after))
    for name in names:
        prior = before.get(name, "")
        updated = after.get(name, "")
        file_name = Path(name).name
        (directory / "before" / file_name).write_text(prior, encoding="utf-8")
        (directory / "after" / file_name).write_text(updated, encoding="utf-8")
        if prior != updated:
            diffs.extend(
                difflib.unified_diff(
                    prior.splitlines(keepends=True),
                    updated.splitlines(keepends=True),
                    fromfile=f"a/{name}",
                    tofile=f"b/{name}",
                )
            )
    (directory / "diff.patch").write_text("".join(diffs), encoding="utf-8")
    (directory / "failure.json").write_text(
        json.dumps(failure, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )
    return directory


def revert_spec(root: Path, spec_name: str) -> Path:
    """Restore feature SQL from the first recorded attempt's ``before`` copies."""

    base = root / "runs" / "dataset-repair" / spec_name
    if not base.is_dir():
        raise DatasetRepairError(
            "escalate",
            {"kind": "contract", "message": f"no dataset repair snapshots for {spec_name}"},
        )
    attempts = sorted(path for path in base.iterdir() if path.is_dir() and path.name.isdigit())
    if not attempts:
        raise DatasetRepairError(
            "escalate",
            {"kind": "contract", "message": f"no dataset repair snapshots for {spec_name}"},
        )
    before = attempts[0] / "before"
    target = root / "datasets" / spec_name / "features"
    target.mkdir(parents=True, exist_ok=True)
    restored = list(before.glob("*.sql"))
    if not restored:
        raise DatasetRepairError(
            "escalate",
            {"kind": "contract", "message": f"repair snapshot {before} has no feature SQL"},
        )
    for path in restored:
        (target / path.name).write_text(path.read_text(encoding="utf-8"), encoding="utf-8")
    return attempts[0]


def _invoke_fixer(root: Path, brief: Brief, spec: DatasetSpec, failure: dict) -> dict:
    note = json.dumps(
        {
            key: failure.get(key)
            for key in ("kind", "message", "fragment", "fragment_path", "examples", "traceback")
        },
        default=str,
    )
    inputs = [str(spec.root / group.sql) for group in spec.feature_groups.values() if spec.root]
    return run_role(
        "dataset_fixer",
        brief,
        root,
        run_id=f"dataset-repair-{spec.name}",
        note=note,
        inputs=inputs,
    )


def repair_features(
    root: Path,
    spec_path: str,
    brief: Brief | None = None,
    *,
    remote: bool = True,
) -> dict:
    """Preflight, repair feature SQL, redeploy, and preflight again.

    Stops when the slice passes. Does not start the full dataset build.
    """

    attempts, dbu_cap = repair_caps()
    path = Path(spec_path)
    if not path.is_absolute():
        path = root / path
    spec = load_spec(path)
    if spec.root is None:
        raise DatasetRepairError(
            "escalate",
            {"kind": "contract", "message": "spec root is not set"},
        )
    yaml_bytes = (spec.root / "dataset.yaml").read_bytes()
    original_columns = feature_columns(spec)
    fixer_brief = brief or Brief.model_validate(
        {
            "name": spec.name,
            "dataset_spec": f"datasets/{spec.name}",
            "budget": {"total_dbu": max(dbu_cap, 1)},
        }
    )
    spent = 0.0
    fixes = 0
    while True:
        cost = slice_dbu() if remote else 0.0
        if spent + cost > dbu_cap + 1e-9:
            raise DatasetRepairError(
                "escalate",
                {
                    "kind": "budget",
                    "message": (
                        f"dataset repair DBU cap {dbu_cap} reached "
                        f"({spent:.4f} spent, next slice {cost:.4f})"
                    ),
                },
            )
        outcome = run_preflight(root, spec_path, brief, remote=remote)
        spent += cost
        if outcome["status"] == "pass":
            return {"status": "pass", "fixes": fixes, "dbu": round(spent, 6), **outcome}
        failure = dict(outcome.get("failure") or {})
        if classify_failure(failure) != "repairable":
            raise DatasetRepairError("escalate", failure)
        if fixes >= attempts:
            failure = {
                **failure,
                "message": f"repair attempt cap {attempts} reached: {failure.get('message')}",
            }
            raise DatasetRepairError("escalate", failure)
        current = load_spec(spec.root)
        before = _read_features(current)
        output = _invoke_fixer(root, fixer_brief, current, failure)
        try:
            apply_fixer_output(current, output if isinstance(output, dict) else {})
            validate_repair(spec.root, yaml_bytes, original_columns)
        except DatasetRepairError:
            _restore_features(spec.root, before)
            raise
        fixes += 1
        after = _read_features(load_spec(spec.root))
        record_attempt(root, spec.name, fixes, before, after, failure)
        deploy(root)
