"""Static checks, then the R0 smoke test, before any Databricks submission.

Static checks reject Spark and dbutils, table names, network or subprocess
calls, file paths outside the candidate directory, undeclared third-party
imports, and any mention of a holdout split. A pinned ``requirements.txt``
is installed by the harness and those import names are then allowed.
"""

from __future__ import annotations

import ast
import importlib.util
import re
import sys
import time
from pathlib import Path

import numpy as np

from harness.deps import RequirementsError, env_root, import_names, install
from harness.interface import FitContext, RankingFrame
from harness.registry import (
    BASE_IMPORTS,
    DENIED_IMPORTS,
    HF_DOWNLOAD_CALLS,
    offline_weights_allowed,
)

_HOLDOUTS = ("select_holdout", "final_holdout")
_TABLE = re.compile(r"\b[A-Za-z_][A-Za-z0-9_]*\.[A-Za-z_][A-Za-z0-9_]*\.[A-Za-z_][A-Za-z0-9_]*\b")
_NAME_BAN = frozenset({"spark", "dbutils"})


class PreflightError(RuntimeError):
    """Static checks or the smoke test rejected the candidate."""

    def __init__(self, violations: list[str]) -> None:
        self.violations = violations
        super().__init__("; ".join(violations))


def static_check(
    candidate_dir: Path, *, extra_imports: frozenset[str] | None = None
) -> list[str]:
    """Return human-readable violations. An empty list means the code is allowed."""

    allowed = BASE_IMPORTS | frozenset(extra_imports or ())
    violations: list[str] = []
    sources = sorted(candidate_dir.glob("*.py"))
    if not sources:
        return ["model.py is missing"]
    for path in sources:
        text = path.read_text(encoding="utf-8")
        rel = path.name
        for token in _HOLDOUTS:
            if token in text:
                violations.append(f"{rel} mentions holdout split {token}")
        try:
            tree = ast.parse(text)
        except SyntaxError as exc:
            violations.append(f"{rel} syntax error: {exc.msg}")
            continue
        violations.extend(_walk(tree, rel, allowed))
    return violations


def preflight_candidate(candidate_dir: Path) -> dict[str, float]:
    """Install declared pins, run static checks, then the smoke fit."""

    extra: frozenset[str] = frozenset()
    target = None
    if (candidate_dir / "requirements.txt").is_file():
        try:
            target = install(candidate_dir, env_root())
            extra = import_names(target)
        except RequirementsError as exc:
            raise PreflightError([str(exc)]) from exc
    violations = static_check(candidate_dir, extra_imports=extra)
    if violations:
        raise PreflightError(violations)
    inserted = False
    if target is not None:
        location = str(target)
        if location not in sys.path:
            sys.path.insert(0, location)
            inserted = True
    try:
        smoke = smoke_candidate(candidate_dir)
    except Exception as exc:
        raise PreflightError([str(exc)]) from exc
    finally:
        if inserted and sys.path and sys.path[0] == str(target):
            sys.path.pop(0)
    smoke["static"] = 1.0
    return smoke


def _walk(tree: ast.AST, rel: str, allowed: frozenset[str]) -> list[str]:
    violations: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                violations.extend(_import(rel, alias.name, allowed))
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                violations.extend(_import(rel, node.module, allowed))
        elif isinstance(node, ast.Name) and node.id in _NAME_BAN:
            violations.append(f"{rel} references {node.id}")
        elif isinstance(node, ast.Attribute) and node.attr in _NAME_BAN:
            violations.append(f"{rel} references {node.attr}")
        elif isinstance(node, ast.Call):
            violations.extend(_call(node, rel))
            violations.extend(_hub_download(node, rel))
        elif isinstance(node, ast.Constant) and isinstance(node.value, str):
            if _TABLE.search(node.value):
                violations.append(f"{rel} names a table ({node.value})")
    return violations


def _import(rel: str, module: str, allowed: frozenset[str]) -> list[str]:
    root = module.split(".", 1)[0]
    if root in DENIED_IMPORTS:
        return [f"{rel} imports {root}, which is blocked"]
    if root in allowed:
        return []
    if root in sys.stdlib_module_names or root in sys.builtin_module_names:
        return []
    return [f"{rel} imports {root}, which is not in the base set or requirements.txt"]


def _call(node: ast.Call, rel: str) -> list[str]:
    name = _call_name(node.func)
    if name in {"eval", "exec", "__import__", "system", "popen"}:
        return [f"{rel} calls {name}"]
    if name != "open" and not name.endswith(".open"):
        if name == "Path" or name.endswith(".Path"):
            return _path_call(node, rel)
        return []
    if not node.args:
        return [f"{rel} calls open() without a path"]
    return _path_arg(node.args[0], rel, "open")


def _path_call(node: ast.Call, rel: str) -> list[str]:
    if not node.args:
        return []
    return _path_arg(node.args[0], rel, "Path")


def _path_arg(arg: ast.AST, rel: str, label: str) -> list[str]:
    if isinstance(arg, ast.Constant) and isinstance(arg.value, str):
        if _unsafe_path(arg.value):
            return [f"{rel} {label}() targets {arg.value}, outside the candidate directory"]
        return []
    if isinstance(arg, ast.JoinedStr):
        return [f"{rel} {label}() uses a formatted path"]
    return []


def _unsafe_path(value: str) -> bool:
    text = value.replace("\\", "/")
    if text.startswith("/") or text.startswith("~"):
        return True
    parts = [part for part in text.split("/") if part not in ("", ".")]
    if ".." in parts:
        return True
    protected = {"harness", "amt", "briefs", "datasets", "resources"}
    if parts and parts[0] in protected:
        return True
    if "databricks.yml" in parts:
        return True
    return False


def _call_name(func: ast.AST) -> str:
    if isinstance(func, ast.Name):
        return func.id
    if isinstance(func, ast.Attribute):
        base = _call_name(func.value)
        return f"{base}.{func.attr}" if base else func.attr
    return ""


def _hub_download(node: ast.Call, rel: str) -> list[str]:
    name = _call_name(node.func)
    leaf = name.rsplit(".", 1)[-1]
    if leaf not in HF_DOWNLOAD_CALLS:
        return []
    if offline_weights_allowed():
        return []
    return [
        f"{rel} calls {leaf}, which downloads weights; "
        "set AMT_HF_OFFLINE_CACHE to a /Volumes path"
    ]


def smoke_frame() -> RankingFrame:
    """120 rows, 6 variable-size queries, numeric NaNs, and missing categories."""

    sizes = (5, 10, 15, 20, 30, 40)
    rows: list[dict] = []
    cursor = 0
    for query_index, size in enumerate(sizes):
        for offset in range(size):
            numeric = float(query_index) + offset / 10.0
            missing_number = cursor % 7 == 0
            missing_category = cursor % 5 == 0
            rows.append(
                {
                    "query_id": f"q{query_index}",
                    "item_id": f"i{cursor:03d}",
                    "feat__num__signal": float("nan") if missing_number else numeric,
                    "feat__num__bias": float("nan") if missing_number else 1.0,
                    "feat__cat__color": None
                    if missing_category
                    else ("red", "blue", "green")[cursor % 3],
                    "ref__production": numeric / 10.0,
                    "label": cursor % 4,
                }
            )
            cursor += 1
    frame = RankingFrame(rows)
    if len(frame) != 120 or len(frame.groups()) != 6:
        raise RuntimeError("smoke frame is not 120 rows in 6 query groups")
    if set(frame.group_sizes()) != set(sizes):
        raise RuntimeError(f"smoke groups are not variable: {frame.group_sizes()}")
    if not frame.numeric_feature_names() or not frame.categorical_feature_names():
        raise RuntimeError("smoke frame is missing numeric or categorical features")
    if "ref__production" not in frame.columns:
        raise RuntimeError("smoke frame is missing ref__production")
    return frame


def smoke_candidate(candidate_dir: Path) -> dict[str, float]:
    """Import Ranker, fit the contract smoke frame, and require a stable score."""

    model_path = candidate_dir / "model.py"
    if not model_path.is_file():
        raise FileNotFoundError(model_path)
    spec = importlib.util.spec_from_file_location(f"amt_candidate_{candidate_dir.name}", model_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot import {model_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if not hasattr(module, "Ranker"):
        raise ImportError(f"{model_path} does not define Ranker")
    frame = smoke_frame()
    hidden = frame.without_labels()
    deadline = time.monotonic() + 30
    first = module.Ranker()
    first.fit(frame, frame, FitContext(seed=0, deadline=deadline))
    scores = np.asarray(first.score(hidden), dtype=float)
    _require_scores(scores, len(frame))
    path = candidate_dir / ".smoke_ranker.pkl"
    try:
        first.save(str(path))
        loaded = module.Ranker.load(str(path))
        restored = np.asarray(loaded.score(hidden), dtype=float)
    finally:
        path.unlink(missing_ok=True)
    _require_scores(restored, len(frame))
    if not np.allclose(scores, restored):
        raise RuntimeError("smoke load did not reproduce scores")
    second = module.Ranker()
    second.fit(frame, frame, FitContext(seed=0, deadline=deadline))
    again = np.asarray(second.score(hidden), dtype=float)
    _require_scores(again, len(frame))
    if not np.allclose(scores, again):
        raise RuntimeError("smoke fit is not deterministic")
    return {"rows": float(len(frame)), "finite": 1.0, "deterministic": 1.0}


def _require_scores(scores: np.ndarray, rows: int) -> None:
    if scores.shape != (rows,) or not np.isfinite(scores).all():
        raise RuntimeError("smoke scores are missing or non-finite")
