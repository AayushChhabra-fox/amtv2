"""Static checks, then the R0 smoke test, before any Databricks submission.

Static checks reject Spark and dbutils, table names, network or subprocess
calls, file paths outside the candidate directory, imports outside the
allowlist, and any mention of a holdout split.
"""

from __future__ import annotations

import ast
import importlib.util
import re
import sys
from pathlib import Path

import numpy as np

ALLOWLIST = frozenset(
    {
        "numpy",
        "pandas",
        "polars",
        "lightgbm",
        "xgboost",
        "torch",
        "sklearn",
        "sentence_transformers",
    }
)
_DENIED = frozenset(
    {
        "pyspark",
        "spark",
        "dbutils",
        "subprocess",
        "socket",
        "requests",
        "urllib",
        "http",
        "ftplib",
        "smtplib",
        "webbrowser",
        "httpx",
        "aiohttp",
        "boto3",
    }
)
_HOLDOUTS = ("select_holdout", "final_holdout")
_TABLE = re.compile(r"\b[A-Za-z_][A-Za-z0-9_]*\.[A-Za-z_][A-Za-z0-9_]*\.[A-Za-z_][A-Za-z0-9_]*\b")
_NAME_BAN = frozenset({"spark", "dbutils"})


class PreflightError(RuntimeError):
    """Static checks or the smoke test rejected the candidate."""

    def __init__(self, violations: list[str]) -> None:
        self.violations = violations
        super().__init__("; ".join(violations))


def static_check(candidate_dir: Path) -> list[str]:
    """Return human-readable violations. An empty list means the code is allowed."""

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
        violations.extend(_walk(tree, rel))
    return violations


def preflight_candidate(candidate_dir: Path) -> dict[str, float]:
    """Run static checks, then the smoke fit. Raises PreflightError on failure."""

    violations = static_check(candidate_dir)
    if violations:
        raise PreflightError(violations)
    try:
        smoke = smoke_candidate(candidate_dir)
    except Exception as exc:
        raise PreflightError([str(exc)]) from exc
    smoke["static"] = 1.0
    return smoke


def _walk(tree: ast.AST, rel: str) -> list[str]:
    violations: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                violations.extend(_import(rel, alias.name))
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                violations.extend(_import(rel, node.module))
        elif isinstance(node, ast.Name) and node.id in _NAME_BAN:
            violations.append(f"{rel} references {node.id}")
        elif isinstance(node, ast.Attribute) and node.attr in _NAME_BAN:
            violations.append(f"{rel} references {node.attr}")
        elif isinstance(node, ast.Call):
            violations.extend(_call(node, rel))
        elif isinstance(node, ast.Constant) and isinstance(node.value, str):
            if _TABLE.search(node.value):
                violations.append(f"{rel} names a table ({node.value})")
    return violations


def _import(rel: str, module: str) -> list[str]:
    root = module.split(".", 1)[0]
    if root in _DENIED:
        return [f"{rel} imports {root}, which is blocked"]
    if root in ALLOWLIST:
        return []
    if root in sys.stdlib_module_names or root in sys.builtin_module_names:
        return []
    return [f"{rel} imports {root}, which is outside the allowlist"]


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


def _log_loss(labels: np.ndarray, scores: np.ndarray) -> float:
    clipped = np.clip(scores, 1e-6, 1 - 1e-6)
    return float(-np.mean(labels * np.log(clipped) + (1 - labels) * np.log(1 - clipped)))


def smoke_candidate(candidate_dir: Path) -> dict[str, float]:
    """Import model.py, fit a separable toy set, and require the loss to fall."""
    model_path = candidate_dir / "model.py"
    if not model_path.is_file():
        raise FileNotFoundError(model_path)
    spec = importlib.util.spec_from_file_location(f"amt_candidate_{candidate_dir.name}", model_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot import {model_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    ranker = module.LogisticRanker()
    rng = np.random.default_rng(0)
    features = rng.normal(size=(80, 3))
    labels = (features[:, 0] + 0.5 * features[:, 1] > 0).astype(float)
    frame = {"X": features, "y": labels}
    ranker.fit(frame, frame, {"seed": 0})
    scores = np.asarray(ranker.score(frame), dtype=float)
    if scores.shape != labels.shape or not np.isfinite(scores).all():
        raise RuntimeError("smoke scores are missing or non-finite")
    baseline = _log_loss(labels, np.full_like(labels, 0.5))
    trained = _log_loss(labels, scores)
    if trained >= baseline:
        raise RuntimeError(f"smoke loss did not decrease: {trained:.4f} >= {baseline:.4f}")
    return {"log_loss": trained, "baseline_log_loss": baseline}
