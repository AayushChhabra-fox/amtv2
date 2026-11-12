"""Read-only check that auth, the workspace, and the agent CLI are usable."""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

from amt.config import Settings, load_settings
from amt.jobs_backend import databricks_bin


def cheap(settings: Settings | None = None) -> list[str]:
    """Failures that should stop a submit before any job is launched."""

    settings = settings or load_settings()
    if settings.jobs_backend == "fake":
        return []
    problems: list[str] = []
    token = os.environ.get("DATABRICKS_TOKEN")
    client = os.environ.get("DATABRICKS_CLIENT_ID")
    profile = os.environ.get("DATABRICKS_CONFIG_PROFILE")
    if not (token or client or profile):
        problems.append("no Databricks auth: set a token, a service principal, or a profile")
    return problems


def report(root: Path, *, runner=None) -> str:
    """Full check. ``runner`` accepts an argument list and returns stdout, for tests."""

    settings = load_settings(root)
    lines = [f"auth: {_auth_method()}", f"backend: {settings.jobs_backend}"]
    if settings.jobs_backend == "fake":
        lines.append("workspace: skipped (fake backend)")
        lines.append(_agent_line())
        return "\n".join(lines)
    execute = runner or _live
    lines.append(
        _check("workspace", execute, [databricks_bin(), "current-user", "me", "-o", "json"])
    )
    lines.append(
        _check(
            "catalog",
            execute,
            [databricks_bin(), "catalogs", "get", settings.catalog, "-o", "json"],
        )
    )
    if settings.cluster_id:
        lines.append(
            _check(
                "cluster",
                execute,
                [
                    databricks_bin(),
                    "api",
                    "get",
                    f"/api/2.0/clusters/get?cluster_id={settings.cluster_id}",
                ],
            )
        )
    if settings.gpu_cluster_id:
        lines.append(
            _check(
                "gpu cluster",
                execute,
                [
                    databricks_bin(),
                    "api",
                    "get",
                    f"/api/2.0/clusters/get?cluster_id={settings.gpu_cluster_id}",
                ],
            )
        )
    lines.append(_agent_line())
    return "\n".join(lines)


def _auth_method() -> str:
    if os.environ.get("DATABRICKS_CLIENT_ID"):
        return "service principal"
    if os.environ.get("DATABRICKS_TOKEN"):
        return "personal access token"
    if os.environ.get("DATABRICKS_CONFIG_PROFILE"):
        return f"profile {os.environ['DATABRICKS_CONFIG_PROFILE']}"
    return "none"


def _agent_line() -> str:
    if shutil.which("cursor-agent") is None:
        return "cursor-agent: not installed"
    completed = subprocess.run(
        ["cursor-agent", "status"],
        check=False,
        capture_output=True,
        text=True,
    )
    text = (completed.stdout or completed.stderr or "").strip().splitlines()
    detail = text[0] if text else f"exit {completed.returncode}"
    return f"cursor-agent: {detail}"


def _check(label: str, execute, args: list[str]) -> str:
    try:
        execute(args)
    except Exception as exc:
        return f"{label}: FAIL {exc}"
    return f"{label}: ok"


def _live(args: list[str]) -> str:
    completed = subprocess.run(args, check=False, capture_output=True, text=True)
    if completed.returncode != 0:
        detail = (completed.stderr or completed.stdout or "failed").strip()
        raise RuntimeError(detail)
    return completed.stdout
