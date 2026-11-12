"""Bundle deploy and the harness freshness check.

``.amt/deployed.json`` records the git SHA and a hash of ``harness/``,
``datasets/``, and ``resources/``. A Databricks submit refuses to run when
that hash no longer matches the tree, so a laptop edit cannot score against
a stale workspace copy.
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
from pathlib import Path

from amt.config import Settings, load_settings
from amt.jobs_backend import databricks_bin

HASH_ROOTS = ("harness", "datasets", "resources")


class DeployError(RuntimeError):
    """The workspace copy is missing or older than the local tree."""


def tree_hash(root: Path) -> str:
    digest = hashlib.sha256()
    for folder in HASH_ROOTS:
        base = root / folder
        if not base.is_dir():
            continue
        files = sorted(
            path
            for path in base.rglob("*")
            if path.is_file() and path.suffix not in {".pyc"} and "__pycache__" not in path.parts
        )
        for path in files:
            digest.update(path.relative_to(root).as_posix().encode())
            digest.update(b"\0")
            digest.update(path.read_bytes())
            digest.update(b"\0")
    return digest.hexdigest()


def deployed_path(root: Path) -> Path:
    return root / ".amt" / "deployed.json"


def read_deployed(root: Path) -> dict:
    path = deployed_path(root)
    if not path.is_file():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def assert_fresh(root: Path) -> None:
    """Raise when the local harness tree is not what the last deploy shipped."""

    recorded = read_deployed(root)
    current = tree_hash(root)
    if recorded.get("harness_tree_hash") != current:
        raise DeployError("redeploy: harness changed since deploy")


def write_deployed(root: Path, *, git_sha: str) -> Path:
    path = deployed_path(root)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"git_sha": git_sha, "harness_tree_hash": tree_hash(root)}
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return path


def deploy(root: Path, *, settings: Settings | None = None) -> Path:
    """Build the wheel, deploy the bundle with the current settings, and record the hash."""

    settings = settings or load_settings(root)
    wheel = subprocess.run(["uv", "build", "--wheel"], cwd=root, check=False)
    if wheel.returncode != 0:
        raise DeployError("uv build --wheel failed")
    command = [
        databricks_bin(),
        "bundle",
        "deploy",
        "-t",
        settings.bundle_target,
        "--auto-approve",
        "--var",
        f"catalog={settings.catalog}",
        "--var",
        f"schema={settings.schema_name}",
        "--var",
        f"artifact_volume={settings.artifact_volume}",
        "--var",
        f"experiment_name={settings.experiment_root}/amt",
        "--var",
        f"cpu_node_type_id={settings.node_type}",
        "--var",
        f"spark_version={settings.spark_version}",
    ]
    completed = subprocess.run(command, cwd=root, check=False)
    if completed.returncode != 0:
        raise DeployError("bundle deploy failed")
    return write_deployed(root, git_sha=_git_sha(root))


def _git_sha(root: Path) -> str:
    override = os.environ.get("AMT_GIT_SHA", "").strip()
    if override:
        return override
    completed = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=root,
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        return "unknown"
    return completed.stdout.strip() or "unknown"
