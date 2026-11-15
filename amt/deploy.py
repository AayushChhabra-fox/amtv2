"""Bundle deploy and the harness freshness check.

``.amt/deployed.json`` records the git SHA, a hash of ``harness/``,
``resources/``, and ``amt/``, and a hash per ``datasets/<spec>/`` directory.
A Databricks submit refuses to run when the harness hash no longer matches,
so a laptop edit of training code cannot score against a stale workspace
copy. Editing one dataset spec does not change the harness hash or another
spec's freshness token.
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
from pathlib import Path

from amt.config import Settings, load_settings
from amt.jobs_backend import databricks_bin

HARNESS_ROOTS = ("harness", "resources", "amt")


class DeployError(RuntimeError):
    """The workspace copy is missing or older than the local tree."""


def _hash_files(root: Path, folders: tuple[str, ...]) -> str:
    digest = hashlib.sha256()
    for folder in folders:
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


def harness_hash(root: Path) -> str:
    """Hash of the code a deploy ships, excluding dataset specs."""

    return _hash_files(root, HARNESS_ROOTS)


def spec_hash(root: Path, spec_name: str) -> str:
    """Hash of one ``datasets/<spec>/`` tree."""

    name = Path(spec_name).name
    if not name or name in {".", ".."}:
        return _hash_files(root, ())
    return _hash_files(root, (f"datasets/{name}",))


def spec_hashes(root: Path) -> dict[str, str]:
    base = root / "datasets"
    if not base.is_dir():
        return {}
    found: dict[str, str] = {}
    for path in sorted(base.iterdir()):
        if path.is_dir() and not path.name.startswith("."):
            found[path.name] = spec_hash(root, path.name)
    return found


def freshness_token(root: Path, spec_name: str) -> str:
    """Short token for one spec. Another spec's edit does not change it."""

    return f"{harness_hash(root)[:12]}-{spec_hash(root, spec_name)[:12]}"


def deployed_path(root: Path) -> Path:
    return root / ".amt" / "deployed.json"


def read_deployed(root: Path) -> dict:
    path = deployed_path(root)
    if not path.is_file():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def assert_fresh(root: Path) -> None:
    """Raise when the local harness tree is not what the last deploy shipped.

    Dataset specs are tracked separately. A change under ``datasets/<spec>/``
    does not by itself make this deploy stale.
    """

    recorded = read_deployed(root)
    if recorded.get("harness_hash") != harness_hash(root):
        raise DeployError("redeploy: harness changed since deploy")


def write_deployed(root: Path, *, git_sha: str) -> Path:
    path = deployed_path(root)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "git_sha": git_sha,
        "harness_hash": harness_hash(root),
        "spec_hashes": spec_hashes(root),
    }
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
