"""Artifact-volume I/O through the Databricks CLI.

``Settings.volume`` returns ``/Volumes/...`` paths, which jobs open as files.
The CLI reads a bare path as local, so ``fs`` commands get ``dbfs:/Volumes/...``.
The token needs the ``files`` scope.
"""

from __future__ import annotations

import subprocess
from pathlib import Path, PurePosixPath

from amt.jobs_backend import databricks_bin


class VolumeError(RuntimeError):
    """A volume call failed for a reason other than a missing path."""


def cli_uri(path: str) -> str:
    return path if path.startswith("dbfs:") else f"dbfs:{path}"


def put(root: Path, local: Path, remote: str, *, overwrite: bool = False) -> None:
    parent = str(PurePosixPath(remote.removeprefix("dbfs:")).parent)
    made = subprocess.run(
        [databricks_bin(), "fs", "mkdir", cli_uri(parent)],
        cwd=root,
        check=False,
        capture_output=True,
        text=True,
    )
    if made.returncode != 0:
        raise VolumeError(f"creating {parent} failed: {_detail(made)}")
    args = [databricks_bin(), "fs", "cp", str(local), cli_uri(remote)]
    if overwrite:
        args.append("--overwrite")
    done = subprocess.run(args, cwd=root, check=False, capture_output=True, text=True)
    if done.returncode != 0:
        raise VolumeError(f"upload to {remote} failed: {_detail(done)}")


def exists(root: Path, remote: str) -> bool:
    path = PurePosixPath(remote.removeprefix("dbfs:"))
    done = subprocess.run(
        [databricks_bin(), "fs", "ls", cli_uri(str(path.parent))],
        cwd=root,
        check=False,
        capture_output=True,
        text=True,
    )
    if done.returncode != 0:
        detail = _detail(done)
        if "not found" in detail.lower() or "does not exist" in detail.lower():
            return False
        raise VolumeError(f"listing {path.parent} failed: {detail}")
    return path.name in {line.strip().rstrip("/") for line in done.stdout.splitlines()}


def _detail(done: subprocess.CompletedProcess) -> str:
    return (done.stderr or done.stdout or "databricks fs failed").strip()
