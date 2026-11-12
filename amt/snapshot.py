"""Immutable copies of candidate code on the artifact volume.

The train job runs from the bundle that was deployed, which does not contain
candidates written after that deploy. Each submit uploads ``model.py`` (and
``requirements.txt`` when present) to a path that includes the git SHA. An
existing snapshot is never overwritten.
"""

from __future__ import annotations

import shutil
from pathlib import Path

from amt import volume
from amt.config import Settings, load_settings


class SnapshotError(RuntimeError):
    """The candidate directory cannot be snapshotted."""


def upload_candidate(
    root: Path,
    run_id: str,
    node_id: str,
    git_sha: str,
    *,
    settings: Settings | None = None,
) -> str:
    """Copy the candidate and return the remote ``model.py`` path."""

    settings = settings or load_settings(root)
    source = root / "candidates" / node_id
    model = source / "model.py"
    if not model.is_file():
        raise SnapshotError(f"{node_id} has no model.py")
    relative = f"candidates/{run_id}/{node_id}/{git_sha}"
    if settings.jobs_backend == "fake":
        return _copy_local(root / ".amt" / "volume" / relative, source)
    remote = settings.volume(relative)
    _copy_volume(root, source, remote)
    return f"{remote}/model.py"


def _copy_local(dest: Path, source: Path) -> str:
    target = dest / "model.py"
    if target.is_file():
        return str(target)
    dest.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source / "model.py", target)
    requirements = source / "requirements.txt"
    if requirements.is_file():
        shutil.copy2(requirements, dest / "requirements.txt")
    return str(target)


def _copy_volume(root: Path, source: Path, remote: str) -> None:
    if volume.exists(root, f"{remote}/model.py"):
        return
    for name in ("model.py", "requirements.txt"):
        local = source / name
        if not local.is_file():
            continue
        try:
            volume.put(root, local, f"{remote}/{name}")
        except volume.VolumeError as exc:
            raise SnapshotError(str(exc)) from exc
