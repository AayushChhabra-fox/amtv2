"""Paths for keyed baseline scores, tied to the dataset manifest hash."""

from __future__ import annotations

import json
from pathlib import Path

from amt.config import Settings, load_settings


def manifest_hash(root: Path, spec_name: str) -> str:
    """Hash recorded by ``amt dataset build``. Empty when the manifest is local-only."""

    named = (
        root / "runs" / spec_name / "manifest.json",
        root / "runs" / f"round0-{spec_name}" / "manifest.json",
    )
    for candidate in named:
        if not candidate.is_file():
            continue
        payload = json.loads(candidate.read_text(encoding="utf-8"))
        value = str(payload.get("content_hash") or payload.get("spec_hash") or "")
        if value:
            return value
    ready = root / "gates" / "ready"
    if ready.is_dir():
        for candidate in sorted(ready.glob("*/manifest.json")):
            payload = json.loads(candidate.read_text(encoding="utf-8"))
            if payload.get("dataset") != spec_name:
                continue
            value = str(payload.get("spec_hash") or payload.get("content_hash") or "")
            if value:
                return value
    return "unhashed"


def scores_path(
    root: Path,
    spec_name: str,
    baseline: str,
    rung: str,
    seed: int,
    *,
    settings: Settings | None = None,
) -> str:
    settings = settings or load_settings(root)
    relative = (
        f"{spec_name}/{manifest_hash(root, spec_name)}/baseline_scores/"
        f"{baseline}/{rung}/seed={seed}.parquet"
    )
    if settings.jobs_backend == "fake":
        return str(root / ".amt" / "volume" / relative)
    return settings.volume(relative)


def scores_exist(path: str, root: Path | None = None) -> bool:
    if path.startswith("/Volumes/"):
        from amt import volume

        return volume.exists(root or Path.cwd(), path)
    return Path(path).is_file()
