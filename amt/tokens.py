"""One-time confirmation tokens.

Holdout splits are refused unless the caller passes a token file that the
orchestrator or ``amt final`` just wrote. The file is deleted after a
successful score, so it cannot be reused. This replaces a cluster environment
variable, which an existing cluster cannot set per run.
"""

from __future__ import annotations

import uuid
from pathlib import Path

from amt.config import Settings, load_settings


def issue_token(root: Path, *, settings: Settings | None = None) -> str:
    settings = settings or load_settings(root)
    name = uuid.uuid4().hex
    if settings.jobs_backend == "fake":
        path = root / ".amt" / "volume" / "confirm_tokens" / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("once\n", encoding="utf-8")
        return str(path)
    import subprocess

    from amt.jobs_backend import databricks_bin

    remote = settings.volume("confirm_tokens", name)
    local = root / ".amt" / "tokens" / name
    local.parent.mkdir(parents=True, exist_ok=True)
    local.write_text("once\n", encoding="utf-8")
    copied = subprocess.run(
        [databricks_bin(), "fs", "cp", str(local), remote],
        cwd=root,
        check=False,
    )
    if copied.returncode != 0:
        raise RuntimeError(f"could not write confirmation token {remote}")
    return remote


def consume_token(path: str) -> None:
    """Delete a local token. Remote tokens are removed by the training job."""

    local = Path(path)
    if local.is_file():
        local.unlink()
