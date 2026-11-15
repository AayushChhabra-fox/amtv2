"""Run-scoped paths. Candidate code for one search never shares a directory with another."""

from __future__ import annotations

from pathlib import Path


def candidate_dir(root: Path, run_id: str, node_id: str) -> Path:
    """``runs/<run_id>/candidates/<node_id>/``."""

    return root / "runs" / run_id / "candidates" / node_id
