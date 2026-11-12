"""Role prompts handed to an agent. Phase 1 M5 replaces the one-brief view."""

from __future__ import annotations

from pathlib import Path

from amt.schemas import Brief


def build_context(role: str, brief: Brief, root: Path) -> str:
    del root
    metric = "ndcg@10_graded"
    if brief.problem.metric is not None:
        metric = brief.problem.metric.name
    pinned = brief.candidate
    candidate_id = pinned.id if pinned else "c00"
    family = pinned.family if pinned else "linear"
    return (
        f"Role: {role}\n"
        f"Brief: {brief.name}\n"
        f"Metric: {metric}\n"
        f"Candidate: {candidate_id} family={family}\n"
        "Write scope for the implementer is candidates/<id>/ only.\n"
    )
