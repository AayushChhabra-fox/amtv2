"""Role prompts handed to an agent. Phase 1 M5 replaces the one-brief view."""

from __future__ import annotations

from pathlib import Path

from amt.schemas import Brief


def build_context(role: str, brief: Brief, root: Path) -> str:
    del root
    return (
        f"Role: {role}\n"
        f"Brief: {brief.name}\n"
        f"Metric: {brief.problem.get('metric', {}).get('name', 'ndcg@10_graded')}\n"
        f"Candidate: {brief.candidate.get('id', 'c00')} "
        f"family={brief.candidate.get('family', 'linear')}\n"
        "Write scope for the implementer is candidates/<id>/ only.\n"
    )
