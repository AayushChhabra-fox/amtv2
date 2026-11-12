"""Confirmation submissions.

The top 3 promoted nodes are each retrained for seeds 0, 1 and 2 and scored on
``select_holdout``. The split is fixed here. ``confirmation_submissions`` runs
only when the caller is ``amt/orchestrator.py``. The Databricks job hardcodes
the same split, and ``harness/train.py`` refuses ``--confirmation`` unless the
cluster was started as that job.
"""

from __future__ import annotations

import inspect
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

CONFIRM_SPLIT = "select_holdout"
CONFIRM_RUNG = "R3"
CONFIRM_SEEDS: tuple[int, ...] = (0, 1, 2)
CONFIRM_TOP_N = 3


class ConfirmNotAllowed(PermissionError):
    """A caller other than the orchestrator tried to open confirmation."""


@dataclass(frozen=True)
class ConfirmationSubmission:
    """One confirm-job run. ``split`` is not a job parameter."""

    candidate_id: str
    seed: str
    rung: str
    split: str
    batch: str
    spec: str
    run_id: str
    family: str
    git_sha: str
    catalog: str = "product_api_dev"
    schema: str = "amt"

    def job_params(self) -> dict[str, str]:
        """Parameters the confirm job accepts. Split and rung stay off this map."""

        return {
            "candidate_id": self.candidate_id,
            "seed": self.seed,
            "batch": self.batch,
            "spec": self.spec,
            "baseline": "",
            "reference_score": "production",
            "model": "",
            "catalog": self.catalog,
            "schema": self.schema,
            "dataset_dir": "",
            "experiment_name": "/Shared/amt/nfy_article_rerank",
            "git_sha": self.git_sha,
            "family": self.family,
            "run_id": self.run_id,
        }


def confirmation_submissions(
    promoted: Sequence[str],
    *,
    batch: str,
    spec: str,
    run_id: str,
    families: Mapping[str, str] | None = None,
    git_sha: str = "unknown",
    catalog: str = "product_api_dev",
    schema: str = "amt",
) -> list[ConfirmationSubmission]:
    """Plan top-3 × 3-seed confirmation. Raises unless the orchestrator called."""

    _assert_orchestrator()
    chosen = _top_promoted(promoted)
    family_map = families or {}
    submissions: list[ConfirmationSubmission] = []
    for node_id in chosen:
        for seed in CONFIRM_SEEDS:
            submissions.append(
                ConfirmationSubmission(
                    candidate_id=node_id,
                    seed=str(seed),
                    rung=CONFIRM_RUNG,
                    split=CONFIRM_SPLIT,
                    batch=batch,
                    spec=spec,
                    run_id=run_id,
                    family=family_map.get(node_id, "candidate"),
                    git_sha=git_sha,
                    catalog=catalog,
                    schema=schema,
                )
            )
    return submissions


def _top_promoted(promoted: Sequence[str]) -> list[str]:
    chosen: list[str] = []
    for node_id in promoted:
        if node_id in chosen:
            continue
        chosen.append(node_id)
        if len(chosen) == CONFIRM_TOP_N:
            break
    return chosen


def _assert_orchestrator() -> None:
    for frame in inspect.stack():
        path = Path(frame.filename).resolve()
        if path.name == "orchestrator.py" and path.parent.name == "amt":
            return
    raise ConfirmNotAllowed("the confirm job is triggered only by the orchestrator")
