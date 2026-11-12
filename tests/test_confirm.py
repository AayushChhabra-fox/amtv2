"""Confirmation is top-3 times 3 seeds, and only the orchestrator may open it."""

from pathlib import Path

import pytest

from amt.confirm import ConfirmNotAllowed, confirmation_submissions
from amt.orchestrator import request_confirmation
from harness.train import _guard_confirmation_caller

ROOT = Path(__file__).resolve().parents[1]


def test_direct_call_is_refused() -> None:
    with pytest.raises(ConfirmNotAllowed, match="orchestrator"):
        confirmation_submissions(
            ["c1"],
            batch="nfy",
            spec="datasets/nfy_article_rerank",
            run_id="r",
        )


def test_orchestrator_plans_top_three_and_three_seeds() -> None:
    planned = request_confirmation(
        ["c1", "c2", "c1", "c3", "c4"],
        batch="nfy",
        spec="datasets/nfy_article_rerank",
        run_id="run-1",
        families={"c1": "boost", "c2": "neural", "c3": "linear"},
        git_sha="abc123",
    )
    assert len(planned) == 9
    assert {item.candidate_id for item in planned} == {"c1", "c2", "c3"}
    assert {item.seed for item in planned} == {"0", "1", "2"}
    assert {item.split for item in planned} == {"select_holdout"}
    assert {item.rung for item in planned} == {"R3"}
    params = planned[0].job_params()
    assert "split" not in params
    assert "rung" not in params
    assert params["candidate_id"] == "c1"
    assert params["spec"] == "datasets/nfy_article_rerank"
    assert params["run_id"] == "run-1"
    assert params["git_sha"] == "abc123"
    by_node = {item.candidate_id: item.family for item in planned}
    assert by_node == {"c1": "boost", "c2": "neural", "c3": "linear"}


def test_orchestrator_plans_nothing_when_nobody_was_promoted() -> None:
    assert request_confirmation([], batch="nfy", spec="datasets/toy", run_id="r") == []


def test_train_refuses_confirmation_without_the_orchestrator_job(monkeypatch) -> None:
    monkeypatch.delenv("AMT_CONFIRM_CALLER", raising=False)
    with pytest.raises(SystemExit, match="orchestrator"):
        _guard_confirmation_caller(True)
    _guard_confirmation_caller(False)
    monkeypatch.setenv("AMT_CONFIRM_CALLER", "orchestrator")
    _guard_confirmation_caller(True)


def test_confirm_job_is_marked_as_the_orchestrator_and_locks_the_split() -> None:
    text = (ROOT / "resources" / "confirm.job.yml").read_text()
    assert "AMT_CONFIRM_CALLER: orchestrator" in text
    assert "--confirmation" in text
    assert "select_holdout" in text
    assert "final_holdout" not in text
