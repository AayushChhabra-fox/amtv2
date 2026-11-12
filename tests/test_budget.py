"""Pool, per-candidate cap, and wall-clock DBU estimate."""

import pytest

from amt.budget import (
    BudgetExceeded,
    assert_can_spend,
    estimate_dbu,
    inflight_dbu,
    node_committed_dbu,
)
from amt.schemas import Brief, LedgerRow


def _brief(*, total: float = 10, cap: float | None = None) -> Brief:
    payload: dict = {"budget": {"total_dbu": total}}
    if cap is not None:
        payload["efficiency"] = {"train_dbu_per_candidate": cap}
    return Brief.model_validate(payload)


def test_pool_refuses_only_when_the_next_submit_would_go_negative() -> None:
    brief = _brief(total=10)
    assert_can_spend(brief, 9, 1)
    assert_can_spend(brief, 10, 0)
    with pytest.raises(BudgetExceeded, match="refusing submit"):
        assert_can_spend(brief, 9, 1.1)


def test_candidate_cap_is_independent_of_the_pool() -> None:
    brief = _brief(total=1000, cap=60)
    assert_can_spend(brief, 0, 60, candidate_spent_dbu=0)
    with pytest.raises(BudgetExceeded, match="cap"):
        assert_can_spend(brief, 0, 10, candidate_spent_dbu=55)


def test_estimate_is_wall_clock_times_cluster_rate() -> None:
    assert estimate_dbu(60, dbu_per_hour=0.75) == pytest.approx(0.75)
    assert estimate_dbu(30, workers=2, dbu_per_hour=0.75) == pytest.approx(1.125)
    assert estimate_dbu(60) == pytest.approx(0.75)


def test_inflight_reservations_and_ledger_commitments() -> None:
    jobs = {
        "c1/R1/0": {"dbu_est": 2, "pruned": False},
        "c1/R2/0": {"dbu_est": 3, "pruned": True},
        "c2/R1/0": {"dbu_est": 4, "pruned": False},
    }
    assert inflight_dbu(jobs) == pytest.approx(6)
    assert inflight_dbu(jobs, "c1") == pytest.approx(2)
    row = LedgerRow(
        run_id="r",
        node_id="c1",
        operator="draft",
        family="trees",
        claim="c1",
        rung="R1",
        seed=0,
        status="kept",
        git_sha="abc123",
        dbu=3.5,
        job_run_id="fake-1",
        mlflow_run_id="m1",
    )
    assert node_committed_dbu([row], "r", "c1") == pytest.approx(3.5)
    assert node_committed_dbu([row], "r", "c2") == 0
