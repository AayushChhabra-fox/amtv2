"""Replay tests for every section 5.3 pruner rule."""

import json
from pathlib import Path

import pytest
import yaml

from amt.ledger import read_rows
from amt.pruner import (
    HoldoutReadError,
    PrunerError,
    QueryDelta,
    RunView,
    commit_rung,
    decide,
    main,
)
from amt.schemas import Brief, DeltaVsBaseline

ROOT = Path(__file__).resolve().parents[1]


def _brief(
    *,
    min_effect: float = 0.01,
    cap: float | None = None,
    latency: int | None = None,
    total: float = 100,
) -> Brief:
    efficiency: dict = {}
    if cap is not None:
        efficiency["train_dbu_per_candidate"] = cap
    if latency is not None:
        efficiency["serving"] = {"latency_p95_ms_per_100_candidates": latency}
    payload: dict = {"min_effect": min_effect, "budget": {"total_dbu": total}}
    if efficiency:
        payload["efficiency"] = efficiency
    return Brief.model_validate(payload)


def _baseline(rung: str = "R1") -> RunView:
    return RunView(
        node_id="lgbm_reference",
        family="lgbm_reference",
        kind="baseline",
        rung=rung,  # type: ignore[arg-type]
        split="search_val",
        metric=0.40,
        claim="lightgbm reference",
        git_sha="abc123",
        mlflow_run_id="m-base",
        job_run_id="fake-base",
    )


def _run(
    node_id: str,
    *,
    point: float = 0.02,
    lo: float = 0.0,
    hi: float = 0.04,
    family: str = "trees",
    rung: str = "R1",
    **extra: object,
) -> RunView:
    prod_point = extra.pop("prod_point", None)
    prod_lo = extra.pop("prod_lo", None)
    prod_hi = extra.pop("prod_hi", None)
    payload: dict = {
        "node_id": node_id,
        "family": family,
        "rung": rung,
        "split": "search_val",
        "metric": 0.40 + point,
        "delta": DeltaVsBaseline(point=point, ci_lo=lo, ci_hi=hi),
        "dbu": 1.0,
        "latency_p95_ms": 10.0,
        "claim": node_id,
        "git_sha": "abc123",
        "mlflow_run_id": f"m-{node_id}",
        "job_run_id": f"fake-{node_id}",
    }
    if prod_point is not None:
        payload["delta_vs_production"] = DeltaVsBaseline(
            point=float(prod_point),
            ci_lo=float(prod_lo),
            ci_hi=float(prod_hi),
        )
    payload.update(extra)
    return RunView.model_validate(payload)


def _queries(value: float, n_days: int = 4, per_day: int = 3) -> list[QueryDelta]:
    rows: list[QueryDelta] = []
    for day in range(n_days):
        for query in range(per_day):
            rows.append(
                QueryDelta(
                    query_id=f"d{day}q{query}",
                    event_day=f"2026-01-{day + 1:02d}",
                    delta=value,
                )
            )
    return rows


def _by_id(decision):
    return {item.node_id: item for item in decision.decisions}


def test_r0_keeps_a_finished_run_and_does_not_halve() -> None:
    runs = [_run(name, point=0.01 * index, rung="R0") for index, name in enumerate("abcd")]
    decision = decide("R0", runs, [], _brief())
    assert decision.survivors == ["d", "c", "b", "a"]
    assert decision.promoted == []
    assert all(item.rule == "survive" for item in decision.decisions)


def test_missing_metric_fails() -> None:
    run = RunView(
        node_id="c1",
        family="trees",
        rung="R0",
        claim="c1",
        git_sha="abc123",
        job_run_id="fake-1",
        mlflow_run_id="m1",
    )
    decision = decide("R0", [run], [], _brief())
    assert decision.decisions[0].status == "failed"
    assert decision.decisions[0].rule == "no_metrics"
    assert decision.decisions[0].kill_reason == "run produced no metric"
    assert decision.survivors == []


def test_nan_and_divergence_are_hard_kills() -> None:
    nan = _run("nan", rung="R0", metric=float("nan"))
    divergent = _run("div", rung="R0", divergent=True)
    brief = _brief()
    assert decide("R0", [nan], [], brief).decisions[0].rule == "nan"
    assert decide("R0", [divergent], [], brief).decisions[0].rule == "nan"
    assert "nan or divergent" in decide("R0", [nan], [], brief).decisions[0].kill_reason


def test_preflight_and_review_fail_before_a_strong_metric() -> None:
    brief = _brief()
    preflight = _run("p", rung="R0", point=0.2, hi=0.3, preflight="FAIL", metric=None, delta=None)
    review = _run("r", rung="R0", point=0.2, hi=0.3, review="FAIL")
    assert decide("R0", [preflight], [], brief).decisions[0].rule == "preflight"
    assert decide("R0", [review], [], brief).decisions[0].rule == "review"


def test_latency_and_dbu_caps() -> None:
    brief = _brief(cap=60, latency=20)
    slow = _run("slow", rung="R0", latency_p95_ms=21)
    costly = _run("cost", rung="R0", dbu=60.1)
    at_cap = _run("edge", rung="R0", dbu=60, latency_p95_ms=20)
    assert decide("R0", [slow], [], brief).decisions[0].rule == "latency"
    assert decide("R0", [costly], [], brief).decisions[0].rule == "dbu_cap"
    assert decide("R0", [at_cap], [], brief).decisions[0].status == "kept"
    summed = [
        _run("sum", rung="R0", seed=0, dbu=40),
        _run("sum", rung="R0", seed=1, dbu=40, mlflow_run_id="m-sum-1", job_run_id="fake-sum-1"),
    ]
    assert decide("R0", summed, [], brief).decisions[0].rule == "dbu_cap"


def test_one_seed_without_metrics_fails_the_node() -> None:
    good = _run("c", rung="R0", seed=0)
    empty = RunView(
        node_id="c",
        family="trees",
        seed=1,
        rung="R0",
        claim="c",
        git_sha="abc123",
        job_run_id="fake-c1",
        mlflow_run_id="m-c1",
    )
    decision = decide("R0", [good, empty], [], _brief())
    assert decision.decisions[0].rule == "no_metrics"
    assert decision.decisions[0].seeds == [0, 1]


@pytest.mark.parametrize(
    ("rung", "hi", "killed"),
    [
        ("R1", -0.03, True),
        ("R1", -0.02, False),
        ("R1", -0.005, False),
        ("R2", -0.02, True),
        ("R2", -0.01, False),
        ("R2", 0.0, False),
    ],
)
def test_floor_is_c00_relative(rung: str, hi: float, killed: bool) -> None:
    run = _run("c1", point=hi - 0.01, lo=hi - 0.02, hi=hi, rung=rung)
    decision = decide(rung, [run], [_baseline(rung)], _brief(min_effect=0.01))
    item = decision.decisions[0]
    if killed:
        assert item.rule == "below_floor"
        assert item.status == "killed"
        assert "c00" in item.kill_reason
    else:
        assert item.status == "kept"
        assert item.rule != "below_floor"


def test_r1_noise_inside_the_interval_survives() -> None:
    run = _run("c1", point=-0.05, lo=-0.08, hi=-0.005)
    decision = decide("R1", [run], [_baseline()], _brief(min_effect=0.01))
    assert decision.decisions[0].status == "kept"
    assert decision.decisions[0].rule == "survive"


def test_production_gap_does_not_set_the_r1_floor() -> None:
    run = _run(
        "c1",
        point=0.02,
        lo=0.0,
        hi=0.04,
        ref_deltas={"production": DeltaVsBaseline(point=-0.04, ci_lo=-0.06, ci_hi=-0.05)},
        prod_point=-0.04,
        prod_lo=-0.06,
        prod_hi=-0.05,
    )
    decision = decide("R1", [run], [_baseline()], _brief())
    item = decision.decisions[0]
    assert item.status == "kept"
    assert item.rule == "survive"


def test_r1_requires_same_rung_lightgbm_baseline() -> None:
    run = _run("c1")
    with pytest.raises(PrunerError, match="LightGBM"):
        decide("R1", [run], [], _brief())
    with pytest.raises(PrunerError, match="LightGBM"):
        decide("R1", [run], [_baseline("R2")], _brief())


def test_baseline_inside_the_candidate_list_is_not_pruned() -> None:
    run = _run("c1", point=0.02, lo=0.0, hi=0.04)
    decision = decide("R1", [run, _baseline()], [], _brief())
    assert [item.node_id for item in decision.decisions] == ["c1"]
    assert decision.survivors == ["c1"]


def test_paired_dominance_fires_when_marginal_intervals_overlap() -> None:
    """Wide marginal CIs do not dominate. The same gap, paired by query, does."""

    brief = _brief()
    baseline = [_baseline("R2")]
    shared = dict(lo=-0.10, hi=0.10)
    weak = _run("a", point=0.0, dbu=2, latency_p95_ms=15, family="trees", rung="R2", **shared)
    strong = _run("b", point=0.03, dbu=1, latency_p95_ms=10, family="linear", rung="R2", **shared)
    other = _run("c", point=-0.01, dbu=1, latency_p95_ms=8, family="neural", rung="R2", **shared)
    without = decide("R2", [weak, strong, other], baseline, brief)
    assert _by_id(without)["a"].rule == "survive"

    weak_q = weak.model_copy(update={"query_deltas": _queries(0.0), "rung": "R2"})
    strong_q = strong.model_copy(update={"query_deltas": _queries(0.03), "rung": "R2"})
    other_q = other.model_copy(update={"query_deltas": _queries(-0.01), "rung": "R2"})
    at_r1 = decide(
        "R1",
        [item.model_copy(update={"rung": "R1"}) for item in (weak_q, strong_q, other_q)],
        [_baseline("R1")],
        brief,
    )
    assert _by_id(at_r1)["a"].rule == "survive"
    paired = decide("R2", [weak_q, strong_q, other_q], [_baseline("R2")], brief)
    assert _by_id(paired)["a"].rule == "dominance"
    assert "b" in _by_id(paired)["a"].kill_reason
    assert "b" in paired.survivors


def test_dominance_requires_cost_no_higher_and_ignores_a_tie() -> None:
    brief = _brief()
    baseline = [_baseline("R2")]
    shared = dict(lo=-0.10, hi=0.10)
    weak = _run(
        "a",
        point=0.0,
        dbu=2,
        latency_p95_ms=15,
        query_deltas=_queries(0.0),
        rung="R2",
        **shared,
    )
    expensive = _run(
        "b",
        point=0.03,
        dbu=5,
        latency_p95_ms=30,
        family="linear",
        query_deltas=_queries(0.03),
        rung="R2",
        **shared,
    )
    other = _run(
        "c",
        point=-0.01,
        dbu=1,
        latency_p95_ms=8,
        family="neural",
        query_deltas=_queries(-0.01),
        rung="R2",
        **shared,
    )
    decision = decide("R2", [weak, expensive, other], baseline, brief)
    assert _by_id(decision)["a"].rule != "dominance"
    assert "a" in decision.survivors

    tied = [
        _run(
            name,
            point=0.02,
            lo=0.02,
            hi=0.02,
            query_deltas=_queries(0.02),
            family="trees",
            rung="R2",
        )
        for name in ("a", "b")
    ]
    tie = decide("R2", tied, baseline, brief)
    assert all(item.rule != "dominance" for item in tie.decisions)


def test_halving_tie_break_keeps_ceil_half_by_node_id() -> None:
    runs = [
        _run(name, point=0.02, lo=0.0, hi=0.04, rung="R2", family="trees")
        for name in ("c", "a", "b")
    ]
    decision = decide("R2", runs, [_baseline("R2")], _brief())
    assert decision.survivors == ["a", "b"]
    assert _by_id(decision)["c"].rule == "halving"
    assert decision.diversity_slot is None


def test_r1_keeps_nodes_that_are_only_slightly_behind() -> None:
    wide = dict(lo=-0.05, hi=0.10, rung="R1")
    runs = [
        _run("aa", point=0.05, family="boost", **wide),
        _run("ab", point=0.04, family="boost", **wide),
        _run("b", point=0.02, family="neural", **wide),
        _run("c", point=0.01, family="linear", **wide),
    ]
    decision = decide("R1", runs, [_baseline()], _brief())
    assert decision.diversity_slot is None
    assert decision.survivors == ["aa", "ab", "b", "c"]
    assert all(_by_id(decision)[run.node_id].rule == "survive" for run in runs)


def test_r2_diversity_slot_keeps_one_wiped_family() -> None:
    wide = dict(lo=-0.05, hi=0.10, rung="R2")
    runs = [
        _run("aa", point=0.05, family="boost", **wide),
        _run("ab", point=0.04, family="boost", **wide),
        _run("b", point=0.02, family="neural", **wide),
        _run("c", point=0.01, family="linear", **wide),
    ]
    decision = decide("R2", runs, [_baseline("R2")], _brief())
    assert decision.diversity_slot == "b"
    assert decision.survivors == ["aa", "ab", "b"]
    assert _by_id(decision)["b"].rule == "diversity"
    assert _by_id(decision)["b"].status == "kept"
    assert _by_id(decision)["c"].rule == "halving"

    at_r3 = [run.model_copy(update={"rung": "R3"}) for run in runs]
    later = decide("R3", at_r3, [_baseline("R3")], _brief())
    assert later.diversity_slot is None
    assert "b" not in later.survivors
    assert _by_id(later)["b"].rule == "halving"


def test_r3_promotes_top_three_that_beat_production_and_c00() -> None:
    runs = [
        _run(
            f"n{index}",
            point=0.05 - index * 0.01,
            lo=0.005,
            hi=0.06,
            rung="R3",
            prod_point=0.05 - index * 0.01,
            prod_lo=0.005,
            prod_hi=0.06,
        )
        for index in range(5)
    ]
    decision = decide("R3", runs, [_baseline("R3")], _brief())
    assert decision.survivors == ["n0", "n1", "n2"]
    assert decision.promoted == ["n0", "n1", "n2"]
    assert all(_by_id(decision)[name].status == "promoted" for name in decision.promoted)
    assert _by_id(decision)["n3"].rule == "halving"


def test_r3_lower_bound_of_zero_is_not_promoted() -> None:
    run = _run("c1", point=0.01, lo=0.0, hi=0.02, rung="R3")
    decision = decide("R3", [run], [_baseline("R3")], _brief())
    assert decision.decisions[0].status == "kept"
    assert decision.promoted == []


def test_seed_deltas_are_averaged_before_the_floor() -> None:
    low = _run("c1", point=0.0, lo=-0.03, hi=-0.02, seed=0)
    high = _run(
        "c1",
        point=0.04,
        lo=0.01,
        hi=0.02,
        seed=1,
        mlflow_run_id="m-c1-1",
        job_run_id="fake-c1-1",
    )
    decision = decide("R1", [low, high], [_baseline()], _brief())
    assert decision.decisions[0].status == "kept"
    assert decision.decisions[0].delta.point == pytest.approx(0.02)
    assert decision.decisions[0].delta.ci_hi == pytest.approx(0.0)


def test_holdout_split_is_refused_before_metrics_are_read() -> None:
    class HoldoutRun:
        def __init__(self, split: str) -> None:
            self.split = split
            self.node_id = "leak"
            self.reads: list[str] = []

        @property
        def metric(self) -> float:
            self.reads.append("metric")
            return 0.99

        @property
        def delta(self) -> DeltaVsBaseline:
            self.reads.append("delta")
            return DeltaVsBaseline(point=0.5, ci_lo=0.4, ci_hi=0.6)

    for split in ("select_holdout", "final_holdout"):
        run = HoldoutRun(split)
        with pytest.raises(HoldoutReadError, match=split):
            decide("R1", [run], [_baseline()], _brief())
        assert run.reads == []

    class HiddenBaseline:
        split = "final_holdout"
        node_id = "lgbm_reference"
        family = "lgbm_reference"

        def __getattribute__(self, name: str):
            if name in {"metric", "delta", "query_deltas", "ref_deltas"}:
                raise AssertionError(name)
            return object.__getattribute__(self, name)

    candidate = HoldoutRun("search_val")
    with pytest.raises(HoldoutReadError, match="final_holdout"):
        decide("R1", [candidate], [HiddenBaseline()], _brief())
    assert candidate.reads == []


def test_mlflow_replay_ignores_holdout_metric_keys() -> None:
    class SpyMetrics(dict):
        def __iter__(self):
            raise AssertionError("pruner iterated metrics")

        def __getitem__(self, key):
            if "holdout" in str(key):
                raise AssertionError(key)
            return dict.__getitem__(self, key)

    metrics = SpyMetrics(
        {
            "ndcg@10_graded": 0.35,
            "delta_point": -0.05,
            "delta_ci_lo": -0.08,
            "delta_ci_hi": -0.04,
            "latency_p95_ms": 10.0,
            "dbu": 1.0,
            "select_holdout_ndcg": 0.99,
            "final_holdout_ndcg": 0.99,
        }
    )
    payload = {
        "info": {"run_id": "m1", "status": "FINISHED"},
        "data": {
            "metrics": metrics,
            "tags": {
                "node_id": "c1",
                "family": "trees",
                "rung": "R1",
                "seed": "0",
                "split": "search_val",
                "git_sha": "abc123",
                "job_run_id": "fake-1",
            },
        },
    }
    decision = decide("R1", [payload], [_baseline()], _brief())
    assert decision.decisions[0].rule == "below_floor"
    assert decision.decisions[0].status == "killed"

    failed = {
        "info": {"run_id": "m2", "status": "FAILED"},
        "data": {
            "metrics": [
                {"key": "select_holdout_ndcg", "value": _Boom()},
            ],
            "tags": [
                {"key": "node_id", "value": "c2"},
                {"key": "family", "value": "trees"},
                {"key": "rung", "value": "R1"},
                {"key": "seed", "value": "0"},
                {"key": "split", "value": "search_val"},
                {"key": "git_sha", "value": "abc123"},
                {"key": "job_run_id", "value": "fake-2"},
            ],
        },
    }
    failed_decision = decide("R1", [failed], [_baseline()], _brief())
    assert failed_decision.decisions[0].rule == "no_metrics"
    assert failed_decision.decisions[0].kill_reason == "run produced no metric"


class _Boom:
    def __float__(self) -> float:
        raise AssertionError("pruner read a holdout metric value")


def test_pruner_source_does_not_load_frames() -> None:
    source = (ROOT / "amt" / "pruner.py").read_text()
    assert "RankingFrame" not in source
    assert "load_dataset" not in source
    assert "select_holdout" in source
    assert "final_holdout" in source


def test_commit_writes_survivors_and_does_not_duplicate(tmp_path: Path) -> None:
    run = _run("c1", point=0.02, lo=0.0, hi=0.04)
    baseline = _baseline()
    brief_path = tmp_path / "brief.yaml"
    brief_path.write_text(
        yaml.safe_dump({"min_effect": 0.01, "budget": {"total_dbu": 100}})
    )
    runs_path = tmp_path / "runs.json"
    runs_path.write_text(json.dumps([json.loads(run.model_dump_json())]))
    baselines_path = tmp_path / "baselines.json"
    baselines_path.write_text(json.dumps([json.loads(baseline.model_dump_json())]))
    argv = [
        "--run-id",
        "search",
        "--rung",
        "R1",
        "--brief",
        str(brief_path),
        "--runs",
        str(runs_path),
        "--baselines",
        str(baselines_path),
        "--root",
        str(tmp_path),
    ]
    main(argv)
    main(argv)
    survivors = json.loads((tmp_path / "runs" / "search" / "rung_R1_survivors.json").read_text())
    assert survivors["survivors"] == ["c1"]
    rows = read_rows(tmp_path)
    assert len(rows) == 1
    assert rows[0].node_id == "c1"
    assert rows[0].status == "kept"
    assert rows[0].mlflow_run_id == "m-c1"
    assert rows[0].job_run_id == "fake-c1"
    assert rows[0].git_sha == "abc123"
    assert rows[0].delta_vs_lgbm is not None
    assert rows[0].delta_vs_lgbm.point == pytest.approx(0.02)


def test_commit_records_a_kill_reason(tmp_path: Path) -> None:
    run = _run("c1", point=-0.05, lo=-0.08, hi=-0.04)
    decision = decide("R1", [run], [_baseline()], _brief())
    commit_rung(tmp_path, "search", [run], decision)
    rows = read_rows(tmp_path)
    assert rows[0].status == "killed"
    assert rows[0].kill_reason
    assert "below floor" in rows[0].kill_reason


def test_family_slot_keeps_the_best_node_through_the_r1_floor() -> None:
    neural = _run(
        "n",
        point=-0.05,
        lo=-0.08,
        hi=-0.04,
        family="neural",
        metric=0.30,
        rung="R1",
    )
    weaker = _run(
        "n2",
        point=-0.06,
        lo=-0.09,
        hi=-0.05,
        family="neural",
        metric=0.22,
        rung="R1",
    )
    trees = _run("t", point=0.02, lo=0.0, hi=0.04, family="trees", metric=0.50, rung="R1")
    decision = decide(
        "R1",
        [neural, weaker, trees],
        [_baseline()],
        _brief(min_effect=0.01),
        random_metric=0.20,
    )
    by_id = _by_id(decision)
    assert by_id["n"].status == "kept"
    assert by_id["n"].rule == "family_slot"
    assert by_id["n2"].rule == "below_floor"
    assert by_id["t"].rule == "family_slot"
    assert "n" in decision.survivors
    assert "n2" not in decision.survivors


def test_family_slot_is_not_halved_at_r2() -> None:
    runs = [
        _run("a", point=0.05, lo=-0.05, hi=0.10, family="boost", metric=0.50, rung="R2"),
        _run("b", point=0.04, lo=-0.05, hi=0.10, family="boost", metric=0.45, rung="R2"),
        _run("c", point=0.03, lo=-0.05, hi=0.10, family="boost", metric=0.44, rung="R2"),
        _run("n", point=0.0, lo=-0.05, hi=0.10, family="neural", metric=0.30, rung="R2"),
    ]
    decision = decide("R2", runs, [_baseline("R2")], _brief(), random_metric=0.20)
    assert "n" in decision.survivors
    assert _by_id(decision)["n"].rule == "family_slot"
    assert _by_id(decision)["c"].rule == "halving"
    assert "a" in decision.survivors


def test_r3_promotes_c00_when_no_candidate_outranks_it() -> None:
    candidate = _run(
        "n1",
        point=0.02,
        lo=-0.01,
        hi=0.04,
        rung="R3",
        prod_point=0.03,
        prod_lo=0.01,
        prod_hi=0.05,
    )
    archive = _baseline("R3").model_copy(
        update={
            "delta_vs_production": DeltaVsBaseline(point=0.02, ci_lo=0.005, ci_hi=0.04),
            "latency_p95_ms": 10.0,
            "metric": 0.5,
        }
    )
    decision = decide("R3", [candidate], [archive], _brief(min_effect=0.01, latency=20))
    assert decision.promoted == ["c00"]
    assert decision.decisions[0].status == "kept"


def test_r3_candidate_that_beats_c00_is_promoted_ahead_of_the_archive() -> None:
    candidate = _run(
        "n1",
        point=0.03,
        lo=0.01,
        hi=0.05,
        rung="R3",
        prod_point=0.02,
        prod_lo=0.005,
        prod_hi=0.04,
    )
    archive = _baseline("R3").model_copy(
        update={
            "delta_vs_production": DeltaVsBaseline(point=0.04, ci_lo=0.01, ci_hi=0.06),
            "latency_p95_ms": 10.0,
        }
    )
    decision = decide("R3", [candidate], [archive], _brief(min_effect=0.01))
    assert decision.promoted == ["n1"]


def test_slow_candidate_fails_the_symmetric_gate_and_c00_is_promoted() -> None:
    candidate = _run(
        "slow",
        point=0.04,
        lo=0.02,
        hi=0.06,
        rung="R3",
        latency_p95_ms=50.0,
        prod_point=0.03,
        prod_lo=0.01,
        prod_hi=0.05,
    )
    archive = _baseline("R3").model_copy(
        update={
            "delta_vs_production": DeltaVsBaseline(point=0.02, ci_lo=0.005, ci_hi=0.04),
            "latency_p95_ms": 10.0,
        }
    )
    decision = decide(
        "R3",
        [candidate],
        [archive],
        _brief(min_effect=0.01, latency=20),
        shared_cluster=True,
    )
    assert decision.promoted == ["c00"]
    assert _by_id(decision)["slow"].status == "kept"
    assert _by_id(decision)["slow"].note == "latency_suspect"


def test_seed_c00_materializes_a_ranker_and_round0_metrics(tmp_path: Path) -> None:
    from amt.preflight import preflight_candidate
    from amt.schemas import RunState
    from amt.search_policy import seed_c00

    brief = _brief()
    brief.name = "nfy"
    rows = []
    for rung, point in (("R1", 0.01), ("R2", 0.02), ("R3", 0.03)):
        for seed in range(3):
            rows.append(
                {
                    "baseline_id": "lgbm_reference",
                    "rung": rung,
                    "seed": seed,
                    "metric": 0.40 + point,
                    "delta_point": point,
                    "delta_ci_lo": point - 0.005,
                    "delta_ci_hi": point + 0.005,
                    "latency_p95_ms": 12.0 + seed,
                }
            )
    directory = tmp_path / "runs" / "round0-nfy"
    directory.mkdir(parents=True)
    (directory / "round0.json").write_text(json.dumps(rows), encoding="utf-8")
    state = RunState.model_validate(
        {
            "run_id": "search",
            "brief_path": "briefs/nfy.yaml",
            "candidate_id": "pending",
            "family": "search",
            "rung": "R1",
            "seed": 0,
            "batch": "nfy",
            "git_sha": "abc123",
        }
    )
    seed_c00(tmp_path, state, brief)
    seed_c00(tmp_path, state, brief)
    assert [node.node_id for node in state.nodes if node.node_id == "c00"] == ["c00"]
    node = state.nodes[0]
    assert node.family == "lgbm_reference"
    assert node.rung_scores["R3"].delta_vs_production is not None
    assert node.rung_scores["R3"].delta_vs_production.point == pytest.approx(0.03)
    assert node.rung_scores["R1"].latency_p95_ms == pytest.approx(14.0)
    assert node.rung_scores["R2"].delta_vs_lgbm.point == pytest.approx(0.0)
    candidate = tmp_path / "runs" / "search" / "candidates" / "c00"
    assert (candidate / "spec.md").is_file()
    assert (candidate / "candidate.yaml").is_file()
    assert "VERDICT: PASS" in (candidate / "review.md").read_text(encoding="utf-8")
    smoke = preflight_candidate(candidate)
    assert smoke["deterministic"] == 1.0
    assert smoke["finite"] == 1.0


def test_pruner_job_passes_run_files() -> None:
    text = (ROOT / "resources" / "pruner.job.yml").read_text()
    assert "amt/pruner.py" in text
    assert "--runs" in text
    assert "--baselines" in text
    assert "--brief" in text
