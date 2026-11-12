"""Ranker, metrics, latency, and baselines on both synthetic surfaces."""

from __future__ import annotations

import json
import math
import sqlite3
from pathlib import Path

import numpy as np
import pytest

from harness.baselines import iter_baselines
from harness.data.build_dataset import build
from harness.data.contract import load_spec
from harness.evaluate import (
    click_ndcg_at_k,
    coverage_at_k,
    graded_ndcg_at_k,
    mrr,
    paired_day_blocked_delta,
)
from harness.interface import FitContext, RankingFrame
from harness.latency import latency_p95_ms
from harness.train import _guard_split, frames_for_rung, load_dataset, train_ranker
from tests.test_dataset_builder import seed, write_spec

ROOT = Path(__file__).resolve().parents[1]
BANNED = ("nfy_", "news_impressions", "news_interactions", "for-you", "total_score")


def _frame(rows: list[dict]) -> RankingFrame:
    return RankingFrame(rows)


def _pair(
    query_id: str,
    day: str,
    label_a: float,
    scores: tuple[float, float],
) -> tuple[list[dict], np.ndarray]:
    rows = [
        {
            "query_id": query_id,
            "item_id": "a",
            "label": label_a,
            "event_day": day,
            "query_ts": f"{day} 12:00:00",
        },
        {
            "query_id": query_id,
            "item_id": "b",
            "label": 0,
            "event_day": day,
            "query_ts": f"{day} 12:00:00",
        },
    ]
    return rows, np.asarray(scores, dtype=float)


def test_graded_ndcg_mrr_coverage_and_bootstrap() -> None:
    perfect_rows, perfect_scores = _pair("q1", "2026-01-01", 3, (1.0, 0.0))
    perfect = _frame(perfect_rows)
    assert graded_ndcg_at_k(perfect, perfect_scores) == pytest.approx(1.0)
    assert mrr(perfect, perfect_scores) == pytest.approx(1.0)
    assert coverage_at_k(perfect, perfect_scores) == pytest.approx(1.0)

    reversed_scores = np.asarray([0.0, 1.0], dtype=float)
    assert graded_ndcg_at_k(perfect, reversed_scores) == pytest.approx(1.0 / math.log2(3))
    click_rows, click_scores = _pair("q1", "2026-01-01", 1, (1.0, 0.0))
    assert click_ndcg_at_k(_frame(click_rows), click_scores) == pytest.approx(1.0)

    day_one, scores_one = _pair("q1", "2026-01-01", 1, (1.0, 0.0))
    day_two, scores_two = _pair("q2", "2026-01-02", 1, (1.0, 0.0))
    baseline = np.asarray([0.0, 1.0, 0.0, 1.0], dtype=float)
    constant = _frame(day_one + day_two)
    same = paired_day_blocked_delta(
        constant,
        np.concatenate([scores_one, scores_two]),
        baseline,
        n_resamples=200,
        seed=0,
    )
    assert same["ci_lo"] == pytest.approx(same["point"])
    assert same["ci_hi"] == pytest.approx(same["point"])

    opposite_rows = day_one + day_two
    opposite_scores = np.asarray([1.0, 0.0, 0.0, 1.0], dtype=float)
    opposite_base = np.asarray([0.0, 1.0, 1.0, 0.0], dtype=float)
    spread = paired_day_blocked_delta(
        _frame(opposite_rows),
        opposite_scores,
        opposite_base,
        n_resamples=200,
        seed=1,
    )
    assert spread["ci_lo"] < spread["ci_hi"]


def test_r0_loss_deadline_and_holdout_guard() -> None:
    rows = []
    for query_id, day in (("q1", "2026-01-01"), ("q2", "2026-01-02")):
        pair, _scores = _pair(query_id, day, 1, (1.0, 0.0))
        rows.extend(pair)
    frame = _frame(rows)

    class FlatRanker:
        def fit(self, train, valid, ctx: FitContext) -> None:
            del train, valid
            ctx.log("train_loss", 1.0)
            ctx.log("train_loss", 1.0)

        def score(self, frame: RankingFrame) -> np.ndarray:
            return np.zeros(len(frame))

        def save(self, path: str) -> None:
            del path

    failed = train_ranker(
        ranker=FlatRanker(),
        train=frame,
        evaluation=frame,
        seed=0,
        rung="R0",
        deadline_s=30,
        require_loss=True,
        check_determinism=False,
    )
    assert failed.status != "ok"
    assert failed.metrics == {}
    assert "decreasing train_loss" in (failed.reason or "")

    expired = train_ranker(
        ranker=FlatRanker(),
        train=frame,
        evaluation=frame,
        seed=0,
        rung="R1",
        deadline_s=-1,
        require_loss=False,
        check_determinism=False,
    )
    assert expired.status != "ok"
    assert expired.metrics == {}

    with pytest.raises(SystemExit):
        _guard_split("select_holdout", False, False)
    with pytest.raises(SystemExit):
        _guard_split("final_holdout", True, False)
    context = FitContext(seed=0, deadline=10**12)
    with pytest.raises(RuntimeError):
        context.log("ndcg@10", 1.0)


def test_latency_is_finite() -> None:
    from harness.baselines.random import RandomRanker

    rows, _scores = _pair("q1", "2026-01-01", 1, (1.0, 0.0))
    ranker = RandomRanker()
    frame = _frame(rows)
    ranker.fit(frame, RankingFrame([]), FitContext(seed=0, deadline=10**12))
    elapsed = latency_p95_ms(ranker, frame, n_candidates=4, n_requests=5, seed=0)
    assert elapsed >= 0
    assert math.isfinite(elapsed)


def test_harness_has_no_surface_specific_code() -> None:
    for path in (ROOT / "harness").rglob("*.py"):
        text = path.read_text()
        for token in BANNED:
            assert token not in text, f"{path} contains {token}"


def test_baselines_every_rung_on_both_surfaces(tmp_path: Path) -> None:
    connection = sqlite3.connect(":memory:")
    seed(connection)
    for entity in ("article", "clip"):
        spec_dir = write_spec(tmp_path / entity, f"toy_{entity}", entity)
        output = tmp_path / f"{entity}-out"
        build(spec_dir, output_dir=output, backend="local", connection=connection)
        spec = load_spec(spec_dir)
        manifest, rows = load_dataset(output)
        for rung in ("R0", "R1", "R2", "R3"):
            train, evaluation = frames_for_rung(rows, manifest, rung)
            assert len(train) > 0
            assert len(evaluation) > 0
            holdout_ids = {
                row["query_id"] for row in rows["select_holdout"] + rows["final_holdout"]
            }
            assert holdout_ids.isdisjoint({row["query_id"] for row in train.rows})
            assert holdout_ids.isdisjoint({row["query_id"] for row in evaluation.rows})
            for ranker in iter_baselines(spec, n_estimators=5, min_child_samples=1):
                result = train_ranker(
                    ranker=ranker,
                    train=train,
                    evaluation=evaluation,
                    seed=0,
                    rung=rung,
                    deadline_s=60,
                    require_loss=False,
                    check_determinism=rung == "R0" and type(ranker).__name__ == "RandomRanker",
                )
                assert result.status == "ok", (entity, rung, type(ranker).__name__, result.reason)
                assert "ndcg@10_graded" in result.metrics
                assert math.isfinite(result.metrics["ndcg@10_graded"])


def test_pruner_cli_reports_missing_metric(capsys) -> None:
    from amt.pruner import main

    main(["--run-id", "r1", "--rung", "R0", "--metric", ""])
    payload = json.loads(capsys.readouterr().out)
    assert payload["status"] == "failed"
    assert payload["kill_reason"] == "run produced no metric"
