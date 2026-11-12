"""Ladder validation for gate G2, including the eight designs."""

import time

import numpy as np
import pytest

from amt.ladder import KENDALL_MIN, build_ladder_report, kendall_tau_b, ladder_jobs, load_cells
from amt.noise import GateDataError
from harness.evaluate import graded_ndcg_at_k
from harness.interface import FitContext, RankingFrame
from harness.ladder import LADDER_DESIGNS, family_key, ladder_families, make_ladder_ranker
from harness.train import make_ranker

PAIRS = {
    "lgbm": ("lgbm_full", "lgbm_recency_category"),
    "neural": ("neural_normal", "neural_label_shuffled"),
    "embedding": ("embedding_trained", "embedding_random_projection"),
    "linear_reweight": ("linear_fitted", "linear_label_shuffled"),
}


def _metric(design_id: str, rung: str, table: dict[tuple[str, str], float]) -> float:
    if (design_id, rung) in table:
        return table[(design_id, rung)]
    if (design_id, "*") in table:
        return table[(design_id, "*")]
    index = [design.design_id for design in LADDER_DESIGNS].index(design_id)
    return float(80 - index)


def _cells(table: dict[tuple[str, str], float] | None = None) -> list:
    table = table or {}
    rows = []
    for design in LADDER_DESIGNS:
        for rung in ("R1", "R2", "R3"):
            metric = _metric(design.design_id, rung, table)
            for seed in (0, 1):
                rows.append(
                    {
                        "design_id": design.design_id,
                        "rung": rung,
                        "seed": seed,
                        "split": "search_val",
                        "metric": metric,
                    }
                )
    return load_cells(rows)


def test_kendall_tau_b_matches_a_hand_count() -> None:
    assert kendall_tau_b([1, 2, 3, 4], [1, 2, 4, 3]) == pytest.approx(4 / 6)
    assert kendall_tau_b([1, 2, 3], [3, 2, 1]) == pytest.approx(-1)
    assert kendall_tau_b([1, 1, 1], [1, 2, 3]) == pytest.approx(0)


def test_aligned_ladder_passes_and_the_jobs_stay_on_search_val() -> None:
    report = build_ladder_report(_cells(), dataset="nfy_article_rerank", r1_train_fraction=0.10)
    assert report.status == "pass"
    assert report.kendall_tau == pytest.approx(1)
    assert report.kendall_tau >= KENDALL_MIN
    assert report.family_set == ladder_families()
    assert report.family_key == family_key()
    assert report.expected_cells == 8 * 3 * 2
    assert all(order.holds for order in report.orders)
    jobs = ladder_jobs()
    assert len(jobs) == 48
    assert {job.split for job in jobs} == {"search_val"}
    assert {job.candidate_id for job in jobs} == {design.design_id for design in LADDER_DESIGNS}


def test_tau_just_above_and_below_the_bar() -> None:
    mild = {
        ("lgbm_full", "R1"): 4,
        ("lgbm_recency_category", "R1"): 2,
        ("linear_fitted", "R1"): 3,
        ("linear_label_shuffled", "R1"): 1,
        ("lgbm_full", "R3"): 4,
        ("lgbm_recency_category", "R3"): 3,
        ("linear_fitted", "R3"): 2,
        ("linear_label_shuffled", "R3"): 1,
        ("lgbm_full", "R2"): 4,
        ("lgbm_recency_category", "R2"): 3,
        ("linear_fitted", "R2"): 2,
        ("linear_label_shuffled", "R2"): 1,
    }
    passing = build_ladder_report(
        _cells(mild),
        dataset="nfy_article_rerank",
        skip_r1=["neural", "embedding"],
    )
    assert passing.kendall_n == 4
    assert passing.kendall_tau == pytest.approx(4 / 6)
    assert passing.status == "pass"

    flipped = {
        ("lgbm_full", "R1"): 2,
        ("lgbm_recency_category", "R1"): 1,
        ("linear_fitted", "R1"): 4,
        ("linear_label_shuffled", "R1"): 3,
        ("lgbm_full", "R2"): 4,
        ("lgbm_recency_category", "R2"): 3,
        ("linear_fitted", "R2"): 2,
        ("linear_label_shuffled", "R2"): 1,
        ("lgbm_full", "R3"): 4,
        ("lgbm_recency_category", "R3"): 3,
        ("linear_fitted", "R3"): 2,
        ("linear_label_shuffled", "R3"): 1,
    }
    report = build_ladder_report(
        _cells(flipped),
        dataset="nfy_article_rerank",
        skip_r1=["neural", "embedding"],
        r1_train_fraction=0.10,
    )
    assert report.kendall_tau == pytest.approx((2 - 4) / 6)
    assert report.kendall_tau < KENDALL_MIN
    assert report.status == "revise"
    text = " ".join(report.remediation)
    assert "below 0.6" in text
    assert "0.10" in text and "0.40" in text
    assert report.suggested_r1_train_fraction == pytest.approx(0.4)


def test_r1_inversion_is_revalidated_by_skipping_that_family() -> None:
    table = {("neural_normal", "R1"): 0.1, ("neural_label_shuffled", "R1"): 0.9}
    report = build_ladder_report(_cells(table), dataset="nfy_article_rerank")
    assert report.status == "revise"
    assert "Skip R1 for neural" in " ".join(report.remediation)
    again = build_ladder_report(
        _cells(table),
        dataset="nfy_article_rerank",
        skip_r1=["neural"],
    )
    assert again.status == "pass"
    neural_r1 = next(
        order for order in again.orders if order.family == "neural" and order.rung == "R1"
    )
    assert not neural_r1.checked
    held = [
        order for order in again.orders if order.family == "neural" and order.rung != "R1"
    ]
    assert all(order.holds for order in held)


def test_skipping_every_family_leaves_no_r1_ranking() -> None:
    report = build_ladder_report(
        _cells(),
        dataset="nfy_article_rerank",
        skip_r1=ladder_families(),
    )
    assert report.status == "fail"
    assert "no R1-versus-R3" in " ".join(report.remediation)


def test_high_rung_inversion_is_not_repaired_by_skipping_r1() -> None:
    table = {("lgbm_full", "R3"): 0.1, ("lgbm_recency_category", "R3"): 0.9}
    report = build_ladder_report(
        _cells(table),
        dataset="nfy_article_rerank",
        skip_r1=["lgbm"],
    )
    assert report.status == "fail"
    assert "do not separate" in " ".join(report.remediation)


def test_ladder_refuses_holdouts_and_incomplete_matrices() -> None:
    cells = _cells()
    held_back = [cell for cell in cells if not (cell.design_id == "lgbm_full" and cell.seed == 1)]
    partial = build_ladder_report(held_back, dataset="nfy_article_rerank")
    assert partial.status == "incomplete"
    assert partial.kendall_tau is None
    raw = cells[0].model_dump()
    raw["split"] = "final_holdout"
    with pytest.raises(GateDataError, match="search_val"):
        load_cells([raw])
    with pytest.raises(GateDataError, match="unknown family"):
        build_ladder_report(cells, dataset="nfy_article_rerank", skip_r1=["trees"])


def test_an_unsupported_family_is_listed_and_left_out() -> None:
    cells = [cell for cell in _cells() if not cell.design_id.startswith("embedding_")]
    reason = "embedding ladder needs feat__profile__emb* and feat__item__emb*"
    report = build_ladder_report(
        cells, dataset="nfy_article_rerank_demo", unsupported={"embedding": reason}
    )
    assert report.status == "pass"
    assert report.family_set == ["lgbm", "neural", "linear_reweight"]
    assert report.family_key == family_key()
    assert report.unsupported_families == {"embedding": reason}
    assert report.expected_cells == 6 * 3 * 2
    assert report.kendall_n == 6
    assert {order.family for order in report.orders} == {"lgbm", "neural", "linear_reweight"}
    from amt.ladder import render_ladder_markdown

    assert f"- embedding: {reason}" in render_ladder_markdown(report)
    with pytest.raises(GateDataError, match="unexpected"):
        build_ladder_report(_cells(), dataset="x", unsupported={"embedding": reason})
    with pytest.raises(GateDataError, match="unknown unsupported"):
        build_ladder_report(cells, dataset="x", unsupported={"trees": reason})
    with pytest.raises(GateDataError, match="unknown family"):
        build_ladder_report(
            cells, dataset="x", skip_r1=["embedding"], unsupported={"embedding": reason}
        )


def test_a_design_without_its_columns_reports_unsupported() -> None:
    from harness.ladder import UnsupportedDataset
    from harness.train import train_ranker

    rng = np.random.default_rng(0)
    rows = [
        {key: value for key, value in row.items() if "__emb" not in key}
        for row in _rows(rng, 0, 40)
    ]
    frame = RankingFrame(rows)
    with pytest.raises(UnsupportedDataset):
        make_ladder_ranker("embedding_trained").fit(
            frame, frame, FitContext(seed=0, deadline=time.monotonic() + 60)
        )
    result = train_ranker(
        ranker=make_ladder_ranker("embedding_random_projection"),
        train=frame,
        evaluation=frame,
        seed=0,
        rung="R1",
        deadline_s=60,
        require_loss=False,
        check_determinism=False,
    )
    assert result.status == "unsupported"
    assert "feat__profile__emb" in (result.reason or "")


def test_each_known_good_design_beats_its_pair() -> None:
    rng = np.random.default_rng(0)
    train = RankingFrame(_rows(rng, 0, 40))
    evaluation = RankingFrame(_rows(rng, 40, 64))
    context = FitContext(seed=0, deadline=time.monotonic() + 60)
    scores: dict[str, float] = {}
    for design in LADDER_DESIGNS:
        ranker = make_ladder_ranker(design.design_id)
        ranker.fit(train, evaluation, context)
        metric = graded_ndcg_at_k(evaluation, ranker.score(evaluation))
        scores[design.design_id] = metric
        assert make_ranker(design.design_id, None, None).__class__ is ranker.__class__
    for family, (good, bad) in PAIRS.items():
        assert scores[good] > scores[bad], (family, scores[good], scores[bad])


def _rows(rng: np.random.Generator, start: int, stop: int) -> list[dict]:
    rows = []
    user = (1.0, 0.0, 0.0)
    items = (
        ("good", 3, 1.0, (1.0, 0.0, 0.0)),
        ("mid", 1, 0.4, (0.2, 0.9, 0.0)),
        ("bad", 0, 0.0, (0.0, 1.0, 0.0)),
        ("bait", 0, 0.05, (0.0, 0.0, 1.0)),
    )
    for query in range(start, stop):
        day = f"2026-01-{(query % 5) + 1:02d}"
        for item_id, label, quality, embedding in items:
            rows.append(
                {
                    "query_id": f"q{query}",
                    "item_id": item_id,
                    "label": label,
                    "event_day": day,
                    "query_ts": f"{day}T12:00:00",
                    "feat__prod_signals__quality": quality,
                    "feat__prod_signals__recency": float(rng.random()),
                    "feat__article__category": float(rng.random()),
                    "feat__profile__emb0": user[0],
                    "feat__profile__emb1": user[1],
                    "feat__profile__emb2": user[2],
                    "feat__item__emb0": embedding[0],
                    "feat__item__emb1": embedding[1],
                    "feat__item__emb2": embedding[2],
                    "ref__production": float(rng.random()),
                }
            )
    return rows
