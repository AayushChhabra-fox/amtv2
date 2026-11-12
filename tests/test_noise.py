"""Round-0 noise report for gate G1."""

import json
from pathlib import Path

import pytest
import yaml

from amt.cli import main
from amt.noise import (
    PROPOSED_MIN_EFFECT,
    GateDataError,
    baseline_ids,
    build_noise_report,
    expected_cells,
    load_cells,
    round0_jobs,
)
from amt.schemas import Brief
from harness.data.contract import load_spec

ROOT = Path(__file__).resolve().parents[1]
NAMES = ["production", "recency", "popularity", "random", "lgbm_reference"]
RUNGS = ["R0", "R1", "R2", "R3"]


def _brief(min_effect: float = 0.01) -> Brief:
    return Brief.model_validate(
        {
            "name": "nfy_article_rerank",
            "dataset_spec": "datasets/nfy_article_rerank",
            "min_effect": min_effect,
            "budget": {"total_dbu": 1500, "max_concurrent_jobs": 4},
            "problem": {
                "metric": {
                    "name": "ndcg@10_graded",
                    "gains": {"impressed": 0, "click": 1, "engaged_read": 2, "deep_read": 3},
                }
            },
            "efficiency": {"train_dbu_per_candidate": 60},
        }
    )


def _cells(
    *,
    width: float = 0.004,
    lgbm_delta: float = 0.02,
    recency_r3: tuple[float, float, float] | None = None,
    production_width: float | None = None,
) -> list:
    rows = []
    for name in NAMES:
        for rung in RUNGS:
            for seed in (0, 1, 2):
                point = 0.0
                if name == "lgbm_reference":
                    point = lgbm_delta
                if recency_r3 is not None and name == "recency" and rung == "R3":
                    point = recency_r3[seed]
                row = {
                    "baseline_id": name,
                    "rung": rung,
                    "seed": seed,
                    "split": "search_val",
                    "metric": 0.40 + point,
                }
                if name != "production":
                    row["delta_point"] = point
                    row["delta_ci_lo"] = point - width
                    row["delta_ci_hi"] = point + width
                elif production_width is not None:
                    row["delta_point"] = 0.0
                    row["delta_ci_lo"] = -production_width
                    row["delta_ci_hi"] = production_width
                rows.append(row)
    return load_cells(rows)


def test_round0_plan_covers_every_baseline_and_stays_on_search_val() -> None:
    for folder in ("nfy_article_rerank", "nfy_clip_rerank"):
        spec = load_spec(ROOT / "datasets" / folder)
        jobs = round0_jobs(spec.reference_scores)
        assert len(jobs) == 5 * 4 * 3
        assert {job.candidate_id for job in jobs} == set(NAMES)
        assert {job.split for job in jobs} == {"search_val"}
        assert all(job.params()["split"] == "search_val" for job in jobs)
        recency = next(job for job in jobs if job.candidate_id == "recency")
        assert recency.baseline == "reference_score"
        assert recency.reference_score == "recency"
        assert recency.params()["reference_score"] == "recency"
    assert baseline_ids(["recency", "production"])[0] == "production"
    with pytest.raises(GateDataError):
        baseline_ids(["recency"])


def test_half_width_sets_min_effect_and_records_either_lgbm_result() -> None:
    passing = build_noise_report(
        _cells(width=0.004, lgbm_delta=0.02),
        _brief(),
        dataset="nfy_article_rerank",
        reference_scores=["production", "recency"],
    )
    assert passing.status == "pass"
    assert passing.r3_half_width == pytest.approx(0.004)
    assert passing.required_min_effect == pytest.approx(0.008)
    assert passing.brief_min_effect == pytest.approx(PROPOSED_MIN_EFFECT)
    assert passing.half_widths["R0"] == pytest.approx(0.004)
    assert passing.lgbm_vs_production is not None
    assert passing.lgbm_vs_production.beats_production
    assert passing.evaluation.startswith("ndcg@10_graded")
    assert passing.budget_total_dbu == 1500
    assert passing.remediation == []

    losing = build_noise_report(
        _cells(width=0.004, lgbm_delta=-0.01),
        _brief(),
        dataset="nfy_article_rerank",
        reference_scores=["production", "recency"],
    )
    assert losing.status == "pass"
    assert losing.lgbm_vs_production is not None
    assert not losing.lgbm_vs_production.beats_production


def test_min_effect_below_twice_the_r3_width_asks_for_a_revision() -> None:
    report = build_noise_report(
        _cells(width=0.006),
        _brief(),
        dataset="nfy_article_rerank",
        reference_scores=["production", "recency"],
    )
    assert report.status == "revise"
    assert report.required_min_effect == pytest.approx(0.012)
    text = " ".join(report.remediation)
    assert "2×" in text
    assert "lengthen search_val" in text
    assert "0.010" in text


def test_seed_scatter_widens_the_floor_past_the_bootstrap() -> None:
    report = build_noise_report(
        _cells(width=0.001, recency_r3=(0.0, 0.0, 0.06)),
        _brief(),
        dataset="nfy_article_rerank",
        reference_scores=["production", "recency"],
    )
    assert report.r3_half_width == pytest.approx(0.086053)
    assert report.required_min_effect == pytest.approx(0.172106)
    assert report.status == "revise"
    recency = next(
        row for row in report.baselines if row.baseline_id == "recency" and row.rung == "R3"
    )
    assert recency.seed_half_width > recency.bootstrap_half_width


def test_production_interval_does_not_set_the_floor() -> None:
    report = build_noise_report(
        _cells(width=0.004, production_width=0.5),
        _brief(),
        dataset="nfy_article_rerank",
        reference_scores=["production", "recency"],
    )
    assert report.r3_half_width == pytest.approx(0.004)
    assert all(row.baseline_id != "production" for row in report.baselines)


def test_incomplete_holdout_and_duplicate_cells() -> None:
    cells = _cells()
    held = [cell for cell in cells if not (cell.baseline_id == "random" and cell.seed == 2)]
    partial = build_noise_report(
        held,
        _brief(),
        dataset="nfy_article_rerank",
        reference_scores=["production", "recency"],
    )
    assert partial.status == "incomplete"
    assert partial.missing == ["random/R0/2", "random/R1/2", "random/R2/2", "random/R3/2"]
    assert len(expected_cells(["production", "recency"])) == 60

    raw = cells[0].model_dump()
    raw["split"] = "select_holdout"
    with pytest.raises(GateDataError, match="search_val"):
        load_cells([raw])
    raw["split"] = "final_holdout"
    with pytest.raises(GateDataError, match="search_val"):
        load_cells([raw])
    with pytest.raises(GateDataError, match="duplicate"):
        load_cells([cells[0].model_dump(), cells[0].model_dump()])
    with pytest.raises(GateDataError, match="paired delta"):
        load_cells(
            [
                {
                    "baseline_id": "random",
                    "rung": "R1",
                    "seed": 0,
                    "metric": 0.2,
                }
            ]
        )


def test_noise_cli_writes_a_passing_report(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    brief_path = tmp_path / "brief.yaml"
    brief_path.write_text(
        yaml.safe_dump(
            {
                "name": "nfy_article_rerank",
                "dataset_spec": str(ROOT / "datasets" / "nfy_article_rerank"),
                "min_effect": 0.01,
                "budget": {"total_dbu": 1500},
                "problem": {"metric": {"name": "ndcg@10_graded"}},
            }
        )
    )
    observations = tmp_path / "obs.json"
    observations.write_text(json.dumps([cell.model_dump() for cell in _cells()]))
    code = main(
        [
            "noise",
            "--root",
            str(tmp_path),
            "--brief",
            str(brief_path),
            "--observations",
            str(observations),
        ]
    )
    assert code == 0
    captured = capsys.readouterr().out
    assert "pass" in captured
    assert (tmp_path / "gates" / "g1" / "nfy_article_rerank" / "noise_report.md").is_file()
