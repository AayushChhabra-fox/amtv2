"""G1 and G2 signatures, reuse, and the search block."""

import json
from datetime import datetime, timezone
from pathlib import Path

import pytest
import yaml

from amt.cli import main
from amt.ladder import build_ladder_report
from amt.ladder import load_cells as load_ladder_cells
from amt.noise import build_noise_report
from amt.orchestrator import run_brief
from amt.schemas import load_brief
from amt.signoff import (
    SignoffError,
    g1_is_current,
    g2_is_current,
    sign_g1,
    sign_g2,
    write_ladder_report,
    write_noise_report,
)
from harness.ladder import family_key
from tests.test_ladder import _cells as ladder_cells
from tests.test_noise import _brief, _cells

ROOT = Path(__file__).resolve().parents[1]
SIGNED = datetime(2026, 11, 12, 15, 0, tzinfo=timezone.utc)


def _write_brief(root: Path, min_effect: float = 0.01) -> Path:
    path = root / "briefs" / "nfy_article_rerank.yaml"
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = _brief(min_effect).model_dump(mode="json")
    payload["dataset_spec"] = str(ROOT / "datasets" / "nfy_article_rerank")
    path.write_text(yaml.safe_dump(payload))
    return path


def _manifest(root: Path, text: str = '{"rows": 1}\n') -> Path:
    path = root / "manifest.json"
    path.write_text(text)
    return path


def _passing_noise(root: Path, brief_path: Path) -> None:
    brief = load_brief(brief_path)
    report = build_noise_report(
        _cells(),
        brief,
        dataset="nfy_article_rerank",
        reference_scores=["production", "recency"],
    )
    assert report.status == "pass"
    write_noise_report(root, brief, brief_path, report)


def test_g1_records_the_brief_manifest_min_effect_and_budget(tmp_path: Path) -> None:
    brief_path = _write_brief(tmp_path)
    manifest = _manifest(tmp_path)
    _passing_noise(tmp_path, brief_path)
    record = sign_g1(tmp_path, brief_path, "Ada Lovelace", manifest, signed_at=SIGNED)
    assert record.gate == "G1"
    assert record.signer == "Ada Lovelace"
    assert record.signed_at == "2026-11-12T15:00:00Z"
    assert record.min_effect == pytest.approx(0.01)
    assert record.total_dbu == pytest.approx(1500)
    assert record.primary_metric == "ndcg@10_graded"
    assert record.evaluation is not None and "ndcg@10_graded" in record.evaluation
    assert record.manifest_sha256
    assert record.spec_sha256
    assert record.brief_sha256
    assert g1_is_current(tmp_path, brief_path)

    manifest.write_text('{"rows": 2}\n')
    assert not g1_is_current(tmp_path, brief_path)
    with pytest.raises(SignoffError, match="human signer"):
        sign_g1(tmp_path, brief_path, "amt", manifest, signed_at=SIGNED)


def test_g1_refuses_a_report_that_misses_the_floor(tmp_path: Path) -> None:
    brief_path = _write_brief(tmp_path)
    brief = load_brief(brief_path)
    report = build_noise_report(
        _cells(width=0.006),
        brief,
        dataset="nfy_article_rerank",
        reference_scores=["production", "recency"],
    )
    write_noise_report(tmp_path, brief, brief_path, report)
    with pytest.raises(SignoffError, match="not ready"):
        sign_g1(tmp_path, brief_path, "Ada Lovelace", _manifest(tmp_path), signed_at=SIGNED)
    assert not g1_is_current(tmp_path, brief_path)


def test_g2_is_reused_for_the_dataset_and_family_set(tmp_path: Path) -> None:
    report = build_ladder_report(ladder_cells(), dataset="nfy_article_rerank")
    write_ladder_report(tmp_path, report)
    record = sign_g2(tmp_path, "nfy_article_rerank", "Grace Hopper", signed_at=SIGNED)
    assert record.family_set == [
        "lgbm",
        "neural",
        "embedding",
        "linear_reweight",
    ]
    assert record.family_key == family_key()
    assert record.kendall_tau == pytest.approx(1)
    assert g2_is_current(tmp_path, "nfy_article_rerank")
    assert not g2_is_current(tmp_path, "nfy_clip_rerank")

    stored = tmp_path / "gates" / "g2" / "nfy_article_rerank" / family_key() / "g2.json"
    edited = json.loads(stored.read_text())
    edited["family_set"] = ["other"]
    stored.write_text(json.dumps(edited))
    assert not g2_is_current(tmp_path, "nfy_article_rerank")


def test_search_brief_stops_until_both_gates_are_signed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("AMT_JOBS_BACKEND", "fake")
    brief_path = _write_brief(tmp_path)
    state = run_brief(brief_path, root=tmp_path)
    assert state.stage == "ESCALATE"
    assert state.submitted_jobs == {}
    text = (tmp_path / "runs" / state.run_id / "ESCALATE.md").read_text()
    assert "G1" in text

    code = main(["gates", "--root", str(tmp_path), "--brief", str(brief_path)])
    assert code == 1


def test_signed_gates_clear_a_search_brief(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    brief_path = _write_brief(tmp_path)
    manifest = _manifest(tmp_path)
    _passing_noise(tmp_path, brief_path)
    sign_g1(tmp_path, brief_path, "Ada Lovelace", manifest, signed_at=SIGNED)
    ladder = build_ladder_report(ladder_cells(), dataset="nfy_article_rerank")
    write_ladder_report(tmp_path, ladder)
    sign_g2(tmp_path, "nfy_article_rerank", "Grace Hopper", signed_at=SIGNED)
    code = main(["gates", "--root", str(tmp_path), "--brief", str(brief_path)])
    assert code == 0
    assert "G1 current" in capsys.readouterr().out


def test_walking_skeleton_does_not_require_the_gates(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    path = tmp_path / "briefs" / "toy.yaml"
    path.parent.mkdir(parents=True)
    path.write_text(
        yaml.safe_dump(
            {
                "name": "toy",
                "run_id": "toy",
                "candidate": {
                    "id": "c00",
                    "family": "linear",
                    "claim": "logistic regression",
                },
                "budget": {"total_dbu": 10},
            }
        )
    )
    assert main(["gates", "--root", str(tmp_path), "--brief", str(path)]) == 0
    assert "not required" in capsys.readouterr().out


def test_ladder_cli_signs_only_a_passing_report(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    observations = tmp_path / "ladder.json"
    observations.write_text(
        json.dumps([cell.model_dump() for cell in ladder_cells()])
    )
    code = main(
        [
            "ladder",
            "--root",
            str(tmp_path),
            "--dataset",
            "nfy_article_rerank",
            "--observations",
            str(observations),
            "--r1-train-fraction",
            "0.10",
        ]
    )
    assert code == 0
    capsys.readouterr()
    signed = main(
        [
            "sign",
            "g2",
            "--root",
            str(tmp_path),
            "--dataset",
            "nfy_article_rerank",
            "--signer",
            "Grace Hopper",
        ]
    )
    assert signed == 0
    assert g2_is_current(tmp_path, "nfy_article_rerank")

    weak = load_ladder_cells(json.loads(observations.read_text()))
    for cell in weak:
        if cell.design_id == "lgbm_full" and cell.rung == "R3":
            cell.metric = 0.0
    report = build_ladder_report(weak, dataset="nfy_clip_rerank")
    write_ladder_report(tmp_path, report)
    with pytest.raises(SignoffError):
        sign_g2(tmp_path, "nfy_clip_rerank", "Grace Hopper", signed_at=SIGNED)
