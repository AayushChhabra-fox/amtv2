"""Ready-gate signature: resolved min_effect, repairs, and family hints."""

import json
from datetime import datetime, timezone
from pathlib import Path

import pytest
import yaml

from amt.cli import main
from amt.dataset_repair import record_attempt
from amt.gates import (
    SignoffError,
    problem_task_sha256,
    publish_ready_packet,
    ready_is_current,
    sign_ready,
)
from amt.ladder import build_ladder_report
from amt.noise import build_noise_report
from amt.orchestrator import run_brief
from amt.schemas import load_brief
from tests.test_ladder import _cells as ladder_cells
from tests.test_noise import _brief, _cells
from tests.test_search import _modality_manifest

ROOT = Path(__file__).resolve().parents[1]
SIGNED = datetime(2026, 11, 12, 15, 0, tzinfo=timezone.utc)


def _write_brief(root: Path, min_effect: float | str = 0.01) -> Path:
    path = root / "briefs" / "nfy_article_rerank.yaml"
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = _brief(min_effect).model_dump(mode="json")
    payload["dataset_spec"] = str(ROOT / "datasets" / "nfy_article_rerank")
    path.write_text(yaml.safe_dump(payload), encoding="utf-8")
    return path


def _packet(root: Path, brief_path: Path, *, width: float = 0.004):
    brief = load_brief(brief_path)
    noise = build_noise_report(
        _cells(width=width),
        brief,
        dataset="nfy_article_rerank",
        reference_scores=["production", "recency"],
    )
    ladder = build_ladder_report(ladder_cells(), dataset="nfy_article_rerank")
    manifest = root / "manifest.json"
    manifest.write_text(json.dumps(_modality_manifest()) + "\n", encoding="utf-8")
    return publish_ready_packet(root, brief, brief_path, noise, ladder, manifest), noise


def test_ready_locks_the_manifest_floor_and_budget(tmp_path: Path) -> None:
    brief_path = _write_brief(tmp_path)
    (report, _path), _noise = _packet(tmp_path, brief_path)
    assert report.status == "pass"
    assert report.viable_families == ["lgbm", "neural", "linear_reweight"]
    assert "embedding" in report.unsupported_families
    record = sign_ready(tmp_path, brief_path, "Ada Lovelace", signed_at=SIGNED)
    assert record.gate == "ready"
    assert record.signer == "Ada Lovelace"
    assert record.signed_at == "2026-11-12T15:00:00Z"
    assert record.resolved_min_effect == pytest.approx(0.01)
    assert record.total_dbu == pytest.approx(1500)
    assert record.primary_metric == "ndcg@10_graded"
    assert record.manifest_sha256
    assert record.spec_sha256
    assert record.problem_task_sha256
    assert (tmp_path / "gates" / "ready" / "nfy_article_rerank" / "ready.json").is_file()
    assert ready_is_current(tmp_path, brief_path)

    manifest = tmp_path / "gates" / "ready" / "nfy_article_rerank" / "manifest.json"
    manifest.write_text('{"modality": {"numeric_count": 1}}\n', encoding="utf-8")
    assert not ready_is_current(tmp_path, brief_path)
    with pytest.raises(SignoffError, match="human signer"):
        sign_ready(tmp_path, brief_path, "amt", signed_at=SIGNED)


def test_auto_floor_is_locked_without_editing_the_brief(tmp_path: Path) -> None:
    brief_path = _write_brief(tmp_path, "auto")
    (report, _path), noise = _packet(tmp_path, brief_path, width=0.006)
    assert noise.status == "pass"
    assert report.resolved_min_effect == pytest.approx(0.012)
    assert "auto" in brief_path.read_text(encoding="utf-8")
    sign_ready(tmp_path, brief_path, "Ada Lovelace", signed_at=SIGNED)
    assert ready_is_current(tmp_path, brief_path)

    before = problem_task_sha256(load_brief(brief_path))
    payload = yaml.safe_load(brief_path.read_text(encoding="utf-8"))
    payload["design_space"] = {"families": ["a hint the planner may ignore"]}
    brief_path.write_text(yaml.safe_dump(payload), encoding="utf-8")
    assert problem_task_sha256(load_brief(brief_path)) == before
    assert ready_is_current(tmp_path, brief_path)

    payload["min_effect"] = 0.05
    brief_path.write_text(yaml.safe_dump(payload), encoding="utf-8")
    assert problem_task_sha256(load_brief(brief_path)) == before
    assert not ready_is_current(tmp_path, brief_path)


def test_explicit_floor_below_the_noise_width_cannot_be_signed(tmp_path: Path) -> None:
    brief_path = _write_brief(tmp_path, 0.01)
    (report, _path), noise = _packet(tmp_path, brief_path, width=0.006)
    assert noise.status == "revise"
    assert report.status == "fail"
    with pytest.raises(SignoffError, match="not ready"):
        sign_ready(tmp_path, brief_path, "Ada Lovelace", signed_at=SIGNED)
    assert not ready_is_current(tmp_path, brief_path)


def test_an_edited_repair_diff_invalidates_the_signature(tmp_path: Path) -> None:
    brief_path = _write_brief(tmp_path)
    record_attempt(
        tmp_path,
        "nfy_article_rerank",
        1,
        {"features/signals.sql": "select 1 as score\n"},
        {"features/signals.sql": "select 2 as score\n"},
        {"kind": "contract"},
    )
    (report, _path), _noise = _packet(tmp_path, brief_path)
    assert report.repair_hashes
    sign_ready(tmp_path, brief_path, "Ada Lovelace", signed_at=SIGNED)
    assert ready_is_current(tmp_path, brief_path)
    patch = tmp_path / "runs" / "dataset-repair" / "nfy_article_rerank" / "1" / "diff.patch"
    patch.write_text(patch.read_text(encoding="utf-8") + "\n", encoding="utf-8")
    assert not ready_is_current(tmp_path, brief_path)


def test_search_stops_until_the_ready_gate_is_signed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("AMT_JOBS_BACKEND", "fake")
    brief_path = _write_brief(tmp_path)
    state = run_brief(brief_path, root=tmp_path)
    assert state.stage == "ESCALATE"
    assert state.submitted_jobs == {}
    text = (tmp_path / "runs" / state.run_id / "ESCALATE.md").read_text(encoding="utf-8")
    assert "ready gate" in text
    assert main(["gates", "--root", str(tmp_path), "--brief", str(brief_path)]) == 1


def test_signed_ready_gate_clears_a_search_brief(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    brief_path = _write_brief(tmp_path)
    _packet(tmp_path, brief_path)
    code = main(
        [
            "sign",
            "ready",
            "--root",
            str(tmp_path),
            "--brief",
            str(brief_path),
            "--signer",
            "Ada Lovelace",
        ]
    )
    assert code == 0
    assert "signed ready" in capsys.readouterr().out
    assert main(["gates", "--root", str(tmp_path), "--brief", str(brief_path)]) == 0
    assert "ready current" in capsys.readouterr().out


def test_walking_skeleton_does_not_require_the_ready_gate(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    path = tmp_path / "briefs" / "toy.yaml"
    path.parent.mkdir(parents=True)
    path.write_text(
        yaml.safe_dump(
            {
                "name": "toy",
                "run_id": "toy",
                "candidate": {"id": "c01", "family": "linear", "claim": "ridge ranker"},
                "budget": {"total_dbu": 10},
            }
        ),
        encoding="utf-8",
    )
    assert main(["gates", "--root", str(tmp_path), "--brief", str(path)]) == 0
    assert "not required" in capsys.readouterr().out
