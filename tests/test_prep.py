"""amt prep: doctor, deploy, preflight, build, and parallel round 0."""

import json
import threading
from pathlib import Path

import pytest
import yaml

from amt.cli import main
from amt.dataset_repair import record_attempt
from amt.gates import ReadyReport
from amt.jobs_backend import FakeJobs
from amt.ladder import build_ladder_report
from amt.modality import summarize_rows
from amt.noise import build_noise_report
from amt.prep import PrepError, PrepHooks, _deploy_if_stale, run_prep
from amt.schemas import load_brief
from tests.test_ladder import _cells as ladder_cells
from tests.test_noise import _brief, _cells
from tests.test_search import _modality_manifest

ROOT = Path(__file__).resolve().parents[1]


def _write_brief(root: Path) -> Path:
    path = root / "briefs" / "nfy_article_rerank.yaml"
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = _brief().model_dump(mode="json")
    payload["dataset_spec"] = str(ROOT / "datasets" / "nfy_article_rerank")
    path.write_text(yaml.safe_dump(payload), encoding="utf-8")
    return path


def _passing_collectors():
    def collect_noise(root, brief, brief_path, dataset, reference_scores):
        report = build_noise_report(
            _cells(),
            brief,
            dataset=dataset,
            reference_scores=reference_scores,
        )
        from amt.gates import write_noise_report

        path = write_noise_report(root, brief, brief_path, report)
        return report.status, path

    def collect_ladder(root, dataset, *, skip_r1, r1_train_fraction, report_name=None):
        report = build_ladder_report(
            ladder_cells(),
            dataset=dataset,
            skip_r1=list(skip_r1),
            r1_train_fraction=r1_train_fraction,
        )
        from amt.gates import write_ladder_report

        path = write_ladder_report(root, report, name=report_name)
        return report.status, path

    return collect_noise, collect_ladder


def test_prep_runs_the_pipeline_and_submits_round0_together(tmp_path: Path) -> None:
    brief_path = _write_brief(tmp_path)
    phases: list[str] = []
    started = {"n": 0}
    release = threading.Event()
    lock = threading.Lock()
    collect_noise, collect_ladder = _passing_collectors()

    def _mark(name: str) -> None:
        with lock:
            phases.append(name)

    def submit_noise(*_args, **_kwargs):
        with lock:
            started["n"] += 1
            phases.append("submit-noise")
            if started["n"] == 2:
                release.set()
        assert release.wait(2)
        _mark("submit-noise-done")

    def submit_ladder(*_args, **_kwargs):
        with lock:
            started["n"] += 1
            phases.append("submit-ladder")
            if started["n"] == 2:
                release.set()
        assert release.wait(2)
        _mark("submit-ladder-done")

    def fetch(root, brief, brief_path, _spec):
        _mark("fetch")
        from amt.gates import brief_key, ready_dir

        path = ready_dir(root, brief_key(brief, brief_path)) / "manifest.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(_modality_manifest()) + "\n", encoding="utf-8")
        return path

    report, path = run_prep(
        tmp_path,
        brief_path,
        hooks=PrepHooks(
            doctor=lambda _root: phases.append("doctor") or [],
            deploy_if_stale=lambda _root: phases.append("deploy") or "deployed",
            preflight=lambda _root, _spec, _brief: phases.append("preflight") or {"status": "pass"},
            build=lambda _root, _spec: phases.append("build") or "build-1",
            fetch_manifest=fetch,
            submit_noise=submit_noise,
            submit_ladder=submit_ladder,
            collect_noise=lambda *args, **kwargs: phases.append("collect-noise")
            or collect_noise(*args, **kwargs),
            collect_ladder=lambda *args, **kwargs: phases.append("collect-ladder")
            or collect_ladder(*args, **kwargs),
        ),
    )
    assert phases[:5] == ["doctor", "deploy", "preflight", "build", "fetch"]
    assert set(phases[5:7]) == {"submit-noise", "submit-ladder"}
    assert "collect-noise" in phases and "collect-ladder" in phases
    assert report.status == "pass", report.remediation
    assert report.resolved_min_effect == pytest.approx(0.01)
    assert path.is_file()
    assert (path.parent / "ready_report.md").is_file()
    loaded = load_brief(brief_path)
    assert loaded.min_effect == 0.01


def test_prep_stops_before_the_build_when_preflight_fails(tmp_path: Path) -> None:
    brief_path = _write_brief(tmp_path)

    def build(_root, _spec):
        raise AssertionError("build should not run")

    with pytest.raises(PrepError, match="search_val"):
        run_prep(
            tmp_path,
            brief_path,
            hooks=PrepHooks(
                doctor=lambda _root: [],
                deploy_if_stale=lambda _root: "fresh",
                preflight=lambda _root, _spec, _brief: {
                    "status": "fail",
                    "failure": {"message": "search_val spans 1 calendar day"},
                },
                build=build,
            ),
        )


def test_prep_repair_records_the_feature_diff_and_skips_plain_preflight(tmp_path: Path) -> None:
    brief_path = _write_brief(tmp_path)
    called: list[str] = []
    collect_noise, collect_ladder = _passing_collectors()

    def repair(root, _spec, _brief):
        called.append("repair")
        record_attempt(
            root,
            "nfy_article_rerank",
            1,
            {"features/signals.sql": "select 1 as score\n"},
            {"features/signals.sql": "select cast(1 as double) as score\n"},
            {"kind": "contract"},
        )
        return {"status": "pass", "fixes": 1}

    def fetch(root, brief, brief_path, _spec):
        from amt.gates import brief_key, ready_dir

        path = ready_dir(root, brief_key(brief, brief_path)) / "manifest.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(_modality_manifest()) + "\n", encoding="utf-8")
        return path

    report, _path = run_prep(
        tmp_path,
        brief_path,
        repair=True,
        hooks=PrepHooks(
            doctor=lambda _root: [],
            deploy_if_stale=lambda _root: "fresh",
            preflight=lambda *_args, **_kwargs: called.append("preflight") or {"status": "pass"},
            repair=repair,
            build=lambda _root, _spec: "build-1",
            fetch_manifest=fetch,
            submit_noise=lambda *_args, **_kwargs: None,
            submit_ladder=lambda *_args, **_kwargs: None,
            collect_noise=collect_noise,
            collect_ladder=collect_ladder,
        ),
    )
    assert called == ["repair"]
    assert report.status == "pass"
    assert len(report.repairs) == 1
    assert report.repair_hashes[0] == report.repairs[0].diff_sha256


def test_fake_deploy_is_recorded_once(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("AMT_JOBS_BACKEND", "fake")
    assert _deploy_if_stale(tmp_path) == "recorded"
    assert (tmp_path / ".amt" / "deployed.json").is_file()
    assert _deploy_if_stale(tmp_path) == "fresh"


def test_prep_cli_forwards_repair(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    seen: dict[str, bool] = {}

    def fake(_root, _brief, *, repair=False, hooks=None):
        del hooks
        seen["repair"] = repair
        report = ReadyReport(
            brief_name="nfy",
            dataset="nfy",
            status="pass",
            proposed_min_effect=0.01,
            budget_total_dbu=1,
            noise_status="pass",
            ladder_status="pass",
            noise_sha256="a" * 64,
            ladder_sha256="b" * 64,
            manifest_sha256="c" * 64,
            spec_path="datasets/nfy",
            spec_sha256="d" * 64,
            problem_task_sha256="e" * 64,
        )
        path = tmp_path / "ready_report.json"
        path.write_text("{}\n", encoding="utf-8")
        return report, path

    monkeypatch.setattr("amt.cli.run_prep", fake)
    brief = tmp_path / "brief.yaml"
    brief.write_text("name: nfy\n", encoding="utf-8")
    code = main(["prep", "--root", str(tmp_path), "--brief", str(brief), "--repair"])
    assert code == 0
    assert seen["repair"] is True


def test_summarize_rows_separates_modalities() -> None:
    rows = {
        "search_val": [
            {
                "query_id": "q1",
                "item_id": "a",
                "feat__prod_signals__total_score": 0.2,
                "feat__article__recency": 3.0,
                "feat__cat__section": "news",
                "feat__article__title": "hello",
                "feat__profile__emb0": 1.0,
                "feat__item__emb0": 1.0,
            },
            {
                "query_id": "q1",
                "item_id": "b",
                "feat__prod_signals__total_score": 0.1,
                "feat__article__recency": 1.0,
                "feat__cat__section": "sports",
                "feat__article__title": "there",
                "feat__profile__emb0": 0.0,
                "feat__item__emb0": 0.0,
            },
        ]
    }
    profile = summarize_rows(rows)
    assert profile["numeric_count"] == 2
    assert profile["categorical_columns"] == ["feat__cat__section"]
    assert profile["text_columns"] == ["feat__article__title"]
    assert profile["embedding_groups"] == ["item", "profile"]
    assert profile["prod_signal_columns"] == ["feat__prod_signals__total_score"]
    assert profile["row_count"] == 2
    assert profile["query_count"] == 1


def test_prep_on_the_fake_backend_submits_build_and_round0(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("AMT_JOBS_BACKEND", "fake")
    monkeypatch.setenv("AMT_GIT_SHA", "abc123")
    brief_path = _write_brief(tmp_path)
    staged = tmp_path / ".amt" / "fake_manifest.json"
    staged.parent.mkdir(parents=True)
    staged.write_text(json.dumps(_modality_manifest()) + "\n", encoding="utf-8")
    report, path = run_prep(tmp_path, brief_path)
    assert path.is_file()
    assert report.status != "pass"
    jobs = FakeJobs(tmp_path).submitted_jobs()
    assert jobs.count("build_dataset") >= 1
    assert jobs.count("train") >= 2
