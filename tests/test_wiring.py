"""Job routing, snapshots, baselines, and the automated gate commands."""

import json
import os
from pathlib import Path

import pytest
import yaml

from amt.config import INTERNAL_ENV, effective_concurrency, load_settings
from amt.deploy import DeployError, assert_fresh, tree_hash, write_deployed
from amt.doctor import report as doctor_report
from amt.final import FinalError, open_final
from amt.jobs_backend import (
    JOB_NAMES,
    FakeJobs,
    declared_parameters,
    existing_cluster_payload,
    resolve_job_ids,
)
from amt.pruner import decide
from amt.schemas import Brief, load_brief
from amt.snapshot import SnapshotError, upload_candidate
from amt.submit import lgbm_baseline_view, missing_baseline_message
from harness.interface import RankingFrame
from harness.scores_io import align_scores, read_scores, write_scores
from tests.test_pruner import _baseline, _brief, _run

ROOT = Path(__file__).resolve().parents[1]


def test_resolve_job_ids_uses_the_cache_and_env_override(tmp_path: Path, monkeypatch) -> None:
    cached = {name: f"id-{name}" for name in JOB_NAMES}
    path = tmp_path / ".amt" / "job_ids.json"
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps(cached), encoding="utf-8")
    monkeypatch.setenv("AMT_TRAIN_JOB_ID", "override-train")
    found = resolve_job_ids(tmp_path)
    assert found["train"] == "override-train"
    assert found["confirm"] == "id-confirm"
    assert found["hello_train"] == "id-hello_train"


def test_fake_submit_rejects_undeclared_parameters(tmp_path: Path) -> None:
    backend = FakeJobs(tmp_path)
    with pytest.raises(ValueError, match="does not declare"):
        backend.submit(
            job="train",
            job_key="bad",
            params={"not_a_parameter": "1"},
            idempotency_token="bad",
        )
    assert "candidate_id" in declared_parameters("train")
    assert "confirm_token" in declared_parameters("confirm")
    assert "split" not in declared_parameters("confirm")


def test_snapshot_layout_and_refusal_to_overwrite(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("AMT_JOBS_BACKEND", "fake")
    model = tmp_path / "candidates" / "c00" / "model.py"
    model.parent.mkdir(parents=True)
    model.write_text("class Ranker: pass\n", encoding="utf-8")
    first = upload_candidate(tmp_path, "run", "c00", "abc")
    assert first.endswith("candidates/run/c00/abc/model.py")
    model.write_text("changed\n", encoding="utf-8")
    second = upload_candidate(tmp_path, "run", "c00", "abc")
    assert Path(second).read_text(encoding="utf-8") == "class Ranker: pass\n"
    with pytest.raises(SnapshotError):
        upload_candidate(tmp_path, "run", "missing", "abc")


def test_deploy_freshness(tmp_path: Path) -> None:
    (tmp_path / "harness").mkdir()
    (tmp_path / "harness" / "note.txt").write_text("v1", encoding="utf-8")
    write_deployed(tmp_path, git_sha="abc")
    assert_fresh(tmp_path)
    (tmp_path / "harness" / "note.txt").write_text("v2", encoding="utf-8")
    with pytest.raises(DeployError, match="redeploy"):
        assert_fresh(tmp_path)
    assert tree_hash(tmp_path) != json.loads(
        (tmp_path / ".amt" / "deployed.json").read_text(encoding="utf-8")
    )["harness_tree_hash"]


def test_keyed_scores_fail_closed_on_partial_coverage(tmp_path: Path) -> None:
    frame = RankingFrame(
        [
            {"query_id": "q1", "item_id": "a", "label": 1, "ref__production": 0.2},
            {"query_id": "q1", "item_id": "b", "label": 0, "ref__production": 0.1},
        ]
    )
    path = write_scores(tmp_path / "seed=0.json", frame, [0.2, 0.1])
    assert len(read_scores(path)) == 2
    aligned = align_scores(frame, read_scores(path))
    assert list(aligned) == [0.2, 0.1]
    with pytest.raises(RuntimeError, match="cover"):
        align_scores(frame, [{"query_id": "q1", "item_id": "a", "score": 1.0}])


def test_round0_file_supplies_the_pruner_baseline(tmp_path: Path) -> None:
    brief = Brief.model_validate(
        {
            "name": "nfy",
            "dataset_spec": "datasets/nfy",
            "budget": {"total_dbu": 1},
        }
    )
    assert missing_baseline_message(tmp_path, brief, "R1") == "run amt noise --submit first"
    directory = tmp_path / "runs" / "round0-nfy"
    directory.mkdir(parents=True)
    rows = [
        {"baseline_id": "lgbm_reference", "rung": "R1", "metric": 0.33},
        {"baseline_id": "lgbm_reference", "rung": "R1", "metric": 0.35},
    ]
    (directory / "round0.json").write_text(json.dumps(rows), encoding="utf-8")
    view = lgbm_baseline_view(tmp_path, brief, "R1", "abc")
    assert view.metric == pytest.approx(0.34)
    assert view.family == "lgbm_reference"


def test_noise_submit_is_idempotent(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("AMT_JOBS_BACKEND", "fake")
    from amt.batch import submit_noise

    brief = load_brief(ROOT / "briefs" / "toy.yaml")
    spec = str(ROOT / "datasets" / "nfy_article_rerank")
    from harness.data.contract import load_spec

    reference = list(load_spec(spec).reference_scores)
    submit_noise(tmp_path, brief, spec, reference)
    first = FakeJobs(tmp_path).submit_count()
    submit_noise(tmp_path, brief, spec, reference)
    assert FakeJobs(tmp_path).submit_count() == first
    jobs = FakeJobs(tmp_path).submitted_jobs()
    assert jobs
    assert set(jobs) == {"train"}


def test_final_refuses_a_second_call(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("AMT_JOBS_BACKEND", "fake")
    model = tmp_path / "candidates" / "c00" / "model.py"
    model.parent.mkdir(parents=True)
    model.write_text("class Ranker: pass\n", encoding="utf-8")
    row = {
        "run_id": "demo",
        "node_id": "c00",
        "operator": "draft",
        "kind": "model",
        "family": "linear",
        "claim": "a linear ranker",
        "rung": "R3",
        "seed": 0,
        "metric": 0.5,
        "status": "promoted",
        "git_sha": "abc",
        "delta_vs_baseline": {"point": 0.01, "ci_lo": 0.0, "ci_hi": 0.02},
    }
    ledger = tmp_path / "ledger" / "results.jsonl"
    ledger.parent.mkdir(parents=True)
    ledger.write_text(json.dumps(row) + "\n", encoding="utf-8")
    path = open_final(tmp_path, "demo", "Ada Lovelace")
    assert path.is_file()
    with pytest.raises(FinalError, match="already"):
        open_final(tmp_path, "demo", "Ada Lovelace")
    with pytest.raises(FinalError, match="person"):
        open_final(tmp_path, "other", "amt")


def test_search_submits_train_and_confirm(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("AMT_JOBS_BACKEND", "fake")
    monkeypatch.setenv("AMT_AGENT_BACKEND", "fake")
    monkeypatch.setenv("AMT_GIT_SHA", "abc123")
    from amt.orchestrator import run_brief
    from tests.test_search import _full_script, _sign

    brief = _sign(tmp_path)
    _full_script(tmp_path)
    state = run_brief(brief, root=tmp_path)
    assert state.stage == "REPORT"
    jobs = FakeJobs(tmp_path).submitted_jobs()
    assert "train" in jobs
    assert "confirm" in jobs
    assert "hello_train" not in jobs


def test_missing_baselines_escalate(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("AMT_JOBS_BACKEND", "fake")
    monkeypatch.setenv("AMT_AGENT_BACKEND", "fake")
    monkeypatch.setenv("AMT_GIT_SHA", "abc123")
    monkeypatch.setenv("AMT_REQUIRE_BASELINES", "1")
    from amt.orchestrator import run_brief
    from tests.test_search import _full_script, _sign

    brief = _sign(tmp_path, run_id="need-base")
    _full_script(tmp_path, "need-base")
    state = run_brief(brief, root=tmp_path, run_id="need-base")
    assert state.stage == "ESCALATE"
    text = (tmp_path / "runs" / "need-base" / "ESCALATE.md").read_text(encoding="utf-8")
    assert "amt noise --submit" in text


def test_settings_tighten_a_brief(monkeypatch) -> None:
    monkeypatch.setenv("AMT_MAX_CONCURRENT_JOBS", "2")
    monkeypatch.setenv("AMT_CATALOG", "custom_catalog")
    settings = load_settings()
    assert settings.catalog == "custom_catalog"
    brief = _brief(total=10)
    brief.budget.max_concurrent_jobs = 4
    assert effective_concurrency(brief, settings) == 2
    brief.budget.max_concurrent_jobs = 1
    assert effective_concurrency(brief, settings) == 1


def test_fixed_cluster_payload_has_no_new_cluster() -> None:
    settings = load_settings()
    body = existing_cluster_payload(
        "train",
        {"candidate_id": "c00", "rung": "R1", "seed": "0"},
        workspace_path="/Workspace/amt",
        cluster_id="cluster-1",
        idempotency_token="token",
        settings=settings,
    )
    assert body["tasks"][0]["existing_cluster_id"] == "cluster-1"
    assert "new_cluster" not in body["tasks"][0]
    assert "libraries" not in body["tasks"][0]
    rendered = " ".join(body["tasks"][0]["spark_python_task"]["parameters"])
    assert "whl" not in rendered


def test_gpu_candidate_routes_to_the_gpu_cluster(monkeypatch) -> None:
    from amt.jobs_backend import _cluster_for

    monkeypatch.setenv("AMT_CLUSTER_ID", "cpu-cluster")
    monkeypatch.setenv("AMT_GPU_CLUSTER_ID", "gpu-cluster")
    settings = load_settings()
    assert _cluster_for({"device": "gpu"}, settings) == "gpu-cluster"
    assert _cluster_for({"device": "cpu"}, settings) == "cpu-cluster"
    monkeypatch.delenv("AMT_GPU_CLUSTER_ID")
    monkeypatch.delenv("AMT_GPU_NODE_TYPE", raising=False)
    assert load_settings().gpu_available is False


def test_shared_cluster_latency_is_suspect_not_a_kill() -> None:
    decision = decide(
        "R1",
        [_run("slow", latency_p95_ms=500)],
        [_baseline("R1")],
        _brief(latency=20),
        shared_cluster=True,
    )
    kept = next(item for item in decision.decisions if item.node_id == "slow")
    assert kept.status == "kept"
    assert kept.note == "latency_suspect"
    killed = decide(
        "R1",
        [_run("slow", latency_p95_ms=500)],
        [_baseline("R1")],
        _brief(latency=20),
        shared_cluster=False,
    )
    hard = next(item for item in killed.decisions if item.node_id == "slow")
    assert hard.status == "killed"
    assert hard.rule == "latency"


def test_doctor_reports_the_auth_method(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("AMT_JOBS_BACKEND", "fake")
    monkeypatch.delenv("DATABRICKS_TOKEN", raising=False)
    text = doctor_report(tmp_path)
    assert "backend: fake" in text
    assert "auth:" in text


def test_env_example_covers_every_setting_the_code_reads() -> None:
    example = (ROOT / ".env.example").read_text(encoding="utf-8")
    listed = set()
    for line in example.splitlines():
        stripped = line.strip().lstrip("#").strip()
        if "=" in stripped and stripped.split("=", 1)[0].isupper():
            listed.add(stripped.split("=", 1)[0].strip())
    listed.update(INTERNAL_ENV)
    missing = []
    for folder in ("amt", "harness"):
        for path in (ROOT / folder).rglob("*.py"):
            text = path.read_text(encoding="utf-8")
            for token in ("AMT_", "DATABRICKS_", "CURSOR_API_KEY"):
                start = 0
                while True:
                    index = text.find(f'"{token}', start)
                    if index < 0:
                        break
                    end = text.find('"', index + 1)
                    name = text[index + 1 : end]
                    if name.endswith("_") or "{" in name:
                        start = end + 1
                        continue
                    if name not in listed and name.isupper():
                        missing.append(name)
                    start = end + 1
    assert sorted(set(missing)) == []


def test_confirm_token_is_one_time(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("AMT_JOBS_BACKEND", "fake")
    from amt.tokens import consume_token, issue_token

    path = issue_token(tmp_path)
    assert Path(path).is_file()
    consume_token(path)
    assert not Path(path).is_file()


def test_deploy_command_vars_match_settings(monkeypatch) -> None:
    monkeypatch.setenv("AMT_CATALOG", "catalog_a")
    monkeypatch.setenv("AMT_SCHEMA", "schema_b")
    monkeypatch.setenv("AMT_BUNDLE_TARGET", "dev")
    settings = load_settings()
    assert settings.catalog == "catalog_a"
    assert settings.schema_name == "schema_b"
    assert settings.bundle_target == "dev"
    assert os.environ["AMT_CATALOG"] == "catalog_a"
    rendered = yaml.safe_load((ROOT / "databricks.yml").read_text(encoding="utf-8"))
    assert "catalog" in rendered["variables"]
    assert "schema" in rendered["variables"]


def test_cursor_api_key_counts_as_logged_in(monkeypatch) -> None:
    import amt.agents as agents

    monkeypatch.setattr(agents, "cursor_agent_bin", lambda: "/bin/cursor-agent")
    monkeypatch.setattr(agents.subprocess, "run", lambda *a, **k: pytest.fail("status ran"))
    monkeypatch.setenv("CURSOR_API_KEY", "key")
    assert agents.cursor_agent_logged_in()
