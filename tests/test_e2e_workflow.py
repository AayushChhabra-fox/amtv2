"""Cross-phase fake-backend path: prep --repair, sign ready, run, final, resume."""

import json
from pathlib import Path

import pytest
import yaml

from amt.cli import main
from amt.deploy import write_deployed
from amt.jobs_backend import FakeJobs
from amt.schemas import load_brief
from tests.test_search import _candidate, _modality_manifest, _pass

ROOT = Path(__file__).resolve().parents[1]
OUTCOMES = ("winner_promoted", "baseline_c00_promoted", "no_winner")
BAD_CAST = "CAST('bad' AS DOUBLE)"
GOOD_CAST = "CAST(1 AS DOUBLE)"
FEATURE_SQL = (
    "SELECT 1 AS query_id, 1 AS item_id, "
    f"{BAD_CAST} AS signal_value, "
    "TIMESTAMP '2026-01-01 00:00:00' AS available_at\n"
)
FIXED_SQL = FEATURE_SQL.replace(BAD_CAST, GOOD_CAST)
LADDER_METRICS = {
    "lgbm_full": 0.90,
    "lgbm_recency_category": 0.50,
    "neural_normal": 0.80,
    "neural_label_shuffled": 0.40,
    "embedding_trained": 0.70,
    "embedding_random_projection": 0.30,
    "linear_fitted": 0.60,
    "linear_label_shuffled": 0.20,
}


def test_demo_briefs_use_the_constraint_schema() -> None:
    for name in ("nfy_demo.yaml", "nfy_demo_v2.yaml"):
        brief = load_brief(ROOT / "briefs" / name)
        assert brief.min_effect == "auto"
        assert brief.constraints.allow_gpu_serving is False
        assert brief.constraints.allow_external_weights is False
        assert brief.constraints.license_policy == "commercial_only"
        serving = brief.efficiency.serving
        assert serving is not None
        assert serving.latency_p95_ms_per_100_candidates == 20
        assert brief.design_space is None or not brief.design_space.families


def test_legacy_gate_directories_are_gone() -> None:
    assert not (ROOT / "gates" / "g1").exists()
    assert not (ROOT / "gates" / "g2").exists()
    assert (ROOT / "gates" / "ready").is_dir()


@pytest.mark.parametrize("outcome", OUTCOMES)
def test_prep_repair_then_search_covers_each_outcome(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    outcome: str,
) -> None:
    run_id = outcome.replace("_", "-")
    _prepare_env(tmp_path, monkeypatch)
    _write_workspace(tmp_path, run_id=run_id, outcome=outcome, k=2)
    _script_faulty_search(tmp_path, run_id)
    _patch_repair(monkeypatch)
    assert main(["prep", "--brief", "briefs/workflow.yaml", "--repair"]) == 0
    signed = main(
        ["sign", "ready", "--brief", "briefs/workflow.yaml", "--signer", "Ada Lovelace"]
    )
    assert signed == 0
    packet = tmp_path / "gates" / "ready" / "workflow"
    assert (packet / "ready.json").is_file()
    assert (packet / "ready_report.md").is_file()
    sql = (tmp_path / "datasets" / "workflow_spec" / "features" / "signals.sql").read_text(
        encoding="utf-8"
    )
    assert GOOD_CAST in sql
    diff = (
        tmp_path / "runs" / "dataset-repair" / "workflow_spec" / "1" / "diff.patch"
    ).read_text(encoding="utf-8")
    assert GOOD_CAST in diff
    _patch_r1_ranker_fault(monkeypatch)
    assert main(["run", "briefs/workflow.yaml"]) == 0
    captured = capsys.readouterr()
    assert f"OUTCOME: {outcome}" in captured.out
    report = json.loads((tmp_path / "runs" / run_id / "report.json").read_text(encoding="utf-8"))
    assert report["outcome"] == outcome
    debug = (tmp_path / "runs" / run_id / "candidates" / "c01" / "debug_log.md").read_text(
        encoding="utf-8"
    )
    assert "Ranker.score blew up" in debug
    assert not (tmp_path / "runs" / run_id / "candidates" / "c02" / "model.py").exists()
    calls = [
        json.loads(line)
        for line in (tmp_path / "runs" / run_id / "agent_calls.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    assert any(row["role"] == "planner" and row["retries"] >= 1 and row["ok"] for row in calls)
    verdicts = (tmp_path / "runs" / run_id / "round_1_verdicts.json").read_text(encoding="utf-8")
    assert "REVISE" in verdicts
    assert main(["final", "--run-id", run_id, "--signer", "Ada Lovelace"]) == 0
    assert (tmp_path / "runs" / run_id / "final_holdout.md").is_file()
    again = main(["final", "--run-id", run_id, "--signer", "Ada Lovelace"])
    assert again == 1
    assert "already" in capsys.readouterr().out


def test_resume_after_a_cluster_escalate(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    _prepare_env(tmp_path, monkeypatch)
    _write_workspace(tmp_path, run_id="cluster", outcome="winner_promoted", k=1)
    _script(tmp_path, "cluster", _plain_round("c01"))
    _patch_repair(monkeypatch)
    assert main(["prep", "--brief", "briefs/workflow.yaml", "--repair"]) == 0
    signed = main(
        ["sign", "ready", "--brief", "briefs/workflow.yaml", "--signer", "Ada Lovelace"]
    )
    assert signed == 0
    _patch_cluster_kills(monkeypatch, times=2)
    assert main(["run", "briefs/workflow.yaml"]) == 1
    text = (tmp_path / "runs" / "cluster" / "ESCALATE.md").read_text(encoding="utf-8")
    assert "infra_crash" in text
    saved = json.loads((tmp_path / "runs" / "cluster" / "state.json").read_text(encoding="utf-8"))
    assert saved["escalated_from_stage"] == "SUBMIT"
    assert main(["resume", "--run-id", "cluster"]) == 0
    assert "OUTCOME: winner_promoted" in capsys.readouterr().out


def test_resume_after_a_budget_escalate(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    _prepare_env(tmp_path, monkeypatch)
    monkeypatch.setenv("AMT_AGENT_BACKEND", "stub")
    path = tmp_path / "briefs" / "toy.yaml"
    path.parent.mkdir(parents=True)
    path.write_text(
        yaml.safe_dump(
            {
                "name": "toy",
                "run_id": "toy",
                "approval_mode": "none",
                "k": 1,
                "candidate": {
                    "id": "c01",
                    "family": "linear",
                    "rung": "R1",
                    "seed": 0,
                    "batch": "toy",
                    "claim": "ridge ranker on numeric feat columns",
                    "est_dbu": 1,
                },
                "budget": {"total_dbu": 0.01},
            }
        ),
        encoding="utf-8",
    )
    assert main(["run", "briefs/toy.yaml"]) == 1
    assert "refusing submit" in (tmp_path / "runs" / "toy" / "ESCALATE.md").read_text(
        encoding="utf-8"
    )
    capsys.readouterr()
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    payload["budget"]["total_dbu"] = 10
    path.write_text(yaml.safe_dump(payload), encoding="utf-8")
    monkeypatch.setenv("AMT_STOP_AFTER", "SUBMIT")
    assert main(["resume", "--run-id", "toy"]) == 1
    saved = json.loads((tmp_path / "runs" / "toy" / "state.json").read_text(encoding="utf-8"))
    assert saved["stage"] == "WAIT"
    assert saved["submitted_jobs"]["c01/R1/0"]["job_run_id"]
    assert "stage WAIT" in capsys.readouterr().out


def _prepare_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("AMT_JOBS_BACKEND", "fake")
    monkeypatch.setenv("AMT_AGENT_BACKEND", "fake")
    monkeypatch.setenv("AMT_GIT_SHA", "abc123")
    monkeypatch.setenv("AMT_DBU_PER_HOUR", "1")
    monkeypatch.setenv("AMT_DATASET_REPAIR_MAX_ATTEMPTS", "3")
    monkeypatch.setenv("AMT_DATASET_REPAIR_MAX_DBU", "10")
    monkeypatch.delenv("AMT_CLUSTER_ID", raising=False)
    monkeypatch.setattr("amt.agents.time.sleep", lambda *_args, **_kwargs: None)


def _write_workspace(tmp_path: Path, *, run_id: str, outcome: str, k: int) -> None:
    spec = tmp_path / "datasets" / "workflow_spec"
    features = spec / "features"
    features.mkdir(parents=True)
    (spec / "queries.sql").write_text("SELECT 1 AS query_id\n", encoding="utf-8")
    (spec / "candidates.sql").write_text(
        "SELECT 1 AS query_id, 1 AS item_id\n", encoding="utf-8"
    )
    (spec / "labels.sql").write_text(
        "SELECT 1 AS query_id, 1 AS item_id, 1 AS label\n", encoding="utf-8"
    )
    (features / "signals.sql").write_text(FEATURE_SQL, encoding="utf-8")
    (spec / "dataset.yaml").write_text(_SPEC_YAML, encoding="utf-8")
    brief = tmp_path / "briefs" / "workflow.yaml"
    brief.parent.mkdir(parents=True)
    brief.write_text(
        yaml.safe_dump(
            {
                "name": "workflow",
                "run_id": run_id,
                "approval_mode": "none",
                "dataset_spec": "datasets/workflow_spec",
                "min_effect": "auto",
                "k": k,
                "max_rounds": 1,
                "problem": {
                    "surface": "workflow",
                    "metric": {
                        "name": "ndcg@10_graded",
                        "gains": {"impressed": 0, "click": 1},
                    },
                },
                "constraints": {
                    "allow_gpu_serving": False,
                    "allow_external_weights": False,
                    "license_policy": "commercial_only",
                },
                "efficiency": {
                    "train_dbu_per_candidate": 1,
                    "serving": {"latency_p95_ms_per_100_candidates": 20},
                },
                "budget": {"total_dbu": 40, "max_concurrent_jobs": 2},
            }
        ),
        encoding="utf-8",
    )
    amt = tmp_path / ".amt"
    amt.mkdir(parents=True)
    (amt / "fake_manifest.json").write_text(
        json.dumps({"dataset": "workflow_spec", "modality": _modality_manifest()["modality"]})
        + "\n",
        encoding="utf-8",
    )
    (amt / "fake_metrics.json").write_text(
        json.dumps(_metrics(outcome)) + "\n",
        encoding="utf-8",
    )
    fixer = tmp_path / "runs" / "dataset-repair-workflow_spec" / "fake_agent.json"
    fixer.parent.mkdir(parents=True)
    fixer.write_text(
        json.dumps(
            {
                "cursor": 0,
                "calls": [
                    {
                        "role": "dataset_fixer",
                        "output": {"edits": [{"path": "features/signals.sql", "sql": FIXED_SQL}]},
                    }
                ],
            }
        )
        + "\n",
        encoding="utf-8",
    )


def _patch_repair(monkeypatch: pytest.MonkeyPatch) -> None:
    def preflight(root, spec_path, brief, *, remote=True, slice_window="1h"):
        del brief, remote, slice_window
        path = Path(spec_path)
        if not path.is_absolute():
            path = Path(root) / path
        sql = (path / "features" / "signals.sql").read_text(encoding="utf-8")
        if BAD_CAST in sql:
            return {
                "status": "fail",
                "failure": {
                    "kind": "",
                    "fragment": "signals",
                    "fragment_path": "features/signals.sql",
                    "exception_type": "AnalysisException",
                    "message": "cannot cast string to double",
                    "traceback": "",
                    "examples": [],
                },
            }
        return {"status": "pass", "search_val_days": 5, "positive_grades": [1]}

    monkeypatch.setattr("amt.dataset_repair.run_preflight", preflight)
    monkeypatch.setattr(
        "amt.dataset_repair.deploy",
        lambda root: write_deployed(root, git_sha="abc123"),
    )


def _patch_r1_ranker_fault(monkeypatch: pytest.MonkeyPatch) -> None:
    real_wait = FakeJobs.wait
    seen = {"done": False}

    def wait(self, job_run_id, timeout_s=None):
        payload = self._load()
        for record in payload["submits"].values():
            if record["job_run_id"] != job_run_id:
                continue
            params = record.get("params") or {}
            if (
                not seen["done"]
                and params.get("candidate_id") == "c01"
                and params.get("rung") == "R1"
                and record.get("job") == "train"
            ):
                seen["done"] = True
                directory = (
                    self.root / "runs" / str(params.get("run_id")) / "candidates" / "c01"
                )
                directory.mkdir(parents=True, exist_ok=True)
                (directory / "failure.json").write_text(
                    json.dumps(
                        {
                            "exception_type": "ValueError",
                            "message": "Ranker.score blew up",
                            "traceback": (
                                "Traceback (most recent call last):\n"
                                "ValueError: Ranker.score blew up\n"
                            ),
                        }
                    ),
                    encoding="utf-8",
                )
                return {
                    "life_cycle_state": "TERMINATED",
                    "result_state": "FAILED",
                    "state_message": "Task failed",
                }
        return real_wait(self, job_run_id, timeout_s)

    monkeypatch.setattr(FakeJobs, "wait", wait)


def _patch_cluster_kills(monkeypatch: pytest.MonkeyPatch, *, times: int) -> None:
    real_wait = FakeJobs.wait
    left = {"n": times}

    def wait(self, job_run_id, timeout_s=None):
        if left["n"] > 0:
            left["n"] -= 1
            return {
                "life_cycle_state": "INTERNAL_ERROR",
                "result_state": "FAILED",
                "state_message": "cluster driver was killed",
            }
        return real_wait(self, job_run_id, timeout_s)

    monkeypatch.setattr(FakeJobs, "wait", wait)


def _metrics(outcome: str) -> dict:
    strong = {
        "ndcg@10_graded": 0.55,
        "delta_point": 0.02,
        "delta_ci_lo": 0.019,
        "delta_ci_hi": 0.021,
        "delta_lgbm_point": 0.0,
        "delta_lgbm_ci_lo": -0.001,
        "delta_lgbm_ci_hi": 0.001,
        "latency_p95_ms": 5.0,
    }
    weak = {
        **strong,
        "ndcg@10_graded": 0.40,
        "delta_point": 0.0,
        "delta_ci_lo": -0.001,
        "delta_ci_hi": 0.001,
    }
    if outcome == "no_winner":
        base = weak
        candidate = {
            **weak,
            "ndcg@10_graded": 0.50,
            "delta_lgbm_point": -0.01,
            "delta_lgbm_ci_lo": -0.02,
            "delta_lgbm_ci_hi": 0.0,
        }
        reference = weak
    elif outcome == "baseline_c00_promoted":
        base = strong
        reference = {**strong, "ndcg@10_graded": 0.60}
        candidate = {
            **strong,
            "ndcg@10_graded": 0.50,
            "delta_lgbm_point": -0.01,
            "delta_lgbm_ci_lo": -0.02,
            "delta_lgbm_ci_hi": -0.005,
        }
    else:
        base = strong
        reference = {**strong, "ndcg@10_graded": 0.55}
        candidate = {
            **strong,
            "ndcg@10_graded": 0.70,
            "delta_lgbm_point": 0.02,
            "delta_lgbm_ci_lo": 0.01,
            "delta_lgbm_ci_hi": 0.03,
        }
    table: dict = {"*": base, "lgbm_reference": reference, "c01": candidate}
    for design, metric in LADDER_METRICS.items():
        table[design] = {"ndcg@10_graded": metric}
    return table


def _plain_round(node_id: str) -> list[dict]:
    return [
        {"role": "planner", "output": {"candidates": [_candidate(node_id, "draft")]}},
        {"role": "critic", "output": {"verdicts": [_pass(node_id)]}},
        {"role": "implementer", "node_id": node_id},
        {"role": "reviewer", "node_id": node_id},
    ]


def _script_faulty_search(tmp_path: Path, run_id: str) -> None:
    proposal = {
        "candidates": [
            _candidate("c01", "draft"),
            _candidate("c02", "draft", claim="a second linear draft"),
        ]
    }
    revise = {
        "candidate_id": "c02",
        "decision": "REVISE",
        "reason": "The second draft repeats a serve-time input.",
    }
    mixed = {"verdicts": [_pass("c01"), revise]}
    _script(
        tmp_path,
        run_id,
        [
            {"role": "planner", "error": "resource_exhausted"},
            {"role": "planner", "output": proposal},
            {"role": "critic", "output": mixed},
            {"role": "planner", "output": proposal},
            {"role": "critic", "output": mixed},
            {"role": "implementer", "node_id": "c01"},
            {"role": "reviewer", "node_id": "c01"},
            {"role": "implementer", "node_id": "c01"},
            {"role": "reviewer", "node_id": "c01"},
        ],
    )


def _script(tmp_path: Path, run_id: str, calls: list[dict]) -> None:
    path = tmp_path / "runs" / run_id / "fake_agent.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"cursor": 0, "calls": calls}) + "\n", encoding="utf-8")


_SPEC_YAML = """
name: workflow_spec
task_type: ranking
primary_metric: ndcg@10_graded
sources:
  requests:
    table: src_requests
    unpartitioned: true
queries: queries.sql
candidates: candidates.sql
labels:
  sql: labels.sql
  label_window: 24h
feature_groups:
  signals:
    sql: features/signals.sql
    keys: [query_id, item_id]
    available_at: available_at
reference_scores:
  production: total_score
split:
  time_col: query_ts
  gap_days: 1
  train: 2026-01-01..2026-01-02
  search_val: 2026-01-04..2026-01-08
  select_holdout: 2026-01-10..2026-01-11
  final_holdout: 2026-01-13..2026-01-14
sampling:
  seed: 0
  smoke:
    train_queries: 1
    eval_queries: 1
  rungs:
    R1: {train_fraction: 1.0, eval_queries: 1}
    R2: {train_fraction: 1.0, eval_queries: all}
    R3: {train_fraction: 1.0, eval_queries: all}
preflight:
  min_search_val_days: 5
""".lstrip()
