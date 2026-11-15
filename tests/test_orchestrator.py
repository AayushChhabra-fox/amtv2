import argparse
import json
from pathlib import Path

import yaml

from amt.cli import main
from amt.jobs_backend import FakeJobs
from amt.orchestrator import _search_submit, _search_wait, run_brief
from amt.schemas import Brief, NodeWork, RunState


def _write_brief(root: Path) -> Path:
    path = root / "briefs" / "toy.yaml"
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
                    "est_dbu": 0.1,
                },
                "budget": {"total_dbu": 10},
            }
        )
    )
    return path


def test_resume_does_not_resubmit(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("AMT_JOBS_BACKEND", "fake")
    monkeypatch.setenv("AMT_GIT_SHA", "abc123")
    monkeypatch.setenv("AMT_STOP_AFTER", "SUBMIT")
    brief = _write_brief(tmp_path)
    first = run_brief(brief, root=tmp_path)
    assert first.stage == "WAIT"
    assert FakeJobs(tmp_path).submit_count() == 1
    saved = json.loads((tmp_path / "runs" / "toy" / "state.json").read_text())
    assert saved["submitted_jobs"]["c01/R1/0"]["job_run_id"] == "fake-1"

    monkeypatch.setenv("AMT_STOP_AFTER", "WAIT")
    second = run_brief(brief, root=tmp_path)
    assert second.stage == "WAIT"
    assert second.submitted_jobs["c01/R1/0"]["result_state"] == "SUCCESS"
    assert second.revisions == 0
    assert FakeJobs(tmp_path).submit_count() == 1
    saved = json.loads((tmp_path / "runs" / "toy" / "state.json").read_text())
    assert saved["submitted_jobs"]["c01/R1/0"]["job_run_id"] == "fake-1"
    model = (tmp_path / "runs" / "toy" / "candidates" / "c01" / "model.py").read_text(
        encoding="utf-8"
    )
    assert "class Ranker" in model
    assert "hello_train" not in FakeJobs(tmp_path).submitted_jobs()


def test_idempotent_token_returns_same_run(tmp_path: Path) -> None:
    backend = FakeJobs(tmp_path)
    params = {"candidate_id": "c00", "rung": "R0", "seed": "0"}
    first = backend.submit(job_key="c00/R0/0", params=params, idempotency_token="toy:c00/R0/0")
    second = backend.submit(job_key="c00/R0/0", params=params, idempotency_token="toy:c00/R0/0")
    assert first == second == "fake-1"
    assert backend.submit_count() == 1


def test_pool_refusal_does_not_submit(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("AMT_JOBS_BACKEND", "fake")
    monkeypatch.setenv("AMT_GIT_SHA", "abc123")
    brief = _write_brief(tmp_path)
    payload = yaml.safe_load(brief.read_text())
    payload["budget"]["total_dbu"] = 0.01
    brief.write_text(yaml.safe_dump(payload))
    state = run_brief(brief, root=tmp_path)
    assert state.stage == "ESCALATE"
    assert FakeJobs(tmp_path).submit_count() == 0
    assert "refusing submit" in (tmp_path / "runs" / "toy" / "ESCALATE.md").read_text()


def test_candidate_cap_refusal_does_not_submit(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("AMT_JOBS_BACKEND", "fake")
    monkeypatch.setenv("AMT_GIT_SHA", "abc123")
    brief = _write_brief(tmp_path)
    payload = yaml.safe_load(brief.read_text())
    payload["efficiency"] = {"train_dbu_per_candidate": 0.01}
    brief.write_text(yaml.safe_dump(payload))
    state = run_brief(brief, root=tmp_path)
    assert state.stage == "ESCALATE"
    assert FakeJobs(tmp_path).submit_count() == 0
    assert "cap" in (tmp_path / "runs" / "toy" / "ESCALATE.md").read_text()


def test_completed_run_is_a_no_op(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("AMT_JOBS_BACKEND", "fake")
    monkeypatch.setenv("AMT_GIT_SHA", "abc123")
    brief = _write_brief(tmp_path)
    first = run_brief(brief, root=tmp_path)
    assert first.stage in {"REPORT", "ESCALATE"}
    count = FakeJobs(tmp_path).submit_count()
    assert count >= 1
    assert "hello_train" not in FakeJobs(tmp_path).submitted_jobs()
    second = run_brief(brief, root=tmp_path)
    assert second.stage == first.stage
    assert FakeJobs(tmp_path).submit_count() == count


class _Results:
    def __init__(self, results: list[dict]) -> None:
        self.results = list(results)

    def wait(self, job_run_id: str) -> dict:
        del job_run_id
        return self.results.pop(0)


def _search_state(*nodes: NodeWork) -> RunState:
    return RunState(
        run_id="toy",
        brief_path="briefs/toy.yaml",
        stage="WAIT",
        round=1,
        candidate_id="c01",
        family="linear",
        rung="R1",
        seed=0,
        batch="toy",
        budget_dbu=10,
        nodes=list(nodes),
    )


def _submitted(node_id: str, rung: str = "R1", status: str = "submitted") -> NodeWork:
    return NodeWork(
        node_id=node_id,
        family="linear",
        status=status,  # type: ignore[arg-type]
        rung=rung,  # type: ignore[arg-type]
        search_round=1,
        preflight="PASS",
        review="PASS",
    )


def test_r1_runtime_failure_debugs_one_node_and_keeps_its_sibling(tmp_path: Path) -> None:
    candidate_failure = {
        "exception_type": "ValueError",
        "message": "bad ranker",
        "traceback": "Traceback (most recent call last):\nValueError: bad ranker\n",
    }
    directory = tmp_path / "runs" / "toy" / "candidates" / "c01"
    directory.mkdir(parents=True)
    (directory / "failure.json").write_text(json.dumps(candidate_failure), encoding="utf-8")
    state = _search_state(_submitted("c01"), _submitted("c02"))
    state.submitted_jobs = {
        "c01/R1/0": {"job_run_id": "j1", "submitted_at": 0.0, "dbu_est": 0.1},
        "c02/R1/0": {"job_run_id": "j2", "submitted_at": 0.0, "dbu_est": 0.1},
    }
    failed_result = {
        "life_cycle_state": "TERMINATED",
        "result_state": "FAILED",
        "state_message": "Task failed",
    }
    ok = {"life_cycle_state": "TERMINATED", "result_state": "SUCCESS"}
    state = _search_wait(state, tmp_path, _Results([failed_result]))
    assert state.stage == "WAIT"
    assert state.nodes[0].status == "debugging"
    assert state.nodes[0].debug_attempts == 1
    assert "bad ranker" in (
        tmp_path / "runs" / "toy" / "candidates" / "c01" / "debug_log.md"
    ).read_text(encoding="utf-8")
    assert state.nodes[1].status == "submitted"
    state = _search_wait(state, tmp_path, _Results([ok]))
    assert state.nodes[1].status == "submitted"
    assert state.submitted_jobs["c02/R1/0"]["result_state"] == "SUCCESS"
    state = _search_wait(state, tmp_path, _Results([]))
    assert state.stage == "DEBUG"
    assert state.nodes[1].status == "submitted"
    assert state.submitted_jobs["c02/R1/0"]["job_run_id"] == "j2"


def test_r1_debug_cap_drops_only_that_node(tmp_path: Path) -> None:
    directory = tmp_path / "runs" / "toy" / "candidates" / "c01"
    directory.mkdir(parents=True)
    (directory / "failure.json").write_text(
        json.dumps({"traceback": "Traceback\nRuntimeError: still broken\n", "message": "still"}),
        encoding="utf-8",
    )
    node = _submitted("c01")
    node.debug_attempts = 3
    state = _search_state(node, _submitted("c02"))
    state.submitted_jobs = {
        "c01/R1/0": {"job_run_id": "j1", "submitted_at": 0.0, "dbu_est": 0.1},
        "c02/R1/0": {"job_run_id": "j2", "submitted_at": 0.0, "dbu_est": 0.1},
    }
    state = _search_wait(
        state,
        tmp_path,
        _Results(
            [{"life_cycle_state": "TERMINATED", "result_state": "FAILED", "state_message": "boom"}]
        ),
    )
    assert state.stage != "ESCALATE"
    assert state.nodes[0].status == "failed"
    assert state.submitted_jobs["c01/R1/0"]["failure_kind"] == "runtime_error"
    assert state.nodes[1].status == "submitted"


def test_fixed_cluster_runs_one_r3_job(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("AMT_CLUSTER_ID", "cluster-1")
    monkeypatch.setenv("AMT_JOBS_BACKEND", "fake")
    monkeypatch.setenv("AMT_GIT_SHA", "abc123")
    brief = Brief.model_validate({"name": "toy", "budget": {"total_dbu": 100}})
    nodes = []
    for node_id in ("c01", "c02"):
        directory = tmp_path / "runs" / "toy" / "candidates" / node_id
        directory.mkdir(parents=True)
        (directory / "model.py").write_text("class Ranker:\n    pass\n", encoding="utf-8")
        nodes.append(_submitted(node_id, rung="R3", status="ready"))
    state = _search_state(*nodes)
    state.stage = "SUBMIT"
    state.rung = "R3"
    backend = FakeJobs(tmp_path)
    state = _search_submit(state, brief, tmp_path, backend)
    assert backend.submit_count() == 1
    state = _search_submit(state, brief, tmp_path, backend)
    assert backend.submit_count() == 1
    assert state.stage == "WAIT"


def test_resume_retries_submit_after_a_budget_escalate(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("AMT_JOBS_BACKEND", "fake")
    monkeypatch.setenv("AMT_GIT_SHA", "abc123")
    brief = _write_brief(tmp_path)
    payload = yaml.safe_load(brief.read_text())
    payload["budget"]["total_dbu"] = 0.01
    brief.write_text(yaml.safe_dump(payload))
    state = run_brief(brief, root=tmp_path)
    assert state.stage == "ESCALATE"
    assert state.escalated_from_stage == "SUBMIT"
    payload["budget"]["total_dbu"] = 10
    brief.write_text(yaml.safe_dump(payload))
    monkeypatch.setenv("AMT_STOP_AFTER", "SUBMIT")
    code = main(["resume", "--run-id", "toy", "--root", str(tmp_path)])
    assert code == 1
    saved = json.loads((tmp_path / "runs" / "toy" / "state.json").read_text(encoding="utf-8"))
    assert saved["stage"] == "WAIT"
    assert saved["submitted_jobs"]["c01/R1/0"]["job_run_id"]


def test_ranker_exception_writes_failure_json(tmp_path: Path, monkeypatch) -> None:
    from harness.interface import RankingFrame
    from harness.train import _record_candidate_failure, train_ranker

    monkeypatch.setenv("AMT_TRAIN_FAILURE_ROOT", str(tmp_path / "vol"))
    frame = RankingFrame(
        [
            {
                "query_id": "q",
                "item_id": "a",
                "label": 1,
                "feat__num__x": 1.0,
                "event_day": "2024-01-01",
            },
            {
                "query_id": "q",
                "item_id": "b",
                "label": 0,
                "feat__num__x": 0.0,
                "event_day": "2024-01-01",
            },
        ]
    )

    class Boom:
        def fit(self, train, valid, ctx) -> None:
            del train, valid, ctx
            raise RuntimeError("ranker blew up")

        def score(self, frame) -> list[float]:
            del frame
            return []

        def save(self, path: str) -> None:
            del path

    result = train_ranker(
        ranker=Boom(),
        train=frame,
        evaluation=frame,
        seed=0,
        rung="R1",
        deadline_s=30,
        require_loss=False,
        check_determinism=False,
    )
    assert result.status == "failed"
    assert result.traceback and "ranker blew up" in result.traceback
    model = tmp_path / "candidates" / "c01" / "model.py"
    model.parent.mkdir(parents=True)
    model.write_text("class Ranker:\n    pass\n", encoding="utf-8")
    args = argparse.Namespace(
        node_id="c01",
        candidate_id="c01",
        run_id="toy",
        batch="toy",
        rung="R1",
        seed=0,
        baseline="",
    )
    _record_candidate_failure(
        args,
        model,
        "cat",
        "sch",
        {"exception_type": "RuntimeError", "message": "ranker blew up", "traceback": "boom"},
    )
    local = json.loads((model.parent / "failure.json").read_text(encoding="utf-8"))
    remote = tmp_path / "vol" / "train_failures" / "toy" / "c01" / "R1" / "0" / "failure.json"
    assert local["message"] == "ranker blew up"
    assert json.loads(remote.read_text(encoding="utf-8"))["traceback"] == "boom"
