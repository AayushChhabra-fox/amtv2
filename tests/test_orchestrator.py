import json
from pathlib import Path

import yaml

from amt.jobs_backend import FakeJobs
from amt.orchestrator import run_brief
from amt.status import audit_rows


def _write_brief(root: Path) -> Path:
    path = root / "briefs" / "toy.yaml"
    path.parent.mkdir(parents=True)
    path.write_text(
        yaml.safe_dump(
            {
                "name": "toy",
                "run_id": "toy",
                "approval_mode": "none",
                "candidate": {
                    "id": "c00",
                    "family": "linear",
                    "rung": "R0",
                    "seed": 0,
                    "batch": "toy",
                    "claim": "logistic regression",
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
    assert saved["submitted_jobs"]["c00/R0/0"]["job_run_id"] == "fake-1"

    monkeypatch.delenv("AMT_STOP_AFTER")
    second = run_brief(brief, root=tmp_path)
    assert second.stage == "REPORT"
    assert second.revisions == 0
    assert FakeJobs(tmp_path).submit_count() == 1
    assert (tmp_path / "candidates" / "c00" / "model.py").is_file()
    assert audit_rows(tmp_path, "toy") == []


def test_idempotent_token_returns_same_run(tmp_path: Path) -> None:
    backend = FakeJobs(tmp_path)
    params = {"candidate_id": "c00", "rung": "R0", "seed": "0"}
    first = backend.submit(job_key="c00/R0/0", params=params, idempotency_token="toy:c00/R0/0")
    second = backend.submit(job_key="c00/R0/0", params=params, idempotency_token="toy:c00/R0/0")
    assert first == second == "fake-1"
    assert backend.submit_count() == 1


def test_completed_run_is_a_no_op(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("AMT_JOBS_BACKEND", "fake")
    monkeypatch.setenv("AMT_GIT_SHA", "abc123")
    brief = _write_brief(tmp_path)
    run_brief(brief, root=tmp_path)
    run_brief(brief, root=tmp_path)
    assert FakeJobs(tmp_path).submit_count() == 1
