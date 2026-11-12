"""Offline path from a fake deploy through noise, search, confirm, and final."""

import json
from pathlib import Path

import pytest

from amt.batch import collect_noise, submit_noise
from amt.deploy import write_deployed
from amt.final import open_final
from amt.jobs_backend import FakeJobs
from amt.orchestrator import run_brief
from amt.schemas import load_brief
from amt.status import audit_rows
from harness.data.contract import load_spec
from tests.test_search import _full_script, _sign

ROOT = Path(__file__).resolve().parents[1]


def test_fake_backend_runs_deploy_noise_search_and_final(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("AMT_JOBS_BACKEND", "fake")
    monkeypatch.setenv("AMT_AGENT_BACKEND", "fake")
    monkeypatch.setenv("AMT_GIT_SHA", "abc123")
    write_deployed(tmp_path, git_sha="abc123")
    spec_path = ROOT / "datasets" / "nfy_article_rerank"
    spec = load_spec(spec_path)
    brief = load_brief(ROOT / "briefs" / "toy.yaml")
    submit_noise(tmp_path, brief, str(spec_path), list(spec.reference_scores))
    status, report = collect_noise(
        tmp_path,
        brief,
        ROOT / "briefs" / "toy.yaml",
        spec.name,
        list(spec.reference_scores),
    )
    assert status in {"pass", "revise", "incomplete"}
    assert report.is_file()
    observations = json.loads(
        (tmp_path / "runs" / "round0-toy" / "round0.json").read_text(encoding="utf-8")
    )
    assert observations
    assert {row["split"] for row in observations} == {"search_val"}

    search = _sign(tmp_path, run_id="e2e")
    _full_script(tmp_path, "e2e")
    state = run_brief(search, root=tmp_path, run_id="e2e")
    assert state.stage == "REPORT"
    jobs = set(FakeJobs(tmp_path).submitted_jobs())
    assert "train" in jobs
    assert "confirm" in jobs
    assert audit_rows(tmp_path, "e2e") == []

    model = tmp_path / "candidates" / "c03" / "model.py"
    assert model.is_file()
    path = open_final(tmp_path, "e2e", "Ada Lovelace")
    assert "Final holdout" in path.read_text(encoding="utf-8")
    with pytest.raises(Exception, match="already"):
        open_final(tmp_path, "e2e", "Ada Lovelace")
