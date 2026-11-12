"""Fake-agent search loop: resume, caps, budget refusal, and approval gates."""

import json
from datetime import datetime, timezone
from pathlib import Path

import yaml

from amt.agents import run_role
from amt.cli import main
from amt.jobs_backend import FakeJobs
from amt.ladder import build_ladder_report
from amt.noise import build_noise_report
from amt.orchestrator import fork_on_plateau, operator_for_round, run_brief
from amt.schemas import Brief, load_brief
from amt.signoff import sign_g1, sign_g2, write_ladder_report, write_noise_report
from amt.status import audit_rows
from tests.test_ladder import _cells as ladder_cells
from tests.test_noise import _brief, _cells

ROOT = Path(__file__).resolve().parents[1]
SIGNED = datetime(2026, 11, 12, 15, 0, tzinfo=timezone.utc)
METRICS = {
    "*": {
        "ndcg@10_graded": 0.45,
        "delta_point": 0.02,
        "delta_ci_lo": 0.011,
        "delta_ci_hi": 0.03,
    }
}


def _sign(
    tmp_path: Path,
    *,
    total_dbu: float = 1500,
    run_id: str = "search",
    k: int = 1,
    approval_mode: str = "none",
) -> Path:
    path = tmp_path / "briefs" / "nfy.yaml"
    path.parent.mkdir(parents=True)
    payload = _brief().model_dump(mode="json")
    payload["dataset_spec"] = str(ROOT / "datasets" / "nfy_article_rerank")
    payload["run_id"] = run_id
    payload["k"] = k
    payload["approval_mode"] = approval_mode
    payload["budget"]["total_dbu"] = total_dbu
    path.write_text(yaml.safe_dump(payload), encoding="utf-8")
    brief = load_brief(path)
    report = build_noise_report(
        _cells(),
        brief,
        dataset="nfy_article_rerank",
        reference_scores=["production", "recency"],
    )
    assert report.status == "pass"
    write_noise_report(tmp_path, brief, path, report)
    manifest = tmp_path / "manifest.json"
    manifest.write_text('{"rows": 1}\n', encoding="utf-8")
    sign_g1(tmp_path, path, "Ada Lovelace", manifest, signed_at=SIGNED)
    ladder = build_ladder_report(ladder_cells(), dataset="nfy_article_rerank")
    write_ladder_report(tmp_path, ladder)
    sign_g2(tmp_path, "nfy_article_rerank", "Grace Hopper", signed_at=SIGNED)
    metrics = tmp_path / ".amt" / "fake_metrics.json"
    metrics.parent.mkdir(parents=True, exist_ok=True)
    metrics.write_text(json.dumps(METRICS), encoding="utf-8")
    return path


def _candidate(
    node_id: str,
    operator: str,
    *,
    parent: str | None = None,
    change: str | None = None,
    claim: str | None = None,
    est: float = 0.1,
) -> dict:
    payload = {
        "id": node_id,
        "operator": operator,
        "family": "linear",
        "claim": claim or f"{operator} hypothesis for {node_id}",
        "fit_argument": "A linear score ranks reads above bounces.",
        "cheap_test": "Smoke loss falls on the frame passed to fit.",
        "kill_if": "Scores are non-finite.",
        "est_cost": est,
    }
    if parent:
        payload["parent_id"] = parent
    if change:
        payload["change"] = change
    return payload


def _pass(node_id: str) -> dict:
    return {
        "candidate_id": node_id,
        "decision": "PASS",
        "reason": "Serve-time inputs match the task block and the claim is new.",
    }


def _script(tmp_path: Path, run_id: str, calls: list[dict]) -> None:
    path = tmp_path / "runs" / run_id / "fake_agent.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"cursor": 0, "calls": calls}, indent=2) + "\n", encoding="utf-8")


def _round(
    node_id: str,
    operator: str,
    *,
    parent: str | None = None,
    change: str | None = None,
) -> list[dict]:
    return [
        {
            "role": "planner",
            "output": {
                "candidates": [
                    _candidate(node_id, operator, parent=parent, change=change),
                ]
            },
        },
        {"role": "critic", "output": {"verdicts": [_pass(node_id)]}},
        {"role": "implementer", "node_id": node_id},
        {"role": "reviewer", "node_id": node_id},
    ]


def _full_script(tmp_path: Path, run_id: str = "search") -> None:
    calls: list[dict] = []
    calls.extend(_round("c00", "draft"))
    calls.extend(
        _round("c01", "improve", parent="c00", change="add one recency interaction")
    )
    calls.extend(
        _round("c02", "improve", parent="c01", change="drop the noisiest numeric feature")
    )
    calls.extend(_round("c03", "combine", parent="c02", change="blend scores with c01"))
    _script(tmp_path, run_id, calls)


def test_operator_schedule_and_plateau() -> None:
    assert operator_for_round(1, plateau=False, lineage_count=1) == "draft"
    assert operator_for_round(2, plateau=False, lineage_count=1) == "improve"
    assert operator_for_round(3, plateau=True, lineage_count=1) == "draft"
    assert operator_for_round(4, plateau=True, lineage_count=2) == "combine"
    assert operator_for_round(4, plateau=False, lineage_count=1) == "confirm"
    assert operator_for_round(5, plateau=False, lineage_count=2) == "confirm"
    assert fork_on_plateau([0.0, 0.0], 0.01) is True
    assert fork_on_plateau([0.02], 0.01) is False
    assert fork_on_plateau([0.0, 0.05], 0.01) is False


def test_fake_loop_reaches_report(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("AMT_JOBS_BACKEND", "fake")
    monkeypatch.setenv("AMT_AGENT_BACKEND", "fake")
    monkeypatch.setenv("AMT_GIT_SHA", "abc123")
    brief = _sign(tmp_path)
    _full_script(tmp_path)
    state = run_brief(brief, root=tmp_path)
    assert state.stage == "REPORT"
    assert state.promoted == ["c03"]
    assert FakeJobs(tmp_path).submit_count() == 11
    assert (tmp_path / "runs" / "search" / "round_1.md").is_file()
    digest = (tmp_path / "runs" / "search" / "round_1.md").read_text(encoding="utf-8")
    assert "Spent" in digest
    assert "c00" in digest
    report = (tmp_path / "runs" / "search" / "report.md").read_text(encoding="utf-8")
    assert "c03" in report
    assert audit_rows(tmp_path, "search") == []
    again = run_brief(brief, root=tmp_path)
    assert again.stage == "REPORT"
    assert FakeJobs(tmp_path).submit_count() == 11


def test_resume_does_not_resubmit_the_search_job(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("AMT_JOBS_BACKEND", "fake")
    monkeypatch.setenv("AMT_AGENT_BACKEND", "fake")
    monkeypatch.setenv("AMT_GIT_SHA", "abc123")
    monkeypatch.setenv("AMT_STOP_AFTER", "SUBMIT")
    brief = _sign(tmp_path)
    _full_script(tmp_path)
    first = run_brief(brief, root=tmp_path)
    assert first.stage == "WAIT"
    assert FakeJobs(tmp_path).submit_count() == 1
    saved = json.loads((tmp_path / "runs" / "search" / "state.json").read_text(encoding="utf-8"))
    assert saved["submitted_jobs"]["c00/R1/0"]["job_run_id"] == "fake-1"

    monkeypatch.delenv("AMT_STOP_AFTER")
    second = run_brief(brief, root=tmp_path)
    assert second.stage == "REPORT"
    assert FakeJobs(tmp_path).submit_count() == 11
    resumed = json.loads((tmp_path / "runs" / "search" / "state.json").read_text(encoding="utf-8"))
    assert resumed["submitted_jobs"]["c00/R1/0"]["job_run_id"] == "fake-1"


def test_debug_cap_escalates(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("AMT_JOBS_BACKEND", "fake")
    monkeypatch.setenv("AMT_AGENT_BACKEND", "fake")
    monkeypatch.setenv("AMT_GIT_SHA", "abc123")
    brief = _sign(tmp_path, run_id="debug")
    bad = "import pyspark\n\n"
    calls = [
        {"role": "planner", "output": {"candidates": [_candidate("c00", "draft")]}},
        {"role": "critic", "output": {"verdicts": [_pass("c00")]}},
    ]
    for _ in range(4):
        calls.append(
            {
                "role": "implementer",
                "node_id": "c00",
                "files": {"candidates/c00/model.py": bad},
            }
        )
    _script(tmp_path, "debug", calls)
    state = run_brief(brief, root=tmp_path)
    assert state.stage == "ESCALATE"
    text = (tmp_path / "runs" / "debug" / "ESCALATE.md").read_text(encoding="utf-8")
    assert "debug cap" in text
    log = (tmp_path / "candidates" / "c00" / "debug_log.md").read_text(encoding="utf-8")
    assert "attempt 3" in log
    assert FakeJobs(tmp_path).submit_count() == 0


def test_critic_revision_cap_escalates(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("AMT_JOBS_BACKEND", "fake")
    monkeypatch.setenv("AMT_AGENT_BACKEND", "fake")
    monkeypatch.setenv("AMT_GIT_SHA", "abc123")
    brief = _sign(tmp_path, run_id="critic")
    revise = {
        "candidate_id": "c00",
        "decision": "REVISE",
        "reason": "The proposal uses a feature that is not available at serve time.",
    }
    proposal = {"candidates": [_candidate("c00", "draft")]}
    _script(
        tmp_path,
        "critic",
        [
            {"role": "planner", "output": proposal},
            {"role": "critic", "output": {"verdicts": [revise]}},
            {"role": "planner", "output": proposal},
            {"role": "critic", "output": {"verdicts": [revise]}},
        ],
    )
    state = run_brief(brief, root=tmp_path)
    assert state.stage == "ESCALATE"
    assert state.critic_revisions == 2
    assert "critic revision cap" in (tmp_path / "runs" / "critic" / "ESCALATE.md").read_text()
    assert FakeJobs(tmp_path).submit_count() == 0


def test_schema_revision_cap_escalates(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("AMT_JOBS_BACKEND", "fake")
    monkeypatch.setenv("AMT_AGENT_BACKEND", "fake")
    monkeypatch.setenv("AMT_GIT_SHA", "abc123")
    brief = _sign(tmp_path, run_id="schema")
    bad = {"candidates": [{"id": "c00"}]}
    _script(
        tmp_path,
        "schema",
        [
            {"role": "planner", "output": bad},
            {"role": "planner", "output": bad},
        ],
    )
    state = run_brief(brief, root=tmp_path)
    assert state.stage == "ESCALATE"
    assert state.revisions == 2
    assert "counts as 1 revision" in (tmp_path / "runs" / "schema" / "ESCALATE.md").read_text()
    assert FakeJobs(tmp_path).submit_count() == 0


def test_budget_refusal_does_not_submit(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("AMT_JOBS_BACKEND", "fake")
    monkeypatch.setenv("AMT_AGENT_BACKEND", "fake")
    monkeypatch.setenv("AMT_GIT_SHA", "abc123")
    brief = _sign(tmp_path, total_dbu=0.05, run_id="budget")
    _script(tmp_path, "budget", _round("c00", "draft"))
    state = run_brief(brief, root=tmp_path)
    assert state.stage == "ESCALATE"
    assert "refusing submit" in (tmp_path / "runs" / "budget" / "ESCALATE.md").read_text()
    assert FakeJobs(tmp_path).submit_count() == 0


def test_every_stage_pauses_until_approve(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("AMT_JOBS_BACKEND", "fake")
    monkeypatch.setenv("AMT_AGENT_BACKEND", "fake")
    monkeypatch.setenv("AMT_GIT_SHA", "abc123")
    brief = _sign(tmp_path, run_id="gate", approval_mode="every_stage")
    _script(
        tmp_path,
        "gate",
        [
            {"role": "planner", "output": {"candidates": [_candidate("c00", "draft")]}},
            {"role": "critic", "output": {"verdicts": [_pass("c00")]}},
        ],
    )
    first = run_brief(brief, root=tmp_path)
    assert first.stage == "CRITIQUE"
    assert first.awaiting_approval is True
    assert first.approval_gate == "PLAN"
    second = run_brief(brief, root=tmp_path)
    assert second.stage == "CRITIQUE"
    assert second.awaiting_approval is True
    approve = tmp_path / "runs" / "gate" / "APPROVE"
    approve.write_text("approve\n", encoding="utf-8")
    third = run_brief(brief, root=tmp_path)
    assert third.stage == "SPEC"
    assert third.awaiting_approval is True
    assert third.approval_gate == "CRITIQUE"
    assert not (tmp_path / "candidates" / "c00" / "model.py").exists()


def test_round_shortlist_pauses_before_compute(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("AMT_JOBS_BACKEND", "fake")
    monkeypatch.setenv("AMT_AGENT_BACKEND", "fake")
    monkeypatch.setenv("AMT_GIT_SHA", "abc123")
    brief = _sign(tmp_path, run_id="short", approval_mode="round_shortlist")
    _script(tmp_path, "short", _round("c00", "draft"))
    state = run_brief(brief, root=tmp_path)
    assert state.stage == "SPEC"
    assert state.awaiting_approval is True
    assert state.approval_gate == "CRITIQUE"
    assert not (tmp_path / "candidates" / "c00" / "model.py").exists()
    assert FakeJobs(tmp_path).submit_count() == 0


def test_token_cap_and_call_log(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("AMT_AGENT_BACKEND", "fake")
    monkeypatch.setenv("AMT_TOKEN_CAP", "10")
    monkeypatch.setenv("AMT_MODEL_PLANNER", "planner-test")
    brief = Brief.model_validate(
        {"name": "toy", "run_id": "log", "budget": {"total_dbu": 1}, "candidate": {"id": "c00"}}
    )
    _script(
        tmp_path,
        "log",
        [
            {
                "role": "planner",
                "output": {"candidates": [_candidate("c00", "draft")]},
                "usage": {"total_tokens": 50},
            }
        ],
    )
    try:
        run_role("planner", brief, tmp_path, "log")
    except RuntimeError as exc:
        assert "token cap" in str(exc)
    else:
        raise AssertionError("token cap was not enforced")
    record = json.loads(
        (tmp_path / "runs" / "log" / "agent_calls.jsonl").read_text(encoding="utf-8")
    )
    assert record["ok"] is False
    assert record["model"] == "planner-test"
    assert record["token_cap"] == 10
    assert record["timeout_s"] == 600


def test_approve_command_writes_the_file(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    assert main(["approve", "--run-id", "search"]) == 0
    assert (tmp_path / "runs" / "search" / "APPROVE").is_file()
