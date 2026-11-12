"""G3 calibration, the K=4 unattended run, and Phase 2 entry."""

import json
from datetime import datetime, timezone
from pathlib import Path

import pytest

from amt.cli import main
from amt.jobs_backend import FakeJobs
from amt.ledger import append
from amt.orchestrator import _plan_note, _proposal_problems, run_brief
from amt.scale import (
    collect_agent_stats,
    evaluate_small_run,
    g3_is_current,
    scale_block_reason,
    sign_g3,
)
from amt.schemas import Brief, LedgerRow, ProposalBatch, RunState
from amt.status import audit_rows
from tests.test_search import ROOT, _candidate, _pass, _script, _sign

SIGNED = datetime(2026, 11, 12, 16, 0, tzinfo=timezone.utc)


def _prepare(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setenv("AMT_JOBS_BACKEND", "fake")
    monkeypatch.setenv("AMT_AGENT_BACKEND", "fake")
    monkeypatch.setenv("AMT_GIT_SHA", "abc123")
    brief = _sign(tmp_path, total_dbu=40, run_id="scale", k=4)
    dest = tmp_path / ".cursor" / "agents"
    dest.mkdir(parents=True)
    source = ROOT / ".cursor" / "agents"
    for name in ("planner.md", "critic.md", "implementer.md", "reviewer.md"):
        (dest / name).write_text((source / name).read_text(encoding="utf-8"), encoding="utf-8")
    points = {
        "c00": 0.02,
        "a00": 0.08,
        "a01": 0.07,
        "a02": 0.06,
        "a03": 0.05,
        "a10": 0.09,
        "a11": 0.04,
        "a12": 0.03,
        "a20": 0.10,
        "a21": 0.04,
        "a22": 0.03,
        "a30": 0.11,
    }
    table: dict[str, dict[str, float]] = {}
    for node_id, point in points.items():
        table[node_id] = {
            "ndcg@10_graded": 0.45,
            "delta_point": point,
            "delta_ci_lo": 0.011,
            "delta_ci_hi": 0.03,
        }
    table["*"] = dict(table["c00"])
    metrics = tmp_path / ".amt" / "fake_metrics.json"
    metrics.write_text(json.dumps(table), encoding="utf-8")
    return brief


def _drive(brief: Path, root: Path, run_id: str, **kwargs: str) -> RunState:
    state: RunState | None = None
    for _ in range(60):
        state = run_brief(brief, root=root, run_id=run_id, **kwargs)  # type: ignore[arg-type]
        if state.stage in {"REPORT", "ESCALATE"} and not state.awaiting_approval:
            return state
        if state.awaiting_approval:
            approve = root / "runs" / state.run_id / "APPROVE"
            approve.parent.mkdir(parents=True, exist_ok=True)
            approve.write_text("approve\n", encoding="utf-8")
    raise AssertionError(f"stopped in {None if state is None else state.stage}")


def _state(**overrides: object) -> RunState:
    payload: dict[str, object] = {
        "run_id": "s",
        "brief_path": "briefs/nfy.yaml",
        "candidate_id": "pending",
        "family": "search",
        "rung": "R1",
        "seed": 0,
        "batch": "nfy",
        "k": 4,
        "scale": "small",
        "round": 1,
        "round_operator": "draft",
    }
    payload.update(overrides)
    return RunState.model_validate(payload)


def _row(run_id: str, node_id: str, family: str, **overrides: object) -> LedgerRow:
    payload: dict[str, object] = {
        "run_id": run_id,
        "node_id": node_id,
        "operator": "draft",
        "family": family,
        "claim": node_id,
        "rung": "R1",
        "seed": 0,
        "status": "kept",
        "git_sha": "abc123",
        "job_run_id": f"fake-{node_id}",
        "mlflow_run_id": f"m-{node_id}",
        "metric": 0.4,
        "delta_vs_baseline": {"point": 0.02, "ci_lo": 0.011, "ci_hi": 0.03},
    }
    payload.update(overrides)
    return LedgerRow.model_validate(payload)


def test_calibration_is_one_lineage_and_is_not_repeated(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    brief = _prepare(tmp_path, monkeypatch)
    refused = run_brief(
        brief, root=tmp_path, run_id="bad", scale="calibration", approval_mode="none"
    )
    assert refused.stage == "ESCALATE"
    assert "every_stage" in (tmp_path / "runs" / "bad" / "ESCALATE.md").read_text(encoding="utf-8")
    assert not (tmp_path / "gates" / "g3" / "started.json").is_file()

    early = run_brief(brief, root=tmp_path, run_id="early", scale="small")
    assert early.stage == "ESCALATE"
    assert "G3" in (tmp_path / "runs" / "early" / "ESCALATE.md").read_text(encoding="utf-8")
    assert FakeJobs(tmp_path).submit_count() == 0

    _script(
        tmp_path,
        "cal",
        [
            {
                "role": "planner",
                "output": {"candidates": [_candidate("c00", "draft", claim="calibration draft")]},
                "usage": {"total_tokens": 11},
            },
            {"role": "critic", "output": {"verdicts": [_pass("c00")]}},
            {"role": "implementer", "node_id": "c00"},
            {"role": "reviewer", "node_id": "c00"},
        ],
    )
    state = _drive(brief, tmp_path, "cal", scale="calibration")
    assert state.stage == "REPORT"
    assert state.scale == "calibration"
    assert state.k == 1
    assert state.approval_mode == "every_stage"
    assert state.round == 1
    assert state.nodes[0].node_id == "c00"
    assert all(node.operator == "draft" for node in state.nodes)
    assert FakeJobs(tmp_path).submit_count() == 3
    assert not any(key.startswith("confirm/") for key in state.submitted_jobs)
    assert audit_rows(tmp_path, "cal") == []
    packet = tmp_path / "gates" / "g3" / "nfy_article_rerank" / "calibration_report.md"
    text = packet.read_text(encoding="utf-8")
    assert "Status: pass" in text
    assert "planner" in text
    assert "c00: PASS" in text
    assert (tmp_path / "candidates" / "c00" / "spec.md").is_file()
    assert (tmp_path / "candidates" / "c00" / "review.md").is_file()
    stats = json.loads((tmp_path / "runs" / "cal" / "agent_stats.json").read_text(encoding="utf-8"))
    assert stats["tokens"] == 11
    assert stats["bug_rate"] == 0
    assert stats["revisions"] == 0
    assert any(node["node_id"] == "c00" and node["wall_clock_s"] > 0 for node in stats["nodes"])

    record = sign_g3(tmp_path, brief, "Ada Lovelace", "cal", signed_at=SIGNED)
    assert record.gate == "G3"
    assert record.k == 1
    assert g3_is_current(tmp_path)
    assert (
        main(
            [
                "sign",
                "g3",
                "--root",
                str(tmp_path),
                "--brief",
                str(brief),
                "--run-id",
                "cal",
                "--signer",
                "Grace Hopper",
            ]
        )
        == 0
    )
    code = main(["gates", "--root", str(tmp_path), "--brief", str(brief)])
    assert code == 0
    again = run_brief(brief, root=tmp_path, run_id="cal2", scale="calibration")
    assert again.stage == "ESCALATE"
    message = (tmp_path / "runs" / "cal2" / "ESCALATE.md").read_text(encoding="utf-8")
    assert "not repeated" in message


def _improve(parent: str, node_ids: list[str]) -> list[dict]:
    return _improve_many([(parent, node_ids)])


def _improve_many(groups: list[tuple[str, list[str]]]) -> list[dict]:
    candidates = []
    node_ids: list[str] = []
    kept: list[str] = []
    for parent, children in groups:
        node_ids.extend(children)
        kept.extend(children[:2])
        candidates.extend(
            _candidate(
                node_id,
                "improve",
                parent=parent,
                change=f"change {node_id}",
                claim=f"improve {node_id}",
            )
            for node_id in children
        )
    calls: list[dict] = [
        {
            "role": "planner",
            "output": {"candidates": candidates},
            "usage": {"total_tokens": 20},
        },
        {"role": "critic", "output": {"verdicts": [_pass(node_id) for node_id in node_ids]}},
    ]
    calls.extend({"role": "implementer", "node_id": node_id} for node_id in kept)
    calls.extend({"role": "reviewer", "node_id": node_id} for node_id in kept)
    return calls


def _small_script(tmp_path: Path) -> None:
    kept = [("a00", "linear"), ("a01", "trees"), ("a02", "neural"), ("a03", "embedding")]
    dropped = [("a04", "text"), ("a05", "bandit"), ("a06", "wide"), ("a07", "mix")]
    candidates = []
    for node_id, family in kept + dropped:
        item = _candidate(node_id, "draft", claim=f"small draft {node_id}")
        item["family"] = family
        candidates.append(item)
    verdicts = [_pass(node_id) for node_id, _family in kept]
    verdicts.extend(
        {
            "candidate_id": node_id,
            "decision": "DROP",
            "reason": "Outside the K=4 shortlist.",
        }
        for node_id, _family in dropped
    )
    calls: list[dict] = [
        {"role": "planner", "output": {"candidates": candidates}, "usage": {"total_tokens": 20}},
        {"role": "critic", "output": {"verdicts": verdicts}},
    ]
    calls.extend({"role": "implementer", "node_id": node_id} for node_id, _family in kept)
    calls.extend({"role": "reviewer", "node_id": node_id} for node_id, _family in kept)
    # R1 no longer halves, so R3 still has the top two lineages. Each needs 3 proposals.
    calls.extend(
        _improve_many([("a00", ["a10", "a11", "a12"]), ("a01", ["a13", "a14", "a15"])])
    )
    calls.extend(_improve("a10", ["a20", "a21", "a22"]))
    calls.extend(
        [
            {
                "role": "planner",
                "output": {
                    "candidates": [
                        _candidate(
                            "a30",
                            "combine",
                            parent="a20",
                            claim="blend a20 with a10",
                        )
                    ]
                },
                "usage": {"total_tokens": 20},
            },
            {"role": "critic", "output": {"verdicts": [_pass("a30")]}},
            {"role": "implementer", "node_id": "a30"},
            {"role": "reviewer", "node_id": "a30"},
        ]
    )
    _script(tmp_path, "small", calls)


def test_k4_unattended_run_records_stats_and_opens_phase2(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    brief = _prepare(tmp_path, monkeypatch)
    _script(
        tmp_path,
        "cal",
        [
            {
                "role": "planner",
                "output": {"candidates": [_candidate("c00", "draft", claim="calibration draft")]},
                "usage": {"total_tokens": 11},
            },
            {"role": "critic", "output": {"verdicts": [_pass("c00")]}},
            {"role": "implementer", "node_id": "c00"},
            {"role": "reviewer", "node_id": "c00"},
        ],
    )
    calibrated = _drive(brief, tmp_path, "cal", scale="calibration")
    assert calibrated.stage == "REPORT"
    sign_g3(tmp_path, brief, "Ada Lovelace", "cal", signed_at=SIGNED)

    _small_script(tmp_path)
    before = FakeJobs(tmp_path).submit_count()
    state = run_brief(brief, root=tmp_path, run_id="small", scale="small")
    assert state.stage == "REPORT", (
        (tmp_path / "runs" / "small" / "ESCALATE.md").read_text(encoding="utf-8")
        if (tmp_path / "runs" / "small" / "ESCALATE.md").is_file()
        else state.stage
    )
    assert state.k == 4
    assert state.approval_mode == "none"
    assert state.spent_dbu <= state.budget_dbu
    families = {node.family for node in state.nodes}
    assert len(families) >= 3
    for number in range(1, 5):
        digest = (tmp_path / "runs" / "small" / f"round_{number}.md").read_text(encoding="utf-8")
        assert "Kills" in digest
        assert "Agent statistics" in digest
    report = (tmp_path / "runs" / "small" / "report.md").read_text(encoding="utf-8")
    assert "bug rate" in report
    assert "wall-clock" in report
    assert "Small-run check: pass" in report
    assert "a00" in report
    stats = json.loads(
        (tmp_path / "runs" / "small" / "agent_stats.json").read_text(encoding="utf-8")
    )
    assert stats["tokens"] == 80
    assert stats["bug_rate"] == 0
    assert stats["revisions"] == 0
    assert any(node["wall_clock_s"] > 0 for node in stats["nodes"])
    rows = [
        LedgerRow.model_validate(json.loads(line))
        for line in (tmp_path / "ledger" / "results.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    small_rows = [row for row in rows if row.run_id == "small"]
    killed = [row for row in small_rows if row.status in {"killed", "failed"}]
    assert killed
    assert all(row.kill_reason for row in killed)
    assert len({row.family for row in small_rows}) >= 3

    opened = run_brief(brief, root=tmp_path, run_id="full-entry", scale="full")
    assert opened.stage == "REPORT"
    entry = (tmp_path / "runs" / "full-entry" / "phase2_entry.md").read_text(encoding="utf-8")
    assert "does not start training" in entry
    assert "bug rate" in entry
    assert FakeJobs(tmp_path).submit_count() == before + 24
    assert main(["gates", "--root", str(tmp_path), "--brief", str(brief)]) == 0
    assert "full budget ready" in capsys.readouterr().out
    assert main(["stats", "--root", str(tmp_path), "--run-id", "small"]) == 0


def test_full_budget_waits_for_the_small_run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    brief = _prepare(tmp_path, monkeypatch)
    _script(
        tmp_path,
        "cal",
        [
            {
                "role": "planner",
                "output": {"candidates": [_candidate("c00", "draft", claim="calibration draft")]},
                "usage": {"total_tokens": 11},
            },
            {"role": "critic", "output": {"verdicts": [_pass("c00")]}},
            {"role": "implementer", "node_id": "c00"},
            {"role": "reviewer", "node_id": "c00"},
        ],
    )
    assert _drive(brief, tmp_path, "cal", scale="calibration").stage == "REPORT"
    sign_g3(tmp_path, brief, "Ada Lovelace", "cal", signed_at=SIGNED)
    before = FakeJobs(tmp_path).submit_count()
    state = run_brief(brief, root=tmp_path, run_id="full", scale="full")
    assert state.stage == "ESCALATE"
    assert "small-budget" in (tmp_path / "runs" / "full" / "ESCALATE.md").read_text(
        encoding="utf-8"
    )
    assert FakeJobs(tmp_path).submit_count() == before
    assert main(["gates", "--root", str(tmp_path), "--brief", str(brief)]) == 0
    assert "full budget not ready" in capsys.readouterr().out


def test_agent_stats_count_bugs_revisions_tokens_and_wall_clock(tmp_path: Path) -> None:
    state = _state(
        run_id="stats",
        revisions=1,
        critic_revisions=1,
        nodes=[
            {
                "node_id": "n1",
                "family": "linear",
                "debug_attempts": 1,
            }
        ],
    )
    path = tmp_path / "runs" / "stats" / "agent_calls.jsonl"
    path.parent.mkdir(parents=True)
    rows = [
        {"role": "planner", "ok": True, "latency_s": 0.2, "tokens": 5, "node_id": None},
        {"role": "implementer", "ok": True, "latency_s": 1.5, "tokens": 10, "node_id": "n1"},
        {"role": "implementer", "ok": False, "latency_s": 0.5, "tokens": 4, "node_id": "n1"},
    ]
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    stats = collect_agent_stats(tmp_path, state)
    assert stats.bug_rate == pytest.approx(0.5)
    assert stats.revisions == 2
    assert stats.tokens == 19
    assert stats.wall_clock_s == pytest.approx(2.2)
    assert stats.nodes[0].node_id == "n1"
    assert stats.nodes[0].tokens == 14
    assert stats.nodes[0].wall_clock_s == pytest.approx(2.0)
    assert stats.nodes[0].debug_attempts == 1


def test_small_run_pass_rule_requires_three_families(tmp_path: Path) -> None:
    brief = Brief.model_validate(
        {"name": "nfy", "run_id": "s", "k": 4, "budget": {"total_dbu": 40}}
    )
    state = _state(run_id="s", brief_path="briefs/nfy.yaml", spent_dbu=1.0, budget_dbu=40)
    state.stage = "REPORT"
    for node_id, family in (("n1", "linear"), ("n2", "trees")):
        append(tmp_path, _row("s", node_id, family))
    for number in range(1, 5):
        path = tmp_path / "runs" / "s" / f"round_{number}.md"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(f"# Round {number}\n", encoding="utf-8")
    (tmp_path / "runs" / "s" / "agent_stats.json").write_text("{}\n", encoding="utf-8")
    receipt = evaluate_small_run(tmp_path, state, brief, collect_agent_stats(tmp_path, state))
    assert receipt.status == "fail"
    assert any("families" in reason for reason in receipt.reasons)
    append(tmp_path, _row("s", "n3", "neural"))
    receipt = evaluate_small_run(tmp_path, state, brief, collect_agent_stats(tmp_path, state))
    assert receipt.status == "pass"
    assert receipt.family_count == 3


def test_k4_plan_asks_for_eight_drafts_and_three_improvements() -> None:
    state = _state()
    note = _plan_note(state, "")
    assert "8 drafts" in note
    batch = ProposalBatch.model_validate({"candidates": [_candidate("a00", "draft")]})
    problems = _proposal_problems(state, batch)
    assert any("8 drafts" in problem for problem in problems)
    state.round = 2
    state.round_operator = "improve"
    state.nodes = [
        {
            "node_id": "a00",
            "family": "linear",
            "status": "kept",
            "search_round": 1,
            "delta_point": 0.08,
        }
    ]
    improve_note = _plan_note(state, "")
    assert "3 atomic improvements" in improve_note
    assert scale_block_reason(Path("."), scale="small", run_id="s", k=1, approval_mode="none")
    blocked = scale_block_reason(Path("."), scale="small", run_id="s", k=4, approval_mode="none")
    assert blocked is not None and "G3" in blocked
