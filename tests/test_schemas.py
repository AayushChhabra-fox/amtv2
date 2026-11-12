import json
from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

from amt.jobs_backend import FakeJobs
from amt.orchestrator import apply_agent_output, run_brief
from amt.schemas import (
    MLFLOW_TAG_KEYS,
    SCHEMA_MODELS,
    Brief,
    Candidate,
    LedgerRow,
    RunState,
    Verdict,
    json_schema_dir,
    load_brief,
    mlflow_tags,
    parse_agent_output,
)

ROOT = Path(__file__).resolve().parents[1]

CANDIDATE = {
    "id": "c00",
    "operator": "draft",
    "kind": "model",
    "family": "linear",
    "claim": "logistic regression separates the toy classes",
    "fit_argument": "The classes are linearly separable.",
    "cheap_test": "Smoke loss falls on the toy frame.",
    "kill_if": "Scores are non-finite.",
    "est_cost": 0.1,
}


def _state() -> RunState:
    return RunState(
        run_id="toy",
        brief_path="briefs/toy.yaml",
        stage="PLAN",
        candidate_id="c00",
        family="linear",
        rung="R0",
        seed=0,
        batch="toy",
        budget_dbu=10,
    )


def test_toy_brief_yaml_round_trip() -> None:
    brief = load_brief(ROOT / "briefs" / "toy.yaml")
    assert brief.name == "toy"
    assert brief.approval_mode == "none"
    assert brief.resolved_dataset_spec() == "datasets/toy"
    assert brief.task.forbidden == ["holdout labels"]
    assert brief.budget.total_dbu == 10
    again = Brief.model_validate_json(brief.model_dump_json())
    assert again == brief


def test_sample_brief_round_trip() -> None:
    payload = {
        "problem": {
            "surface": "News For You article feed reranker",
            "user_outcome": "Users read articles, not just click them.",
            "metric": {
                "name": "ndcg@10_graded",
                "gains": {"impressed": 0, "click": 1, "engaged_read": 2, "deep_read": 3},
                "why_it_tracks_outcome": "Graded dwell rewards reads over bounces.",
            },
            "dataset_spec": "datasets/nfy_article_rerank",
            "data": {
                "requests": "pnr_prod.silver.nfy_reco_requests",
                "impressions": "pnr_prod.silver.news_impressions",
                "scale": "~37k queries/day with >=1 click",
            },
        },
        "task": {
            "query_unit": "one nfy_reco_requests row, entity_type = article",
            "candidates": "served candidates impressed on for-you within 30 minutes",
            "include_query_if": "at least one click",
            "serve_time_inputs": ["request context", "user profile version at request"],
            "forbidden": ["item_index", "display position"],
        },
        "reference": {
            "name": "production NFY scorer",
            "trained_reference": "lgbm_lambdarank on prod signals",
        },
        "min_effect": 0.01,
        "efficiency": {
            "train_dbu_per_candidate": 60,
            "wall_clock_per_rung": {"R0": "10m", "R1": "30m", "R2": "90m", "R3": "4h"},
            "serving": {
                "latency_p95_ms_per_100_candidates": 20,
                "cost": "no GPU at serving for MVP",
            },
        },
        "eval": {
            "split": "time-based, windows set in dataset_spec",
            "final_holdout": "opened once, at gate G5",
        },
        "budget": {"total_dbu": 1500, "max_concurrent_jobs": 4},
        "approval_mode": "none",
        "mode": "both",
        "design_space": {
            "families": ["learned reweighting of production signals (refine)"],
            "out_of_scope_for_mvp": ["cross-encoders or LLM rerankers at serving time"],
            "data": "existing sources only",
        },
        "off_limits": ["eval and holdout splits", "metric code", "harness"],
    }
    brief = Brief.model_validate(payload)
    assert brief.resolved_dataset_spec() == "datasets/nfy_article_rerank"
    assert brief.k == 4
    again = Brief.model_validate(brief.model_dump())
    assert again == brief
    assert again.reference is not None
    assert again.reference.model_extra["trained_reference"].startswith("lgbm")


def test_conflicting_dataset_spec_is_rejected() -> None:
    with pytest.raises(ValidationError, match="disagree"):
        Brief.model_validate(
            {
                "dataset_spec": "datasets/a",
                "problem": {"dataset_spec": "datasets/b"},
                "budget": {"total_dbu": 1},
            }
        )


def test_unknown_brief_field_is_rejected() -> None:
    with pytest.raises(ValidationError):
        Brief.model_validate({"name": "toy", "budget": {"total_dbu": 1}, "extra_field": "kept"})


def test_models_round_trip() -> None:
    candidate = Candidate.model_validate(CANDIDATE)
    verdict = Verdict(candidate_id="c00", decision="REVISE", reason="claim is vague")
    row = LedgerRow(
        run_id="toy",
        node_id="c00",
        round=1,
        operator="improve",
        family="linear",
        claim="add one feature",
        rung="R2",
        seed=1,
        metric=0.42,
        status="killed",
        kill_reason="upper bound below 0",
        git_sha="abc",
        parent_id="c-parent",
    )
    state = _state()
    for model in (candidate, verdict, row, state):
        restored = type(model).model_validate_json(model.model_dump_json())
        assert restored == model


def test_yaml_candidate_round_trip() -> None:
    text = yaml.safe_dump({"candidates": [CANDIDATE]}, sort_keys=False)
    parsed = parse_agent_output("planner", text)
    assert parsed.revision is None
    assert parsed.value is not None
    again = parse_agent_output("planner", parsed.value.model_dump())
    assert again.revision is None
    assert again.value == parsed.value


def test_malformed_planner_output_is_one_revision() -> None:
    parsed = parse_agent_output(
        "planner",
        {"candidates": [{"id": "c00", "operator": "tweak"}]},
    )
    assert parsed.value is None
    assert parsed.revision is not None
    assert parsed.revision.revision_count == 1
    assert parsed.revision.schema_name == "proposal_batch"
    assert len(parsed.revision.errors) > 1
    assert parsed.revision.message.count("revision") == 1
    assert "counts as 1 revision" in parsed.revision.message
    assert any("operator" in line for line in parsed.revision.errors)
    assert any("claim" in line for line in parsed.revision.errors)


def test_proposals_alias_and_improve_parent() -> None:
    payload = {"proposals": [{**CANDIDATE, "operator": "improve"}]}
    parsed = parse_agent_output("planner", payload)
    assert parsed.revision is not None
    assert parsed.revision.revision_count == 1
    assert any("parent_id" in line for line in parsed.revision.errors)

    valid = {
        "proposals": [
            {
                **CANDIDATE,
                "id": "c01",
                "operator": "improve",
                "parent_id": "c00",
                "change": "drop one feature",
            }
        ]
    }
    parsed = parse_agent_output("planner", valid)
    assert parsed.revision is None


def test_draft_with_parent_is_one_revision() -> None:
    parsed = parse_agent_output("planner", {"candidates": [{**CANDIDATE, "parent_id": "c-parent"}]})
    assert parsed.revision is not None
    assert parsed.revision.revision_count == 1
    assert any("parent_id" in line for line in parsed.revision.errors)


def test_malformed_critic_output_is_one_revision() -> None:
    parsed = parse_agent_output(
        "critic",
        {"verdicts": [{"candidate_id": "c00", "decision": "MAYBE", "reason": "unsure"}]},
    )
    assert parsed.value is None
    assert parsed.revision is not None
    assert parsed.revision.revision_count == 1
    assert any("PASS" in line or "decision" in line for line in parsed.revision.errors)

    single = parse_agent_output(
        "critic",
        {"candidate_id": "c00", "decision": "PASS", "reason": "matches the task block"},
    )
    assert single.revision is None
    assert single.value is not None
    assert single.value.verdicts[0].decision == "PASS"


def test_unparseable_text_is_one_revision() -> None:
    for payload in ("{", "", "   "):
        parsed = parse_agent_output("critic", payload)
        assert parsed.value is None
        assert parsed.revision is not None
        assert parsed.revision.revision_count == 1
        assert "(root)" in parsed.revision.errors[0]


def test_unknown_role_is_a_programmer_error() -> None:
    with pytest.raises(ValueError, match="no output schema"):
        parse_agent_output("implementer", {})


def test_orchestrator_counts_one_revision(tmp_path: Path) -> None:
    state = _state()
    updated, revision = apply_agent_output(state, "planner", {"candidates": [{"id": "c00"}]})
    assert revision is not None
    assert revision.revision_count == 1
    assert len(revision.errors) > 1
    assert updated.revisions == 1
    assert state.revisions == 0

    kept, clean = apply_agent_output(
        state,
        "critic",
        {"candidate_id": "c00", "decision": "DROP", "reason": "near duplicate"},
    )
    assert clean is None
    assert kept.revisions == 0


def test_run_returns_malformed_output_as_one_revision(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("AMT_JOBS_BACKEND", "fake")
    monkeypatch.setenv("AMT_GIT_SHA", "abc123")

    def bad_planner(role: str, brief: Brief, root: Path, run_id: str) -> dict:
        del brief, root, run_id
        assert role == "planner"
        return {"candidates": [{"id": "c00", "operator": "draft"}]}

    monkeypatch.setattr("amt.orchestrator.run_role", bad_planner)
    path = tmp_path / "briefs" / "toy.yaml"
    path.parent.mkdir(parents=True)
    path.write_text(
        yaml.safe_dump(
            {
                "name": "toy",
                "run_id": "toy",
                "approval_mode": "none",
                "budget": {"total_dbu": 10},
                "candidate": {"id": "c00", "family": "linear", "claim": "logistic"},
            }
        ),
        encoding="utf-8",
    )
    state = run_brief(path, root=tmp_path)
    assert state.stage == "ESCALATE"
    assert state.revisions == 1
    feedback = (tmp_path / "runs" / "toy" / "agent_feedback.md").read_text(encoding="utf-8")
    escalate = (tmp_path / "runs" / "toy" / "ESCALATE.md").read_text(encoding="utf-8")
    assert "counts as 1 revision" in feedback
    assert "claim" in feedback
    assert feedback in escalate
    assert FakeJobs(tmp_path).submit_count() == 0


def test_candidate_schema_lists_plan_fields() -> None:
    schema = Candidate.model_json_schema()
    assert set(schema["properties"]) == {
        "id",
        "parent_id",
        "operator",
        "kind",
        "family",
        "claim",
        "fit_argument",
        "change",
        "cheap_test",
        "kill_if",
        "est_cost",
    }
    assert set(schema["required"]) == {
        "id",
        "operator",
        "family",
        "claim",
        "fit_argument",
        "cheap_test",
        "kill_if",
        "est_cost",
    }
    ledger = set(LedgerRow.model_json_schema()["properties"])
    assert "delta_vs_baseline" in ledger
    assert "mlflow_run_id" in ledger
    assert "job_run_id" in ledger


def test_exported_json_schemas_match_models() -> None:
    directory = json_schema_dir()
    names = set(SCHEMA_MODELS)
    assert {path.name for path in directory.glob("*.schema.json")} == {
        f"{name}.schema.json" for name in names
    }
    for name, model in SCHEMA_MODELS.items():
        committed = json.loads((directory / f"{name}.schema.json").read_text(encoding="utf-8"))
        assert committed == model.model_json_schema()


def test_mlflow_tag_convention() -> None:
    tags = mlflow_tags(
        run_id="toy",
        node_id="c00",
        parent_id=None,
        family="linear",
        operator="draft",
        rung="R0",
        seed=0,
        batch="toy",
        split="search_val",
        git_sha="abc",
    )
    assert list(tags) == list(MLFLOW_TAG_KEYS)
    assert tags["parent_id"] == ""
    assert tags["seed"] == "0"
    with pytest.raises(ValueError, match="empty"):
        mlflow_tags(
            run_id="toy",
            node_id="c00",
            parent_id=None,
            family="linear",
            operator="draft",
            rung="R0",
            seed=0,
            batch="toy",
            split="search_val",
            git_sha="",
        )


def test_hello_train_sets_the_tag_convention() -> None:
    source = (ROOT / "src" / "hello_train.py").read_text(encoding="utf-8")
    for key in MLFLOW_TAG_KEYS:
        assert f'"{key}"' in source
