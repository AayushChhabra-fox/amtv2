"""Dataset preflight checks and the scoped feature-SQL repair loop."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from amt.batch import fetch_job_failure
from amt.cli import main
from amt.dataset_preflight import run_local_checks
from amt.dataset_repair import (
    DatasetRepairError,
    apply_fixer_output,
    classify_failure,
    feature_columns,
    record_attempt,
    repair_features,
    revert_spec,
    validate_repair,
)
from amt.schemas import Brief
from amt.volume import VolumeError
from harness.data.contract import DatasetBuildError, load_spec
from harness.data.preflight import explain_issues, search_val_day_count

ROOT = Path(__file__).resolve().parents[1]


def _brief(**extra) -> Brief:
    payload = {
        "name": "nfy_demo",
        "dataset_spec": "datasets/nfy_article_rerank_demo",
        "budget": {"total_dbu": 10},
        "problem": {
            "metric": {
                "name": "ndcg@10_graded",
                "gains": {"impressed": 0, "click": 1, "engaged_read": 2, "deep_read": 3},
            }
        },
    }
    payload.update(extra)
    return Brief.model_validate(payload)


def test_search_val_needs_five_days_unless_the_spec_overrides() -> None:
    article = load_spec(ROOT / "datasets" / "nfy_article_rerank")
    demo = load_spec(ROOT / "datasets" / "nfy_article_rerank_demo")
    short = load_spec(ROOT / "datasets" / "nfy_article_rerank_v2")
    assert search_val_day_count(article) >= 5
    assert search_val_day_count(demo) == 1
    assert demo.preflight.min_search_val_days == 1
    run_local_checks(demo, _brief())
    run_local_checks(article, _brief(name="nfy", dataset_spec="datasets/nfy_article_rerank"))
    with pytest.raises(DatasetBuildError) as failure:
        run_local_checks(short)
    assert failure.value.kind == "split"
    assert "5" in str(failure.value)


def test_gains_must_match_the_brief_metric() -> None:
    demo = load_spec(ROOT / "datasets" / "nfy_article_rerank_demo")
    grades = run_local_checks(demo, _brief())["positive_grades"]
    assert grades == [1, 2, 3]
    mismatched = _brief(
        problem={
            "metric": {
                "name": "ndcg@10_click",
                "gains": {"click": 1},
            }
        }
    )
    with pytest.raises(DatasetBuildError, match="primary_metric"):
        run_local_checks(demo, mismatched)
    empty = _brief(problem={"metric": {"name": "ndcg@10_graded", "gains": {"impressed": 0}}})
    with pytest.raises(DatasetBuildError) as failure:
        run_local_checks(demo, empty)
    assert failure.value.kind == "label_support"


def test_explain_flags_cartesian_range_and_unpartitioned_scans() -> None:
    spec = load_spec(ROOT / "datasets" / "nfy_article_rerank_demo")
    table = spec.sources["requests"].table
    assert explain_issues("== Physical Plan ==\nProject\n", spec) == []
    cartesian = explain_issues("+- CartesianProduct\n", spec)
    assert cartesian[0]["kind"] == "cartesian"
    ranged = explain_issues("BroadcastNestedLoopJoin BuildRight", spec)
    assert ranged[0]["kind"] == "unbounded_range_join"
    scan = explain_issues(
        f"FileScan parquet {table}[id]\nPartitionFilters: []\n",
        spec,
    )
    assert scan[0]["kind"] == "unpartitioned_scan"
    assert scan[0]["source"] == "requests"
    filtered = explain_issues(
        f"FileScan parquet {table}\nPartitionFilters: [event_date]\n",
        spec,
    )
    assert filtered == []


def test_failure_classes_split_repairable_sql_from_escalations() -> None:
    feature = {
        "kind": "",
        "fragment": "profile",
        "fragment_path": "features/profile.sql",
        "exception_type": "Py4JJavaError",
        "message": "AnalysisException: cannot resolve `cold_start`",
        "traceback": "",
    }
    assert classify_failure(feature) == "repairable"
    assert (
        classify_failure(
            {
                "kind": "contract",
                "fragment": "article",
                "fragment_path": "features/article.sql",
                "exception_type": "DatasetBuildError",
                "message": "duplicate keys",
                "traceback": "",
            }
        )
        == "repairable"
    )
    assert (
        classify_failure(
            {
                "kind": "point_in_time",
                "fragment": "profile",
                "fragment_path": "features/profile.sql",
                "message": "available after query_ts",
            }
        )
        == "escalate"
    )
    assert (
        classify_failure(
            {
                "kind": "",
                "fragment": "queries",
                "fragment_path": "queries.sql",
                "exception_type": "AnalysisException",
                "message": "cannot resolve",
            }
        )
        == "escalate"
    )
    assert (
        classify_failure(
            {
                "kind": "label_support",
                "fragment": "labels",
                "fragment_path": "labels.sql",
                "message": "grade 2 has no rows",
            }
        )
        == "escalate"
    )
    assert (
        classify_failure(
            {
                "kind": "",
                "fragment": "",
                "fragment_path": "",
                "exception_type": "TypeError",
                "message": "boom",
                "traceback": 'File "harness/data/build_dataset.py", line 1, in build\nTypeError',
            }
        )
        == "escalate"
    )


def test_validate_repair_keeps_columns_and_rejects_new_sources(tmp_path: Path) -> None:
    source = ROOT / "datasets" / "nfy_article_rerank_demo"
    dest = tmp_path / "demo"
    dest.mkdir()
    for path in source.rglob("*"):
        if path.is_file():
            target = dest / path.relative_to(source)
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(path.read_bytes())
    spec = load_spec(dest)
    original = (dest / "dataset.yaml").read_bytes()
    columns = feature_columns(spec)
    assert validate_repair(dest, original, columns).name == spec.name
    context = dest / "features" / "context.sql"
    broken = context.read_text(encoding="utf-8").replace(
        "FROM {requests}",
        "FROM secret.warehouse.extra",
    )
    context.write_text(broken, encoding="utf-8")
    with pytest.raises(DatasetRepairError, match="undeclared source"):
        validate_repair(dest, original, columns)
    context.write_text(
        (source / "features" / "context.sql")
        .read_text(encoding="utf-8")
        .replace("AS hour_of_day", "AS hour_of_day_renamed"),
        encoding="utf-8",
    )
    with pytest.raises(DatasetRepairError, match="feat__ column set"):
        validate_repair(dest, original, columns)


def test_repair_loop_records_a_diff_and_revert_restores_it(tmp_path: Path, monkeypatch) -> None:
    spec_dir = tmp_path / "datasets" / "toy_spec"
    features = spec_dir / "features"
    features.mkdir(parents=True)
    (spec_dir / "queries.sql").write_text("SELECT 1 AS query_id\n", encoding="utf-8")
    (spec_dir / "candidates.sql").write_text(
        "SELECT 1 AS query_id, 1 AS item_id\n", encoding="utf-8"
    )
    (spec_dir / "labels.sql").write_text(
        "SELECT 1 AS query_id, 1 AS item_id, 1 AS label\n", encoding="utf-8"
    )
    (features / "signals.sql").write_text(
        "SELECT 1 AS query_id, 1 AS item_id, 1.0 AS signal_value, "
        "TIMESTAMP '2026-01-01 00:00:00' AS available_at\n",
        encoding="utf-8",
    )
    (spec_dir / "dataset.yaml").write_text(
        """
name: toy_spec
task_type: ranking
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
""".strip()
        + "\n",
        encoding="utf-8",
    )
    spec = load_spec(spec_dir)
    calls = {"n": 0}

    def fake_preflight(root, spec_path, brief, *, remote=True, slice_window="1h"):
        del root, spec_path, brief, remote, slice_window
        calls["n"] += 1
        if calls["n"] == 1:
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
        return {"status": "pass", "search_val_days": 5, "positive_grades": []}

    def fake_role(*args, **kwargs):
        del args, kwargs
        sql = (features / "signals.sql").read_text(encoding="utf-8")
        sql = sql.replace("1.0", "CAST(1 AS DOUBLE)")
        return {"edits": [{"path": "features/signals.sql", "sql": sql}]}

    monkeypatch.setattr("amt.dataset_repair.run_preflight", fake_preflight)
    monkeypatch.setattr("amt.dataset_repair.run_role", fake_role)
    monkeypatch.setattr("amt.dataset_repair.deploy", lambda root: root)
    monkeypatch.setenv("AMT_DATASET_REPAIR_MAX_ATTEMPTS", "3")
    monkeypatch.setenv("AMT_DATASET_REPAIR_MAX_DBU", "10")

    outcome = repair_features(tmp_path, "datasets/toy_spec", remote=False)
    assert outcome["status"] == "pass"
    assert outcome["fixes"] == 1
    assert "CAST(1 AS DOUBLE)" in (features / "signals.sql").read_text(encoding="utf-8")
    diff = (tmp_path / "runs" / "dataset-repair" / "toy_spec" / "1" / "diff.patch").read_text(
        encoding="utf-8"
    )
    assert "CAST(1 AS DOUBLE)" in diff
    restored = revert_spec(tmp_path, "toy_spec")
    assert restored.name == "1"
    assert "CAST(1 AS DOUBLE)" not in (features / "signals.sql").read_text(encoding="utf-8")
    assert spec.feature_groups["signals"].sql == "features/signals.sql"


def test_repair_escalates_a_label_support_failure(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(
        "amt.dataset_repair.run_preflight",
        lambda *args, **kwargs: {
            "status": "fail",
            "failure": {
                "kind": "label_support",
                "fragment": "labels",
                "fragment_path": "labels.sql",
                "message": "grade 2 has no rows",
            },
        },
    )
    with pytest.raises(DatasetRepairError) as failure:
        repair_features(ROOT, "datasets/nfy_article_rerank_demo", _brief(), remote=False)
    assert failure.value.action == "escalate"
    assert failure.value.failure["kind"] == "label_support"


def test_apply_refuses_paths_outside_features(tmp_path: Path) -> None:
    spec = load_spec(ROOT / "datasets" / "nfy_article_rerank_demo")
    with pytest.raises(DatasetRepairError, match="refusing"):
        apply_fixer_output(spec, {"edits": [{"path": "queries.sql", "sql": "SELECT 1"}]})
    with pytest.raises(DatasetRepairError, match="refusing"):
        apply_fixer_output(
            spec,
            {"edits": [{"path": "features/nested/x.sql", "sql": "SELECT 1"}]},
        )
    record_attempt(
        tmp_path,
        "demo",
        1,
        {"features/a.sql": "before\n"},
        {"features/a.sql": "after\n"},
        {"message": "cast"},
    )
    assert (tmp_path / "runs" / "dataset-repair" / "demo" / "1" / "before" / "a.sql").is_file()


def test_fetch_job_failure_reads_the_file_then_get_output(tmp_path: Path, monkeypatch) -> None:
    from harness.data.build_dataset import failure_dir

    monkeypatch.setenv("AMT_CATALOG", "cat")
    monkeypatch.setenv("AMT_SCHEMA", "sch")
    monkeypatch.setenv("AMT_DATASET_FAILURE_ROOT", str(tmp_path / "volume"))
    spec = load_spec(ROOT / "datasets" / "nfy_article_rerank_demo")
    payload = {
        "exception_type": "AnalysisException",
        "message": "cannot cast",
        "kind": "explain",
        "fragment": "profile",
        "fragment_path": "features/profile.sql",
    }
    path = failure_dir("cat", "sch", spec.name, spec.spec_hash())
    path.mkdir(parents=True)
    (path / "failure.json").write_text(json.dumps(payload), encoding="utf-8")
    record = tmp_path / "runs" / "jobs" / "run-1.json"
    record.parent.mkdir(parents=True)
    record.write_text(
        json.dumps(
            {
                "spec": "datasets/nfy_article_rerank_demo",
                "spec_name": spec.name,
                "spec_hash": spec.spec_hash(),
                "catalog": "cat",
                "schema": "sch",
            }
        ),
        encoding="utf-8",
    )
    assert fetch_job_failure(tmp_path, "run-1")["fragment"] == "profile"

    monkeypatch.delenv("AMT_DATASET_FAILURE_ROOT")
    seen: dict[str, list[str]] = {}

    def fake_get(*_args, **_kwargs):
        raise VolumeError("missing")

    def fake_run_json(args, *, cwd):
        del cwd
        joined = " ".join(str(arg) for arg in args)
        seen.setdefault("calls", []).append(joined)
        if "runs/get?" in joined and "get-output" not in joined:
            return {"tasks": [{"run_id": "task-9", "task_key": "build_dataset"}]}
        return {"error": "AnalysisException: cannot cast", "error_trace": "trace"}

    monkeypatch.setattr("amt.volume.get", fake_get)
    monkeypatch.setattr("amt.jobs_backend._run_json", fake_run_json)
    monkeypatch.setenv("AMT_JOBS_BACKEND", "databricks")
    fallback = fetch_job_failure(tmp_path, "run-1")
    assert fallback["source"] == "jobs/runs/get-output"
    assert "cannot cast" in fallback["message"]
    assert fallback["traceback"] == "trace"
    assert any("get-output?run_id=task-9" in call for call in seen["calls"])


def test_local_preflight_cli_accepts_the_demo(capsys: pytest.CaptureFixture[str]) -> None:
    code = main(
        [
            "dataset",
            "preflight",
            "--spec",
            "datasets/nfy_article_rerank_demo",
            "--brief",
            "briefs/nfy_demo.yaml",
            "--local-only",
            "--root",
            str(ROOT),
        ]
    )
    assert code == 0
    assert json.loads(capsys.readouterr().out)["status"] == "pass"
    short = main(
        [
            "dataset",
            "preflight",
            "--spec",
            "datasets/nfy_article_rerank_v2",
            "--local-only",
            "--root",
            str(ROOT),
        ]
    )
    assert short == 1
