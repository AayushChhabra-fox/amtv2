"""Spec-driven builder: planted leaks, two surfaces, stable hashes, NFY SQL."""

from __future__ import annotations

import json
import sqlite3
from datetime import date
from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

from harness.data.assemble import (
    assemble_select_sql,
    fragment_selects,
    render_params,
    source_view_sql,
    violation_sqls,
)
from harness.data.build_dataset import build
from harness.data.contract import DatasetBuildError, load_spec
from harness.data.sampling import choose_query_ids, query_times

ROOT = Path(__file__).resolve().parents[1]
WINDOWS = {
    "train": ("2026-01-01", "2026-01-03"),
    "search_val": ("2026-01-05", "2026-01-06"),
    "select_holdout": ("2026-01-08", "2026-01-09"),
    "final_holdout": ("2026-01-11", "2026-01-12"),
}
QUERIES_SQL = """
SELECT query_id, query_ts
FROM {requests}
WHERE kind = '{entity_type}'
  AND substr(query_ts, 1, 10) BETWEEN '{start}' AND '{end}'
""".strip()
CANDIDATES_SQL = """
SELECT query_id, item_id, total_score, item_age_seconds
FROM {candidates}
WHERE kind = '{entity_type}'
  AND substr(query_ts, 1, 10) BETWEEN '{start}' AND '{end}'
""".strip()
CANDIDATES_LEAK_SQL = """
SELECT query_id, item_id, total_score, item_age_seconds, item_index
FROM {candidates}
WHERE kind = '{entity_type}'
  AND substr(query_ts, 1, 10) BETWEEN '{start}' AND '{end}'
""".strip()
LABELS_SQL = """
SELECT query_id, item_id, label, label_event_ts, click
FROM {labels}
WHERE kind = '{entity_type}'
  AND substr(query_ts, 1, 10) BETWEEN '{start}' AND '{end}'
""".strip()
SIGNALS_SQL = """
SELECT query_id, item_id, signal_value, available_at
FROM {signals}
WHERE kind = '{entity_type}'
  AND substr(query_ts, 1, 10) BETWEEN '{start}' AND '{end}'
""".strip()
SPEC_YAML = """
name: {name}
task_type: ranking
entity_type: {entity}
primary_metric: ndcg@10_graded
output:
  catalog: local
  schema: amt
sources:
  requests:
    table: src_requests
    unpartitioned: true
  candidates:
    table: src_candidates
    unpartitioned: true
  labels:
    table: src_labels
    unpartitioned: true
  signals:
    table: src_signals
    unpartitioned: true
queries: queries.sql
candidates: candidates.sql
labels:
  sql: labels.sql
  label_window: 24h
  aux_labels: [click]
feature_groups:
  signals:
    sql: features/signals.sql
    keys: [query_id, item_id]
    available_at: available_at
filters:
  include_query_if: "max(label) >= 1"
forbidden_columns:
  - item_index
  - display_position
  - latest_profile_*
candidate_bounds:
  min: 1
  max: 100
reference_scores:
  production: total_score
  recency: "-item_age_seconds"
split:
  time_col: query_ts
  gap_days: 1
  train: 2026-01-01..2026-01-03
  search_val: 2026-01-05..2026-01-06
  select_holdout: 2026-01-08..2026-01-09
  final_holdout: 2026-01-11..2026-01-12
sampling:
  seed: 0
  smoke:
    train_queries: 2
    eval_queries: 1
  rungs:
    R1:
      train_fraction: 0.50
      eval_queries: 1
    R2:
      train_fraction: 1.00
      eval_queries: all
    R3:
      train_fraction: 1.00
      eval_queries: all
checks:
  - name: queries_per_day
    min: 1
  - name: candidates_per_query_mean
    between: [1, 4]
""".strip()


def _days() -> list[str]:
    days = []
    for start, end in WINDOWS.values():
        cursor = date.fromisoformat(start)
        last = date.fromisoformat(end)
        while cursor <= last:
            days.append(cursor.isoformat())
            cursor = date.fromordinal(cursor.toordinal() + 1)
    return days


def seed(connection: sqlite3.Connection) -> None:
    connection.executescript(
        """
        CREATE TABLE src_requests (
          kind TEXT, query_id TEXT, query_ts TEXT
        );
        CREATE TABLE src_candidates (
          kind TEXT, query_id TEXT, item_id TEXT, query_ts TEXT,
          total_score REAL, item_age_seconds REAL, item_index INT
        );
        CREATE TABLE src_labels (
          kind TEXT, query_id TEXT, item_id TEXT, query_ts TEXT,
          label INT, label_event_ts TEXT, click INT
        );
        CREATE TABLE src_signals (
          kind TEXT, query_id TEXT, item_id TEXT, query_ts TEXT,
          signal_value REAL, available_at TEXT
        );
        """
    )
    kinds = ("article", "clip")
    rows = [(kind, day) for kind in kinds for day in _days()]
    rows.append(("article", "2026-01-04"))
    rows.append(("clip", "2026-01-04"))
    for kind, day in rows:
        query_id = f"{kind}-gap" if day == "2026-01-04" else f"{kind}-{day}"
        query_ts = f"{day} 12:00:00"
        connection.execute(
            "INSERT INTO src_requests VALUES (?, ?, ?)",
            (kind, query_id, query_ts),
        )
        for item_id, label, score in (("a", 1, 1.0), ("b", 0, 0.0)):
            _insert_pair(connection, kind, query_id, query_ts, item_id, label, score)
        if day == "2026-01-01":
            noclick = f"{kind}-noclick"
            connection.execute(
                "INSERT INTO src_requests VALUES (?, ?, ?)",
                (kind, noclick, query_ts),
            )
            for item_id in ("a", "b"):
                _insert_pair(connection, kind, noclick, query_ts, item_id, 0, 0.0)
    connection.commit()


def _insert_pair(connection, kind, query_id, query_ts, item_id, label, score) -> None:
    hour = query_ts.replace("12:00:00", "13:00:00")
    connection.execute(
        "INSERT INTO src_candidates VALUES (?, ?, ?, ?, ?, ?, ?)",
        (kind, query_id, item_id, query_ts, score, 10.0, 0),
    )
    connection.execute(
        "INSERT INTO src_labels VALUES (?, ?, ?, ?, ?, ?, ?)",
        (kind, query_id, item_id, query_ts, label, hour, 1 if label else 0),
    )
    connection.execute(
        "INSERT INTO src_signals VALUES (?, ?, ?, ?, ?, ?)",
        (kind, query_id, item_id, query_ts, 0.5, query_ts),
    )


def write_spec(directory: Path, name: str, entity: str, *, leak_position: bool = False) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "features").mkdir(exist_ok=True)
    (directory / "queries.sql").write_text(QUERIES_SQL)
    candidates = CANDIDATES_LEAK_SQL if leak_position else CANDIDATES_SQL
    (directory / "candidates.sql").write_text(candidates)
    (directory / "labels.sql").write_text(LABELS_SQL)
    (directory / "features" / "signals.sql").write_text(SIGNALS_SQL)
    (directory / "dataset.yaml").write_text(SPEC_YAML.format(name=name, entity=entity))
    return directory


def _ids(path: Path) -> set[str]:
    found = set()
    for line in path.read_text().splitlines():
        if line.strip():
            found.add(json.loads(line)["query_id"])
    return found


def test_planted_leaks(tmp_path: Path) -> None:
    spec_dir = write_spec(tmp_path / "article", "toy_article", "article")
    late = sqlite3.connect(":memory:")
    seed(late)
    late.execute(
        "UPDATE src_signals SET available_at = '2026-01-02 12:00:00' "
        "WHERE query_id = 'article-2026-01-01' AND item_id = 'a'"
    )
    with pytest.raises(DatasetBuildError) as pit:
        build(spec_dir, output_dir=tmp_path / "pit", backend="local", connection=late)
    assert pit.value.kind == "point_in_time"

    window = sqlite3.connect(":memory:")
    seed(window)
    window.execute(
        "UPDATE src_labels SET label_event_ts = '2026-01-04 12:00:00' "
        "WHERE query_id = 'article-2026-01-01'"
    )
    with pytest.raises(DatasetBuildError) as labels:
        build(spec_dir, output_dir=tmp_path / "window", backend="local", connection=window)
    assert labels.value.kind == "label_window"

    leaked = write_spec(tmp_path / "leaked", "toy_article", "article", leak_position=True)
    forbidden = sqlite3.connect(":memory:")
    seed(forbidden)
    with pytest.raises(DatasetBuildError) as columns:
        build(leaked, output_dir=tmp_path / "forbidden", backend="local", connection=forbidden)
    assert columns.value.kind == "forbidden_column"


def test_two_surfaces_hashes_and_holdouts(tmp_path: Path) -> None:
    connection = sqlite3.connect(":memory:")
    seed(connection)
    manifests = {}
    for entity in ("article", "clip"):
        spec_dir = write_spec(tmp_path / entity, f"toy_{entity}", entity)
        first = tmp_path / f"{entity}-a"
        second = tmp_path / f"{entity}-b"
        left = build(spec_dir, output_dir=first, backend="local", connection=connection)
        right = build(spec_dir, output_dir=second, backend="local", connection=connection)
        assert left["splits"]["train"]["content_hash"] == right["splits"]["train"]["content_hash"]
        assert left["samples"] == right["samples"]
        manifests[entity] = (left, first)

    article_ids = set()
    clip_ids = set()
    for entity, (manifest, directory) in manifests.items():
        by_split = {name: _ids(directory / f"{name}.jsonl") for name in WINDOWS}
        assert set().union(*by_split.values())
        assert len(set().union(*by_split.values())) == sum(len(ids) for ids in by_split.values())
        for query_id in set().union(*by_split.values()):
            assert not query_id.endswith("-gap")
            assert "noclick" not in query_id
            assert query_id.startswith(entity)
        holdout = by_split["select_holdout"] | by_split["final_holdout"]
        rows = {}
        for split in ("train", "search_val"):
            loaded = [
                json.loads(line)
                for line in (directory / f"{split}.jsonl").read_text().splitlines()
                if line.strip()
            ]
            rows[split] = loaded
        for roles in manifest["samples"].values():
            for rule in roles.values():
                chosen = set(choose_query_ids(query_times(rows[rule["split"]]), rule))
                assert chosen.isdisjoint(holdout)
        if entity == "article":
            article_ids = by_split["train"]
        else:
            clip_ids = by_split["train"]
    assert article_ids != clip_ids

    spec = load_spec(tmp_path / "article")
    columns = {
        "queries": ["query_id", "query_ts"],
        "candidates": ["query_id", "item_id", "total_score", "item_age_seconds"],
        "labels": ["query_id", "item_id", "label", "label_event_ts", "click"],
        "signals": ["query_id", "item_id", "signal_value", "available_at"],
    }
    select_sql = assemble_select_sql(spec, columns)
    assert "ref__production" in select_sql
    assert "ref__recency" in select_sql
    assert "feat__signals__signal_value" in select_sql
    assert "aux_label__click" in select_sql
    violations = violation_sqls(spec, columns)
    assert "point_in_time" in {kind for kind, _sql in violations}
    assert "label_window" in {kind for kind, _sql in violations}
    assert "> q.query_ts" in "\n".join(sql for _kind, sql in violations)


def test_nfy_specs_render_without_harness_edits() -> None:
    start, end = date(2026, 10, 20), date(2026, 10, 26)
    for name, entity in (
        ("nfy_article_rerank", "article"),
        ("nfy_clip_rerank", "clip"),
    ):
        spec = load_spec(ROOT / "datasets" / name)
        assert spec.entity_type == entity
        views = dict(source_view_sql(spec, start, end))
        assert "2026-10-13" in views["amt_src_impressions"]
        rendered = dict(fragment_selects(spec, render_params(spec, start, end)))
        queries = rendered["queries"]
        assert f"entity_type = '{entity}'" in queries
        assert "DATE '2026-10-20'" in queries
        assert "DATE '2026-10-26'" in queries
        joined = "\n".join(rendered.values())
        assert "latest_impression_rn = 1" in joined
        assert "for-you" in joined
        assert "item_index" not in joined
    article_sql = (ROOT / "datasets" / "nfy_article_rerank" / "queries.sql").read_text()
    clip_sql = (ROOT / "datasets" / "nfy_clip_rerank" / "queries.sql").read_text()
    assert article_sql == clip_sql


class _Row(dict):
    def asDict(self, recursive: bool = False) -> dict:
        return dict(self)


class _Result:
    def __init__(self, rows: list[dict]) -> None:
        self._rows = [_Row(row) for row in rows]

    def collect(self) -> list[_Row]:
        return self._rows


class _Writer:
    def mode(self, _mode: str) -> _Writer:
        return self

    def saveAsTable(self, _name: str) -> None:
        return None


class _Frame:
    columns = [
        "query_id",
        "item_id",
        "query_ts",
        "label",
        "label_event_ts",
        "click",
        "total_score",
        "item_age_seconds",
        "profile_version_ts",
        "published_ts",
        "popularity_window_end",
        "value",
    ]
    write = _Writer()


class _RecordingSpark:
    """Records every SQL statement and returns empty or minimal summary rows."""

    def __init__(self) -> None:
        self.statements: list[str] = []

    def sql(self, text: str) -> _Result:
        self.statements.append(text)
        if "count(*) AS rows" in text:
            return _Result([{"rows": 4, "queries": 2, "min_query_ts": None, "max_query_ts": None,
                             "min_day": None, "max_day": None}])
        if "AS content_hash" in text:
            return _Result([{"content_hash": "h"}])
        if "avg(n) AS mean_n" in text:
            return _Result([{"mean_n": 2.0}])
        return _Result([])

    def table(self, _name: str) -> _Frame:
        return _Frame()

    def createDataFrame(self, _rows, _columns) -> _Frame:
        return _Frame()


def test_spark_build_scans_sources_once_and_stages_fragments(tmp_path: Path) -> None:
    from harness.data.build_dataset import build_spark

    spec = load_spec(ROOT / "datasets" / "nfy_article_rerank_demo").model_copy(
        update={"checks": []}
    )
    spark = _RecordingSpark()
    build_spark(spec, spark, catalog="cat", schema="sch", output_dir=tmp_path)
    statements = spark.statements
    prefix = f"`cat`.`sch`.`{spec.name}"

    source_views = [s for s in statements if s.startswith("CREATE OR REPLACE TEMP VIEW amt_src_")]
    assert len(source_views) == len(spec.sources)
    requests = next(s for s in source_views if "amt_src_requests " in s)
    assert "BETWEEN DATE '2026-09-30' AND DATE '2026-10-08'" in requests
    queries = next(
        s for s in statements if s.startswith("CREATE OR REPLACE TEMP VIEW amt_frag_queries")
    )
    assert "request_ts >= TIMESTAMP '2026-10-07 12:00:00'" in queries
    assert "request_ts < TIMESTAMP '2026-10-07 18:00:00'" in queries

    fragments = ["queries", "candidates", "labels", *spec.feature_groups]
    for name in fragments:
        staged = f"CREATE OR REPLACE TABLE {prefix}_stage_{name}` AS SELECT * FROM amt_frag_{name}"
        assert statements.count(staged) == 1

    first_check = next(i for i, s in enumerate(statements) if "LIMIT 5" in s)
    repointed = [
        i
        for i, s in enumerate(statements)
        if s.startswith("CREATE OR REPLACE TEMP VIEW amt_frag_") and "_stage_" in s
    ]
    assert len(repointed) == len(fragments)
    assert max(repointed) < first_check

    split_writes = [
        s
        for s in statements
        if s.startswith(f"CREATE OR REPLACE TABLE {prefix}_")
        and "_stage_" not in s.split(" AS ", 1)[0]
    ]
    assert len(split_writes) == 4
    assert all(f"FROM {prefix}_stage_assembled`" in s for s in split_writes)
    search_val = next(s for s in split_writes if f"{prefix}_search_val`" in s)
    assert "query_ts >= TIMESTAMP '2026-10-07 15:00:00'" in search_val
    assert "query_ts < TIMESTAMP '2026-10-07 16:00:00'" in search_val
    assert not any("amt_frag_" in s or "amt_src_" in s for s in split_writes)

    dropped = [s for s in statements if s.startswith("DROP TABLE IF EXISTS")]
    assert len(dropped) == len(fragments) + 1


def test_overlapping_windows_rejected(tmp_path: Path) -> None:
    source = ROOT / "datasets" / "nfy_article_rerank"
    dest = tmp_path / "overlap"
    dest.mkdir()
    for path in source.rglob("*"):
        if path.is_file():
            target = dest / path.relative_to(source)
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(path.read_text())
    spec_path = dest / "dataset.yaml"
    spec_path.write_text(
        spec_path.read_text().replace(
            "search_val: 2026-10-20..2026-10-26",
            "search_val: 2026-10-18..2026-10-26",
        )
    )
    with pytest.raises(ValidationError, match="overlaps"):
        load_spec(dest)


def test_hour_windows_split_one_day(tmp_path: Path) -> None:
    from harness.data.contract import assign_split

    spec = load_spec(ROOT / "datasets" / "nfy_article_rerank_demo")
    assert [(name, start, end) for name, start, end in spec.split_ranges()] == [
        (name, date(2026, 10, 7), date(2026, 10, 7))
        for name in ("train", "search_val", "select_holdout", "final_holdout")
    ]
    assert assign_split(spec, "2026-10-07 11:59:59") is None
    assert assign_split(spec, "2026-10-07 12:00:00") == "train"
    assert assign_split(spec, "2026-10-07 14:59:59.9") == "train"
    assert assign_split(spec, "2026-10-07 15:00:00") == "search_val"
    assert assign_split(spec, "2026-10-07 17:30:00") == "final_holdout"
    assert assign_split(spec, "2026-10-07 18:00:00") is None

    full = load_spec(ROOT / "datasets" / "nfy_article_rerank")
    assert assign_split(full, "2026-10-18 23:59:59") == "train"
    assert assign_split(full, "2026-10-19 00:00:00") is None
    params = render_params(full, date(2026, 10, 20), date(2026, 10, 26))
    assert params["start_ts"] == "2026-10-20 00:00:00"
    assert params["end_ts"] == "2026-10-27 00:00:00"

    source = ROOT / "datasets" / "nfy_article_rerank_demo"
    dest = tmp_path / "gap"
    dest.mkdir()
    for path in source.rglob("*"):
        if path.is_file():
            target = dest / path.relative_to(source)
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(path.read_text())
    spec_path = dest / "dataset.yaml"
    text = spec_path.read_text()
    spec_path.write_text(text.replace("gap_days: 0", "gap_days: 0\n  gap_hours: 1"))
    with pytest.raises(ValidationError, match="spec requires 1:00:00"):
        load_spec(dest)
    spec_path.write_text(
        text.replace("final_holdout: 2026-10-07T17:00", "final_holdout: 2026-10-07T16:30")
    )
    with pytest.raises(ValidationError, match="overlaps"):
        load_spec(dest)
    spec_path.write_text(
        text.replace("T17:00..2026-10-07T18:00", "T17:00..2026-10-08")
    )
    with pytest.raises(ValidationError, match="both be dates or both be timestamps"):
        load_spec(dest)


def test_prod_ndcg_tolerance() -> None:
    from harness.data.checks import apply_checks

    spec = load_spec(ROOT / "datasets" / "nfy_article_rerank")
    summary = {
        "label_distribution": {"0": 1, "1": 1},
        "null_rates": {},
        "queries_per_day": {"2026-10-20": 40000},
        "min_queries_per_day": 40000,
        "candidates_per_query_mean": 30.0,
    }
    summaries = {"search_val": summary}
    apply_checks(
        spec,
        summaries,
        {"prod_ndcg_reproduces": 0.500},
        {"prod_ndcg_reproduces": 0.504},
    )
    with pytest.raises(DatasetBuildError) as failure:
        apply_checks(
            spec,
            summaries,
            {"prod_ndcg_reproduces": 0.500},
            {"prod_ndcg_reproduces": 0.520},
        )
    assert failure.value.kind == "check"


def test_dab_jobs_wire_builder_train_confirm_pruner() -> None:
    bundle = yaml.safe_load((ROOT / "databricks.yml").read_text())
    assert bundle["variables"]["cpu_node_type_id"]["default"] == "m5d.xlarge"
    assert bundle["artifacts"]["amt_wheel"]["type"] == "whl"
    jobs = {
        "build_dataset": yaml.safe_load((ROOT / "resources" / "build_dataset.job.yml").read_text()),
        "train": yaml.safe_load((ROOT / "resources" / "train.job.yml").read_text()),
        "confirm": yaml.safe_load((ROOT / "resources" / "confirm.job.yml").read_text()),
        "pruner": yaml.safe_load((ROOT / "resources" / "pruner.job.yml").read_text()),
    }
    build_job = jobs["build_dataset"]["resources"]["jobs"]["build_dataset"]
    train_job = jobs["train"]["resources"]["jobs"]["train"]
    confirm_job = jobs["confirm"]["resources"]["jobs"]["confirm"]
    pruner_job = jobs["pruner"]["resources"]["jobs"]["pruner"]
    assert build_job["timeout_seconds"] == 21600
    assert build_job["max_concurrent_runs"] == 1
    assert train_job["timeout_seconds"] == 14400
    assert train_job["max_concurrent_runs"] == 4
    assert confirm_job["timeout_seconds"] == 14400
    assert confirm_job["max_concurrent_runs"] == 3
    assert pruner_job["timeout_seconds"] == 1800
    train_argv = train_job["tasks"][0]["spark_python_task"]["parameters"]
    confirm_argv = confirm_job["tasks"][0]["spark_python_task"]["parameters"]
    assert "--confirmation" not in train_argv
    assert confirm_argv[:5] == ["--confirmation", "--split", "select_holdout", "--rung", "R3"]
    confirm_names = {item["name"] for item in confirm_job["parameters"]}
    assert "split" not in confirm_names
    assert "rung" not in confirm_names
    assert pruner_job["tasks"][0]["spark_python_task"]["python_file"].endswith("amt/pruner.py")
    assert train_job["tasks"][0]["new_cluster"]["num_workers"] == 0
    assert build_job["tasks"][0]["new_cluster"]["node_type_id"] == "${var.cpu_node_type_id}"
