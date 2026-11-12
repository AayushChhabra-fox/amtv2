"""Build a dataset spec into contract tables and a manifest.

``python -m harness.data.build_dataset --spec datasets/<name>`` is the
Databricks job. Local tests pass a sqlite connection; the warehouse path
uses Spark and does not collect the fact table onto the driver.
"""

from __future__ import annotations

import argparse
import json
import os
import sqlite3
import sys
from datetime import date
from pathlib import Path
from typing import Any


def _bundle_root() -> Path:
    """Directory that contains ``harness/``. Databricks may copy the entry script."""

    script = globals().get("__file__")
    if script:
        resolved = Path(script).resolve()
        for parent in resolved.parents:
            if (parent / "harness").is_dir() and (parent / "datasets").is_dir():
                return parent
    argv = sys.argv
    if "--root" in argv:
        index = argv.index("--root")
        if index + 1 < len(argv):
            root = Path(argv[index + 1])
            if (root / "harness").is_dir():
                return root
    env = os.environ.get("AMT_ROOT")
    if env and (Path(env) / "harness").is_dir():
        return Path(env)
    return Path.cwd()


_ROOT = _bundle_root()
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from harness.data.assemble import (  # noqa: E402
    assemble_select_sql,
    candidate_bounds_sql,
    filtered_sql,
    fragment_selects,
    render,
    render_params,
    source_view_sql,
    violation_sqls,
)
from harness.data.checks import (  # noqa: E402
    apply_checks,
    measure_metrics,
    metric_value,
    score_array,
    summarize,
)
from harness.data.contract import (  # noqa: E402
    DatasetBuildError,
    DatasetSpec,
    assemble_rows,
    assign_split,
    forbidden_hit,
    load_spec,
    split_summary,
)
from harness.data.sampling import (  # noqa: E402
    choose_query_ids,
    query_ids_hash,
    query_times,
    sample_rules,
)
from harness.interface import RankingFrame  # noqa: E402


def project_root() -> Path:
    override = os.environ.get("AMT_ROOT")
    if override:
        return Path(override)
    import harness

    sibling = Path(harness.__file__).resolve().parent.parent
    if (sibling / "datasets").is_dir():
        return sibling
    for parent in _bundle_root().parents:
        if (parent / "datasets").is_dir() and (parent / "harness").is_dir():
            return parent
    raise FileNotFoundError("could not find the AMT project root")


def resolve_spec(path: str | Path) -> Path:
    candidate = Path(path)
    if candidate.exists():
        return candidate
    rooted = project_root() / candidate
    if rooted.exists():
        return rooted
    return candidate


class SqliteExecutor:
    """Runs spec SQL against tables already loaded in a connection."""

    def __init__(self, connection: sqlite3.Connection) -> None:
        self.connection = connection
        self.connection.row_factory = sqlite3.Row

    def execute(self, sql: str) -> list[dict]:
        cursor = self.connection.execute(sql)
        self.connection.commit()
        if cursor.description is None:
            return []
        columns = [column[0] for column in cursor.description]
        return [dict(zip(columns, row, strict=True)) for row in cursor.fetchall()]

    def replace_view(self, view: str, select_sql: str) -> None:
        self.execute(f"DROP VIEW IF EXISTS temp.{view}")
        self.execute(f"CREATE TEMP VIEW {view} AS {select_sql}")


class SparkExecutor:
    def __init__(self, spark) -> None:
        self.spark = spark

    def execute(self, sql: str) -> list[dict]:
        return [row.asDict(recursive=True) for row in self.spark.sql(sql).collect()]

    def columns(self, view: str) -> list[str]:
        return list(self.spark.table(view).columns)

    def replace_view(self, view: str, select_sql: str) -> None:
        self.spark.sql(f"CREATE OR REPLACE TEMP VIEW {view} AS {select_sql}")

    def write_table(self, table: str, select_sql: str) -> None:
        self.spark.sql(f"CREATE OR REPLACE TABLE {table} AS {select_sql}")

    def drop_table(self, table: str) -> None:
        self.spark.sql(f"DROP TABLE IF EXISTS {table}")


def restrict_to_split(rows: list[dict], spec: DatasetSpec, split_name: str) -> list[dict]:
    kept = []
    for row in rows:
        assigned = assign_split(spec, row["query_ts"])
        if assigned == split_name:
            kept.append(row)
        elif assigned is not None:
            raise DatasetBuildError(
                "split",
                f"{split_name} build contains a row from {assigned}",
                [{"query_id": row.get("query_id")}],
            )
    return kept


def _prepare_sources(executor, spec: DatasetSpec, start: date, end: date) -> None:
    for view, select_sql in source_view_sql(spec, start, end):
        executor.replace_view(view, select_sql)


def _expected_metrics(spec: DatasetSpec, executor: SqliteExecutor) -> dict[str, float]:
    expected: dict[str, float] = {}
    ranges = {name: (start, end) for name, start, end in spec.split_ranges()}
    for check in spec.checks:
        if not check.expected or not check.split:
            continue
        start, end = ranges[check.split]
        _prepare_sources(executor, spec, start, end)
        params = render_params(spec, start, end)
        rows = executor.execute(render(spec.sql_text(check.expected), params))
        if not rows:
            raise DatasetBuildError("check", f"{check.name} expected SQL returned no rows")
        row = rows[0]
        if "metric" in row:
            expected[check.name] = float(row["metric"])
        elif len(row) == 1:
            expected[check.name] = float(next(iter(row.values())))
        else:
            raise DatasetBuildError("check", f"{check.name} expected SQL needs a metric column")
    return expected


def _manifest(
    spec: DatasetSpec,
    rows_by_split: dict[str, list[dict]],
    checks: list[dict],
    *,
    hash_alg: str = "sha256-canonical-rows",
) -> dict:
    rules = sample_rules(spec)
    times = {name: query_times(rows) for name, rows in rows_by_split.items()}
    samples: dict[str, dict] = {}
    for rung, roles in rules.items():
        samples[rung] = {}
        for role, rule in roles.items():
            chosen = choose_query_ids(times[rule["split"]], rule)
            samples[rung][role] = {
                **rule,
                "n_queries": len(chosen),
                "query_ids_hash": query_ids_hash(chosen),
            }
    splits = {}
    for name, rows in rows_by_split.items():
        summary = split_summary(rows)
        summary["content_hash_alg"] = hash_alg
        splits[name] = summary
    return {
        "dataset": spec.name,
        "task_type": spec.task_type,
        "entity_type": spec.entity_type,
        "spec_hash": spec.spec_hash(),
        "label_window": spec.labels.label_window,
        "feature_groups": {
            name: {"available_at": group.available_at, "keys": list(group.keys)}
            for name, group in spec.feature_groups.items()
        },
        "forbidden_columns": list(spec.forbidden_columns),
        "reference_scores": list(spec.reference_scores),
        "splits": splits,
        "samples": samples,
        "checks": checks,
    }


def _write_local(
    directory: Path,
    rows_by_split: dict[str, list[dict]],
    manifest: dict,
    expected: dict,
) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    for split, rows in rows_by_split.items():
        path = directory / f"{split}.jsonl"
        with path.open("w") as handle:
            for row in rows:
                handle.write(json.dumps(_plain_row(row), sort_keys=True))
                handle.write("\n")
    (directory / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True))
    if expected:
        (directory / "expected_metrics.json").write_text(
            json.dumps(expected, indent=2, sort_keys=True)
        )


def _plain_row(row: dict) -> dict:
    exported = {}
    for key, value in row.items():
        if hasattr(value, "isoformat") and not isinstance(value, str):
            exported[key] = value.isoformat(sep=" ")
        else:
            exported[key] = value
    return exported


def build_local(
    spec: DatasetSpec,
    connection: sqlite3.Connection,
    output_dir: Path,
    *,
    skip_checks: bool = False,
) -> dict:
    if any("." in source.table for source in spec.sources.values()):
        raise DatasetBuildError(
            "contract",
            "warehouse table names need --backend spark",
        )
    executor = SqliteExecutor(connection)
    rows_by_split: dict[str, list[dict]] = {}
    for split_name, start, end in spec.split_ranges():
        _prepare_sources(executor, spec, start, end)
        params = render_params(spec, start, end)
        loaded = {name: executor.execute(sql) for name, sql in fragment_selects(spec, params)}
        features = {
            name: rows
            for name, rows in loaded.items()
            if name not in {"queries", "candidates", "labels"}
        }
        assembled = assemble_rows(
            queries=loaded["queries"],
            candidates=loaded["candidates"],
            labels=loaded["labels"],
            features=features,
            spec=spec,
        )
        rows_by_split[split_name] = restrict_to_split(assembled, spec, split_name)
    expected = {} if skip_checks else _expected_metrics(spec, executor)
    checks: list[dict] = []
    if not skip_checks:
        summaries = {split: summarize(spec, split, rows) for split, rows in rows_by_split.items()}
        measured = measure_metrics(spec, rows_by_split)
        checks = [result.as_dict() for result in apply_checks(spec, summaries, measured, expected)]
    manifest = _manifest(spec, rows_by_split, checks)
    _write_local(output_dir, rows_by_split, manifest, expected)
    return manifest


def _table_name(catalog: str, schema: str, dataset: str, split: str) -> str:
    parts = [catalog, schema, f"{dataset}_{split}"]
    if not all(part.replace("_", "").isalnum() for part in parts):
        raise DatasetBuildError("contract", f"unsafe table name {parts}")
    return ".".join(f"`{part}`" for part in parts)


def _full_span(spec: DatasetSpec) -> tuple[date, date]:
    """First and last day across all splits. Gap days are dropped by the split filter."""

    ranges = spec.split_ranges()
    return min(start for _, start, _ in ranges), max(end for _, _, end in ranges)


def _stage(
    executor: SparkExecutor,
    catalog: str,
    schema: str,
    spec: DatasetSpec,
    name: str,
    select_sql: str,
) -> str:
    table = _table_name(catalog, schema, spec.name, f"stage_{name}")
    executor.write_table(table, select_sql)
    return table


def _spark_hash(executor: SparkExecutor, table: str) -> str:
    rows = executor.execute(
        "SELECT sha2(cast(sum(xxhash64(query_id, item_id, cast(label AS STRING))) AS STRING), 256) "
        f"AS content_hash FROM {table}"
    )
    return str(rows[0]["content_hash"])


def _spark_summary_inputs(executor: SparkExecutor, table: str) -> dict[str, Any]:
    counts = executor.execute(
        f"SELECT count(*) AS rows, count(DISTINCT query_id) AS queries, "
        f"min(query_ts) AS min_query_ts, max(query_ts) AS max_query_ts, "
        f"min(event_day) AS min_day, max(event_day) AS max_day FROM {table}"
    )[0]
    day_rows = executor.execute(
        f"SELECT cast(event_day AS STRING) AS event_day, count(DISTINCT query_id) AS n "
        f"FROM {table} GROUP BY event_day"
    )
    return {"counts": counts, "days": {row["event_day"]: int(row["n"]) for row in day_rows}}


def build_spark(
    spec: DatasetSpec,
    spark,
    *,
    catalog: str,
    schema: str,
    output_dir: Path,
    skip_checks: bool = False,
) -> dict:
    executor = SparkExecutor(spark)
    column_sets: dict[str, list[str]] = {}
    written: dict[str, str] = {}
    staged: list[str] = []
    span_start, span_end = _full_span(spec)
    _prepare_sources(executor, spec, span_start, span_end)
    fragments = fragment_selects(spec, render_params(spec, span_start, span_end))
    for name, sql in fragments:
        view = f"amt_frag_{name}"
        executor.replace_view(view, sql)
        for column in executor.columns(view):
            pattern = forbidden_hit(column, spec.forbidden_columns)
            if pattern is not None:
                raise DatasetBuildError(
                    "forbidden_column",
                    f"{name} contains forbidden column {column}",
                    [{"column": column, "pattern": pattern, "fragment": name}],
                )
    # Temp views are recomputed from the sources on every query, so each
    # fragment is materialized once and every later query reads the copy.
    for name, _sql in fragments:
        view = f"amt_frag_{name}"
        table = _stage(executor, catalog, schema, spec, name, f"SELECT * FROM {view}")
        staged.append(table)
        executor.replace_view(view, f"SELECT * FROM {table}")
        column_sets[name] = executor.columns(view)
    for kind, sql in violation_sqls(spec, column_sets):
        examples = executor.execute(sql)
        if examples:
            raise DatasetBuildError(kind, "contract violation", examples)
    assembled = _stage(
        executor, catalog, schema, spec, "assembled", assemble_select_sql(spec, column_sets)
    )
    staged.append(assembled)
    assembled_sql = f"SELECT * FROM {assembled}"
    for split_name, start, end in spec.split_ranges():
        bounds = candidate_bounds_sql(spec, assembled_sql, start, end)
        if bounds:
            examples = executor.execute(bounds)
            if examples:
                raise DatasetBuildError(
                    "contract",
                    "candidates per query are outside candidate_bounds",
                    examples,
                )
        table = _table_name(catalog, schema, spec.name, split_name)
        executor.write_table(table, filtered_sql(assembled_sql, spec, start, end))
        written[split_name] = table

    summaries = {}
    split_meta = {}
    rows_for_metrics: dict[str, list[dict]] = {}
    metric_splits = {check.split for check in spec.checks if check.split and check.metric}
    for split_name, table in written.items():
        observed = _spark_summary_inputs(executor, table)
        counts = observed["counts"]
        split_meta[split_name] = {
            "rows": int(counts["rows"] or 0),
            "queries": int(counts["queries"] or 0),
            "min_query_ts": str(counts["min_query_ts"]) if counts["min_query_ts"] else None,
            "max_query_ts": str(counts["max_query_ts"]) if counts["max_query_ts"] else None,
            "min_day": str(counts["min_day"]) if counts["min_day"] else None,
            "max_day": str(counts["max_day"]) if counts["max_day"] else None,
            "content_hash": _spark_hash(executor, table),
            "content_hash_alg": "spark-xxhash64-sum-sha2",
        }
        if split_name in metric_splits:
            needed = ["query_id", "item_id", "query_ts", "event_day", "label"]
            for check in spec.checks:
                if check.split == split_name and check.score:
                    from harness.data.contract import ref_column

                    needed.append(ref_column(check.score))
            columns = ", ".join(f"`{column}`" for column in dict.fromkeys(needed))
            rows_for_metrics[split_name] = executor.execute(f"SELECT {columns} FROM {table}")
        day_counts = observed["days"]
        mean_rows = executor.execute(
            "SELECT avg(n) AS mean_n FROM ("
            f"SELECT query_id, count(*) AS n FROM {table} GROUP BY query_id)"
        )
        label_rows = executor.execute(
            f"SELECT cast(label AS STRING) AS label, count(*) AS n FROM {table} GROUP BY label"
        )
        from harness.data.checks import window_min_queries

        summaries[split_name] = {
            "rows": split_meta[split_name]["rows"],
            "queries": split_meta[split_name]["queries"],
            "label_distribution": {row["label"]: int(row["n"]) for row in label_rows},
            "null_rates": {},
            "queries_per_day": {day: int(count) for day, count in day_counts.items()},
            "min_queries_per_day": window_min_queries(spec, split_name, day_counts),
            "candidates_per_query_mean": float(mean_rows[0]["mean_n"] or 0.0),
        }

    expected: dict[str, float] = {}
    ranges = {name: (start, end) for name, start, end in spec.split_ranges()}
    for check in spec.checks:
        if not check.expected or not check.split:
            continue
        start, end = ranges[check.split]
        _prepare_sources(executor, spec, start, end)
        params = render_params(spec, start, end)
        rows = executor.execute(render(spec.sql_text(check.expected), params))
        if not rows or "metric" not in rows[0]:
            raise DatasetBuildError("check", f"{check.name} expected SQL needs a metric column")
        expected[check.name] = float(rows[0]["metric"])

    measured: dict[str, float] = {}
    for check in spec.checks:
        if not check.metric or not check.split or not check.score:
            continue
        frame = RankingFrame(rows_for_metrics.get(check.split, []))
        measured[check.name] = metric_value(check.metric, frame, score_array(frame, check.score))

    checks: list[dict] = []
    if not skip_checks:
        checks = [result.as_dict() for result in apply_checks(spec, summaries, measured, expected)]
    times = {}
    for split_name, table in written.items():
        id_rows = executor.execute(
            f"SELECT query_id, min(query_ts) AS query_ts FROM {table} GROUP BY query_id"
        )
        times[split_name] = query_times(id_rows)
    rules = sample_rules(spec)
    samples: dict[str, dict] = {}
    for rung, roles in rules.items():
        samples[rung] = {}
        for role, rule in roles.items():
            chosen = choose_query_ids(times[rule["split"]], rule)
            samples[rung][role] = {
                **rule,
                "n_queries": len(chosen),
                "query_ids_hash": query_ids_hash(chosen),
            }
    manifest = {
        "dataset": spec.name,
        "task_type": spec.task_type,
        "entity_type": spec.entity_type,
        "spec_hash": spec.spec_hash(),
        "label_window": spec.labels.label_window,
        "feature_groups": {
            name: {"available_at": group.available_at, "keys": list(group.keys)}
            for name, group in spec.feature_groups.items()
        },
        "forbidden_columns": list(spec.forbidden_columns),
        "reference_scores": list(spec.reference_scores),
        "splits": split_meta,
        "samples": samples,
        "checks": checks,
        "tables": written,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    text = json.dumps(manifest, indent=2, sort_keys=True, default=str)
    (output_dir / "manifest.json").write_text(text)
    if expected:
        (output_dir / "expected_metrics.json").write_text(
            json.dumps(expected, indent=2, sort_keys=True)
        )
    manifest_table = _table_name(catalog, schema, spec.name, "manifest")
    spark.createDataFrame([(text,)], ["manifest"]).write.mode("overwrite").saveAsTable(
        manifest_table.replace("`", "")
    )
    for table in staged:
        executor.drop_table(table)
    return manifest


def build(
    spec_path: str | Path,
    *,
    output_dir: str | Path,
    backend: str = "local",
    connection: sqlite3.Connection | None = None,
    spark=None,
    catalog: str | None = None,
    schema: str | None = None,
    skip_checks: bool = False,
) -> dict:
    spec = load_spec(resolve_spec(spec_path))
    if spec.task_type != "ranking":
        raise DatasetBuildError("contract", f"unsupported task_type {spec.task_type}")
    destination = Path(output_dir)
    if backend == "local":
        if connection is None:
            raise DatasetBuildError("contract", "local backend needs a sqlite connection")
        return build_local(spec, connection, destination, skip_checks=skip_checks)
    if backend == "spark":
        session = spark or _active_spark()
        return build_spark(
            spec,
            session,
            catalog=catalog or spec.output.catalog,
            schema=schema or spec.output.schema_name,
            output_dir=destination,
            skip_checks=skip_checks,
        )
    raise DatasetBuildError("contract", f"unknown backend {backend}")


def _active_spark():
    try:
        from pyspark.sql import SparkSession
    except ImportError as exc:
        raise DatasetBuildError("contract", "spark backend needs pyspark") from exc
    session = SparkSession.getActiveSession()
    if session is None:
        session = SparkSession.builder.getOrCreate()
    return session


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Build a dataset spec into contract tables")
    parser.add_argument("--spec", required=True)
    parser.add_argument("--backend", choices=("local", "spark"), default="spark")
    parser.add_argument("--output-dir", default="")
    parser.add_argument("--catalog", default="")
    parser.add_argument("--schema", default="")
    parser.add_argument("--sqlite", default="")
    parser.add_argument("--skip-checks", action="store_true")
    parser.add_argument("--root", default="")
    args = parser.parse_args(argv)
    if args.root:
        os.environ.setdefault("AMT_ROOT", args.root)
        if args.root not in sys.path:
            sys.path.insert(0, args.root)
    spec = load_spec(resolve_spec(args.spec))
    catalog = args.catalog or spec.output.catalog
    schema = args.schema or spec.output.schema_name
    output = args.output_dir or str(Path(f"/Volumes/{catalog}/{schema}/amt_artifacts") / spec.name)
    connection = sqlite3.connect(args.sqlite) if args.sqlite else None
    manifest = build(
        spec.root or args.spec,
        output_dir=output,
        backend=args.backend,
        connection=connection,
        catalog=catalog,
        schema=schema,
        skip_checks=args.skip_checks,
    )
    print(json.dumps({"dataset": manifest["dataset"], "spec_hash": manifest["spec_hash"]}))


if __name__ == "__main__":
    main()
