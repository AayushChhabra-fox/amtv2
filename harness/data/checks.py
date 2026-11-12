"""Generic data checks plus the assertions declared on a dataset spec."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from datetime import timedelta
from pathlib import Path

import numpy as np

from harness.data.contract import (
    SPLIT_NAMES,
    DatasetBuildError,
    DatasetSpec,
    load_spec,
    ref_column,
)
from harness.evaluate import click_ndcg_at_k, graded_ndcg_at_k
from harness.interface import RankingFrame


@dataclass
class CheckResult:
    name: str
    ok: bool
    detail: str

    def as_dict(self) -> dict:
        return {"name": self.name, "ok": self.ok, "detail": self.detail}


def summarize(spec: DatasetSpec, split: str, rows: list[dict]) -> dict:
    day_counts: dict[str, set] = {}
    per_query: dict = {}
    labels: dict[str, int] = {}
    feature_columns = sorted({key for row in rows for key in row if key.startswith("feat__")})
    nulls = dict.fromkeys(feature_columns, 0)
    for row in rows:
        label = row.get("label")
        labels[str(label)] = labels.get(str(label), 0) + 1
        day_counts.setdefault(str(row.get("event_day")), set()).add(row["query_id"])
        per_query[row["query_id"]] = per_query.get(row["query_id"], 0) + 1
        for column in feature_columns:
            if row.get(column) is None:
                nulls[column] += 1
    observed = {day: len(ids) for day, ids in day_counts.items()}
    return {
        "rows": len(rows),
        "queries": len(per_query),
        "label_distribution": dict(sorted(labels.items())),
        "null_rates": {
            column: (nulls[column] / len(rows) if rows else 0.0) for column in feature_columns
        },
        "queries_per_day": observed,
        "min_queries_per_day": window_min_queries(spec, split, observed),
        "candidates_per_query_mean": (
            sum(per_query.values()) / len(per_query) if per_query else 0.0
        ),
    }


def window_min_queries(spec: DatasetSpec, split: str, observed: dict[str, int]) -> int:
    start = end = None
    for name, window_start, window_end in spec.split_ranges():
        if name == split:
            start, end = window_start, window_end
            break
    if start is None or end is None:
        return min(observed.values()) if observed else 0
    day = start
    counts = []
    while day <= end:
        counts.append(int(observed.get(day.isoformat(), 0)))
        day += timedelta(days=1)
    return min(counts) if counts else 0


def measure_metrics(spec: DatasetSpec, rows_by_split: dict[str, list[dict]]) -> dict[str, float]:
    measured: dict[str, float] = {}
    for check in spec.checks:
        if not check.metric or not check.split or not check.score:
            continue
        frame = RankingFrame(rows_by_split.get(check.split, []))
        measured[check.name] = metric_value(check.metric, frame, score_array(frame, check.score))
    return measured


def metric_value(name: str, frame: RankingFrame, scores: np.ndarray) -> float:
    if name in {"ndcg@10_graded", "ndcg@10"}:
        return graded_ndcg_at_k(frame, scores, k=10)
    if name == "ndcg@10_click":
        return click_ndcg_at_k(frame, scores, k=10)
    raise DatasetBuildError("check", f"unsupported check metric {name}")


def score_array(frame: RankingFrame, score: str) -> np.ndarray:
    column = ref_column(score)
    if column not in frame.column_names():
        raise DatasetBuildError("check", f"score column {column} is not in the split")
    values = np.zeros(len(frame), dtype=float)
    for index, row in enumerate(frame.rows):
        raw = row.get(column)
        values[index] = float("nan") if raw is None else float(raw)
    return values


def apply_checks(
    spec: DatasetSpec,
    summaries: dict[str, dict],
    measured: dict[str, float],
    expected: dict[str, float] | None = None,
) -> list[CheckResult]:
    """Generic stats plus the spec's checks. A failed spec check raises."""

    expected = expected or {}
    results: list[CheckResult] = []
    for split, summary in summaries.items():
        results.append(
            CheckResult(
                f"label_distribution:{split}",
                True,
                json.dumps(summary["label_distribution"]),
            )
        )
        results.append(CheckResult(f"null_rates:{split}", True, json.dumps(summary["null_rates"])))
        results.append(
            CheckResult(
                f"queries_per_day:{split}",
                True,
                json.dumps(summary["queries_per_day"]),
            )
        )
        results.append(
            CheckResult(
                f"candidates_per_query:{split}",
                True,
                json.dumps({"mean": summary["candidates_per_query_mean"]}),
            )
        )
    failures: list[CheckResult] = []
    for check in spec.checks:
        result = _spec_check(check, summaries, measured, expected)
        results.append(result)
        if not result.ok:
            failures.append(result)
    if failures:
        detail = "; ".join(f"{item.name}: {item.detail}" for item in failures)
        raise DatasetBuildError("check", detail)
    return results


def _spec_check(
    check,
    summaries: dict[str, dict],
    measured: dict[str, float],
    expected: dict[str, float],
) -> CheckResult:
    splits = [check.split] if check.split else list(summaries)
    if check.name == "queries_per_day" or check.min is not None and check.name == "queries_per_day":
        worst = min(summaries[split]["min_queries_per_day"] for split in splits)
        ok = worst >= float(check.min if check.min is not None else 0)
        return CheckResult(check.name, ok, f"min daily queries {worst}")
    if check.name == "candidates_per_query_mean":
        lo, hi = check.between or [0, float("inf")]
        means = {split: summaries[split]["candidates_per_query_mean"] for split in splits}
        ok = all(lo <= value <= hi for value in means.values())
        return CheckResult(check.name, ok, json.dumps(means))
    if check.name == "prod_ndcg_reproduces":
        got = measured.get(check.name)
        target = expected.get(check.name)
        if got is None or target is None:
            return CheckResult(check.name, False, "missing harness metric or expected SQL metric")
        ok = abs(float(got) - float(target)) <= float(check.tol)
        detail = f"harness {got:.6f} expected {target:.6f} tol {check.tol}"
        return CheckResult(check.name, ok, detail)
    if check.min is not None:
        worst = min(summaries[split]["min_queries_per_day"] for split in splits)
        return CheckResult(check.name, worst >= float(check.min), f"min {worst}")
    return CheckResult(check.name, False, "unrecognized check")


def load_split_rows(dataset_dir: Path) -> dict[str, list[dict]]:
    rows: dict[str, list[dict]] = {}
    for split in SPLIT_NAMES:
        path = dataset_dir / f"{split}.jsonl"
        if not path.is_file():
            continue
        loaded = []
        for line in path.read_text().splitlines():
            if line.strip():
                loaded.append(json.loads(line))
        rows[split] = loaded
    if not rows:
        raise DatasetBuildError("check", f"no split files in {dataset_dir}")
    return rows


def run_checks(spec_path: str, dataset_dir: str | None = None) -> list[CheckResult]:
    """Check a build that was written to ``dataset_dir``.

    Reference-metric checks that need the warehouse SQL are recorded on the
    manifest by ``build_dataset``. This entry point replays the row-level
    checks and, when ``expected_metrics.json`` is present, the tolerance check.
    """

    spec = load_spec(spec_path)
    if dataset_dir is None:
        raise DatasetBuildError("check", "dataset_dir is required")
    root = Path(dataset_dir)
    rows = load_split_rows(root)
    expected_path = root / "expected_metrics.json"
    expected = json.loads(expected_path.read_text()) if expected_path.is_file() else {}
    summaries = {split: summarize(spec, split, split_rows) for split, split_rows in rows.items()}
    measured = measure_metrics(spec, rows)
    return apply_checks(spec, summaries, measured, expected)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run dataset contract checks")
    parser.add_argument("--spec", required=True)
    parser.add_argument("--dataset-dir", required=True)
    args = parser.parse_args()
    results = run_checks(args.spec, args.dataset_dir)
    for result in results:
        print(f"{'OK' if result.ok else 'FAIL'} {result.name} {result.detail}")


if __name__ == "__main__":
    main()
