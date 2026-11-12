"""Train one candidate on a rung and log the contract metrics.

Search rungs score ``search_val`` only. ``select_holdout`` requires
``--confirmation``. ``final_holdout`` requires ``--final-holdout`` as well.
The candidate's ``fit`` receives a slice of the train split, never a holdout.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from amt.schemas import mlflow_tags  # noqa: E402
from harness.baselines import (  # noqa: E402
    LGBMReferenceRanker,
    PopularityRanker,
    RandomRanker,
    ReferenceScoreRanker,
)
from harness.data.build_dataset import project_root, resolve_spec  # noqa: E402
from harness.data.contract import HOLDOUT_SPLITS, SPLIT_NAMES, load_spec  # noqa: E402
from harness.data.sampling import choose_query_ids, query_ids_hash, query_times  # noqa: E402
from harness.evaluate import metrics, paired_day_blocked_delta  # noqa: E402
from harness.interface import FitContext, RankingFrame  # noqa: E402
from harness.latency import LATENCY_NODE_TYPE, latency_p95_ms  # noqa: E402

DEADLINES = {"R0": 600, "R1": 1800, "R2": 5400, "R3": 14400}


@dataclass
class TrainResult:
    status: str
    metrics: dict[str, float] = field(default_factory=dict)
    tags: dict[str, str] = field(default_factory=dict)
    latency_p95_ms: float | None = None
    reason: str | None = None


def load_rows(path: Path) -> list[dict]:
    rows = []
    for line in path.read_text().splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def load_rung_from_spark(
    spec,
    manifest: dict,
    rung: str,
    *,
    catalog: str,
    schema: str,
    confirmation: bool,
    eval_split: str,
) -> tuple[RankingFrame, RankingFrame]:
    """Read Delta splits written by the build job. Holdouts are read only when confirming."""

    from pyspark.sql import SparkSession

    spark = SparkSession.getActiveSession() or SparkSession.builder.getOrCreate()

    def table(split: str) -> str:
        return f"`{catalog}`.`{schema}`.`{spec.name}_{split}`"

    def choose(split: str, rule: dict) -> list:
        id_rows = spark.sql(
            f"SELECT query_id, min(query_ts) AS query_ts FROM {table(split)} GROUP BY query_id"
        ).collect()
        times = query_times([row.asDict(recursive=True) for row in id_rows])
        return choose_query_ids(times, rule)

    def pull(split: str, query_ids: list | None) -> list[dict]:
        if query_ids is None:
            frame = spark.sql(f"SELECT * FROM {table(split)}")
        else:
            spark.createDataFrame(
                [(str(query_id),) for query_id in query_ids],
                ["query_id"],
            ).createOrReplaceTempView("amt_keep_ids")
            frame = spark.sql(
                f"SELECT t.* FROM {table(split)} AS t "
                "INNER JOIN amt_keep_ids AS k ON t.query_id = k.query_id"
            )
        return [record.asDict(recursive=True) for record in frame.toLocalIterator()]

    if confirmation:
        return RankingFrame(pull("train", None)), RankingFrame(pull(eval_split, None))
    rules = manifest["samples"][rung]
    if rules["train"]["split"] != "train" or rules["eval"]["split"] != "search_val":
        raise ValueError("search rungs only sample train and search_val")
    train_ids = choose("train", rules["train"])
    eval_ids = choose("search_val", rules["eval"])
    if query_ids_hash(train_ids) != rules["train"]["query_ids_hash"]:
        raise RuntimeError("train sample does not match the dataset manifest")
    if query_ids_hash(eval_ids) != rules["eval"]["query_ids_hash"]:
        raise RuntimeError("eval sample does not match the dataset manifest")
    return RankingFrame(pull("train", train_ids)), RankingFrame(pull("search_val", eval_ids))


def load_dataset(dataset_dir: Path) -> tuple[dict, dict[str, list[dict]]]:
    manifest = json.loads((dataset_dir / "manifest.json").read_text())
    rows = {}
    for split in SPLIT_NAMES:
        path = dataset_dir / f"{split}.jsonl"
        if path.is_file():
            rows[split] = load_rows(path)
    if "train" not in rows:
        raise FileNotFoundError(f"{dataset_dir} has no train split")
    return manifest, rows


def _filter(rows: list[dict], query_ids: list) -> list[dict]:
    chosen = set(query_ids)
    return [row for row in rows if row["query_id"] in chosen]


def frames_for_rung(
    rows_by_split: dict[str, list[dict]],
    manifest: dict,
    rung: str,
    *,
    confirmation: bool = False,
    eval_split: str = "search_val",
) -> tuple[RankingFrame, RankingFrame]:
    """Train frame and the frame the harness scores. Fit does not see the eval frame."""

    if confirmation:
        if eval_split not in HOLDOUT_SPLITS:
            raise ValueError("confirmation scores a holdout split")
        train_ids = list(query_times(rows_by_split["train"]))
        eval_ids = list(query_times(rows_by_split[eval_split]))
        eval_rows = rows_by_split[eval_split]
    else:
        rules = manifest["samples"][rung]
        if rules["train"]["split"] != "train" or rules["eval"]["split"] != "search_val":
            raise ValueError("search rungs only sample train and search_val")
        train_ids = choose_query_ids(query_times(rows_by_split["train"]), rules["train"])
        eval_ids = choose_query_ids(query_times(rows_by_split["search_val"]), rules["eval"])
        if query_ids_hash(train_ids) != rules["train"]["query_ids_hash"]:
            raise RuntimeError("train sample does not match the dataset manifest")
        if query_ids_hash(eval_ids) != rules["eval"]["query_ids_hash"]:
            raise RuntimeError("eval sample does not match the dataset manifest")
        eval_rows = rows_by_split["search_val"]
    return RankingFrame(_filter(rows_by_split["train"], train_ids)), RankingFrame(
        _filter(eval_rows, eval_ids)
    )


def validation_slice(train: RankingFrame) -> tuple[RankingFrame, RankingFrame]:
    groups = train.groups()
    if len(groups) < 2:
        return train, RankingFrame([])
    cut = max(1, int(len(groups) * 0.9))
    if cut >= len(groups):
        cut = len(groups) - 1
    start = groups[cut][0]
    return RankingFrame(train.rows[:start]), RankingFrame(train.rows[start:])


def load_ranker_instance(model_path: Path):
    spec = importlib.util.spec_from_file_location("amt_candidate_model", model_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot import {model_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if not hasattr(module, "Ranker"):
        raise ImportError(f"{model_path} does not define Ranker")
    return module.Ranker()


def make_ranker(baseline: str | None, model_path: Path | None, reference: str | None):
    if baseline == "reference_score":
        return ReferenceScoreRanker(reference or "production")
    if baseline == "popularity":
        return PopularityRanker()
    if baseline == "random":
        return RandomRanker()
    if baseline == "lgbm_reference":
        return LGBMReferenceRanker()
    if model_path is None:
        raise FileNotFoundError("candidate model path is required")
    return load_ranker_instance(model_path)


def _finite(scores) -> bool:
    array = np.asarray(scores, dtype=float)
    return array.size > 0 and bool(np.isfinite(array).all())


def train_ranker(
    *,
    ranker,
    train: RankingFrame,
    evaluation: RankingFrame,
    seed: int,
    rung: str,
    deadline_s: int,
    require_loss: bool,
    check_determinism: bool,
    baseline_scores: list[float] | None = None,
    tags: dict[str, str] | None = None,
) -> TrainResult:
    result = TrainResult(status="failed", tags=dict(tags or {}))
    fit_train, fit_valid = validation_slice(train)
    deadline = time.monotonic() + deadline_s

    def _fit(model) -> None:
        context = FitContext(seed=seed, deadline=deadline, device="cpu")
        if context.expired():
            raise TimeoutError("deadline already passed")
        model.fit(fit_train, fit_valid, context)
        if context.expired():
            raise TimeoutError("fit exceeded the rung deadline")
        if require_loss and not context.loss_decreased():
            raise RuntimeError("R0 requires a decreasing train_loss")
        return context

    try:
        _fit(ranker)
        scores = ranker.score(evaluation.without_labels())
        if len(scores) != len(evaluation) or not _finite(scores):
            result.reason = "scores are missing or non-finite"
            return result
        if check_determinism:
            again = make_same(ranker)
            _fit(again)
            second = np.asarray(again.score(evaluation.without_labels()), dtype=float)
            first = np.asarray(scores, dtype=float)
            if first.shape != second.shape or not np.allclose(first, second, equal_nan=False):
                result.reason = "scores changed under the same seed"
                return result
        scored = metrics(evaluation, scores)
        if not all(value == value for value in scored.values()):
            result.reason = "metrics are non-finite"
            return result
        result.latency_p95_ms = latency_p95_ms(ranker, evaluation, n_requests=20, seed=seed)
        scored["latency_p95_ms"] = result.latency_p95_ms
        if baseline_scores is not None:
            delta = paired_day_blocked_delta(
                evaluation,
                np.asarray(scores, dtype=float),
                np.asarray(baseline_scores, dtype=float),
                seed=seed,
            )
            scored["delta_point"] = delta["point"]
            scored["delta_ci_lo"] = delta["ci_lo"]
            scored["delta_ci_hi"] = delta["ci_hi"]
        result.metrics = scored
        result.status = "ok"
        result.reason = None
        result.tags["latency_node_type"] = os.environ.get("AMT_NODE_TYPE", LATENCY_NODE_TYPE)
        return result
    except Exception as exc:
        result.reason = str(exc)
        result.metrics = {}
        return result


def make_same(ranker):
    """A second instance for the R0 determinism check, when the class allows it."""

    kind = type(ranker)
    if kind.__name__ == "ReferenceScoreRanker":
        return kind(ranker.score_name)
    if kind.__name__ in {"PopularityRanker", "RandomRanker"}:
        return kind()
    if kind.__name__ == "LGBMReferenceRanker":
        return kind(
            n_estimators=ranker.n_estimators,
            num_leaves=ranker.num_leaves,
            learning_rate=ranker.learning_rate,
            min_child_samples=ranker.min_child_samples,
        )
    try:
        return kind()
    except TypeError:
        return ranker


def log_mlflow(result: TrainResult, experiment: str | None) -> str | None:
    if not experiment:
        return None
    import mlflow

    uri = os.environ.get("MLFLOW_TRACKING_URI", "databricks")
    mlflow.set_tracking_uri(uri)
    mlflow.set_experiment(experiment)
    with mlflow.start_run() as active:
        mlflow.set_tags(result.tags)
        mlflow.set_tag("status", result.status)
        if result.reason:
            mlflow.set_tag("failure_reason", result.reason[:500])
        if result.status == "ok":
            mlflow.log_metrics(result.metrics)
        return active.info.run_id


def _guard_split(split: str, confirmation: bool, final_holdout: bool) -> None:
    if split == "final_holdout" and not (confirmation and final_holdout):
        raise SystemExit("final_holdout requires --confirmation and --final-holdout")
    if split == "select_holdout" and not confirmation:
        raise SystemExit("select_holdout is confirmation-only")
    if confirmation and split not in HOLDOUT_SPLITS:
        raise SystemExit("confirmation only scores a holdout split")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Train one candidate on one rung")
    parser.add_argument("--candidate_id", required=True)
    parser.add_argument("--rung", required=True, choices=("R0", "R1", "R2", "R3"))
    parser.add_argument("--seed", required=True, type=int)
    parser.add_argument("--split", required=True)
    parser.add_argument("--batch", required=True)
    parser.add_argument("--spec", required=True)
    parser.add_argument("--dataset-dir", default="")
    parser.add_argument("--catalog", default="")
    parser.add_argument("--schema", default="")
    parser.add_argument("--model", default="")
    parser.add_argument("--baseline", default="")
    parser.add_argument("--reference-score", default="production")
    parser.add_argument("--run-id", default="")
    parser.add_argument("--node-id", default="")
    parser.add_argument("--parent-id", default="")
    parser.add_argument("--family", default="candidate")
    parser.add_argument("--operator", default="draft")
    parser.add_argument("--git-sha", default="unknown")
    parser.add_argument("--experiment-name", default="")
    parser.add_argument("--deadline-seconds", type=int, default=0)
    parser.add_argument("--confirmation", action="store_true")
    parser.add_argument("--final-holdout", action="store_true")
    parser.add_argument("--baseline-scores", default="")
    args = parser.parse_args(argv)
    _guard_split(args.split, args.confirmation, args.final_holdout)
    spec = load_spec(resolve_spec(args.spec))
    catalog = args.catalog or spec.output.catalog
    schema = args.schema or spec.output.schema_name
    dataset_dir = (
        Path(args.dataset_dir)
        if args.dataset_dir
        else Path(f"/Volumes/{catalog}/{schema}/amt_artifacts") / spec.name
    )
    eval_split = args.split if args.confirmation else "search_val"
    if not args.confirmation and args.split not in {"search_val", "train"}:
        raise SystemExit("search runs score search_val; pass the rung, not a holdout")
    if (dataset_dir / "train.jsonl").is_file():
        manifest, rows = load_dataset(dataset_dir)
        train, evaluation = frames_for_rung(
            rows,
            manifest,
            args.rung,
            confirmation=args.confirmation,
            eval_split=eval_split,
        )
    else:
        manifest = json.loads((dataset_dir / "manifest.json").read_text())
        train, evaluation = load_rung_from_spark(
            spec,
            manifest,
            args.rung,
            catalog=catalog,
            schema=schema,
            confirmation=args.confirmation,
            eval_split=eval_split,
        )
    model_path = (
        Path(args.model)
        if args.model
        else project_root() / "candidates" / args.candidate_id / "model.py"
    )
    ranker = make_ranker(args.baseline or None, model_path, args.reference_score)
    tags = mlflow_tags(
        run_id=args.run_id or args.batch,
        node_id=args.node_id or args.candidate_id,
        parent_id=args.parent_id,
        family=args.family,
        operator=args.operator,
        rung=args.rung,
        seed=args.seed,
        batch=args.batch,
        split=eval_split,
        git_sha=args.git_sha,
    )
    baseline_scores = None
    if args.baseline_scores:
        baseline_scores = json.loads(Path(args.baseline_scores).read_text())
    result = train_ranker(
        ranker=ranker,
        train=train,
        evaluation=evaluation,
        seed=args.seed,
        rung=args.rung,
        deadline_s=args.deadline_seconds or DEADLINES[args.rung],
        require_loss=not args.baseline and args.rung == "R0",
        check_determinism=args.rung == "R0",
        baseline_scores=baseline_scores,
        tags=tags,
    )
    run_id = log_mlflow(result, args.experiment_name or None)
    payload = {
        "status": result.status,
        "reason": result.reason,
        "metrics": result.metrics,
        "mlflow_run_id": run_id,
    }
    print(json.dumps(payload))
    if result.status != "ok":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
