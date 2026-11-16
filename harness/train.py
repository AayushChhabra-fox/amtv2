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
import traceback
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd


def _bundle_root() -> Path:
    """Directory that contains ``harness/``. Databricks may copy the entry script."""

    script = globals().get("__file__")
    if script:
        resolved = Path(script).resolve()
        for parent in resolved.parents:
            if (parent / "harness").is_dir() and (parent / "amt").is_dir():
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

from amt.schemas import mlflow_tags  # noqa: E402
from harness.baselines import (  # noqa: E402
    LGBMReferenceRanker,
    PopularityRanker,
    RandomRanker,
    ReferenceScoreRanker,
)
from harness.data.build_dataset import project_root, resolve_spec, split_parquet_path  # noqa: E402
from harness.data.contract import HOLDOUT_SPLITS, SPLIT_NAMES, load_spec  # noqa: E402
from harness.data.sampling import choose_query_ids, query_ids_hash, query_times  # noqa: E402
from harness.deps import activate_candidate_requirements  # noqa: E402
from harness.evaluate import metrics, paired_day_blocked_delta  # noqa: E402
from harness.interface import FitContext, RankingFrame  # noqa: E402
from harness.ladder import LADDER_IDS, UnsupportedDataset, make_ladder_ranker  # noqa: E402
from harness.latency import LATENCY_NODE_TYPE, latency_p95_ms  # noqa: E402

DEADLINES = {"R0": 600, "R1": 1800, "R2": 5400, "R3": 14400}


@dataclass
class TrainResult:
    status: str
    metrics: dict[str, float] = field(default_factory=dict)
    tags: dict[str, str] = field(default_factory=dict)
    latency_p95_ms: float | None = None
    reason: str | None = None
    traceback: str | None = None


@dataclass
class SplitLineage:
    """One split's table identity, Delta version, and sample hash."""

    name: str
    table: str = ""
    delta_version: int | None = None
    content_hash: str = ""
    query_ids_hash: str = ""
    rows: int | None = None
    queries: int | None = None
    storage: str = "delta"

    def as_dict(self) -> dict[str, str | int | None]:
        return {
            "name": self.name,
            "table": self.table,
            "delta_version": self.delta_version,
            "content_hash": self.content_hash,
            "query_ids_hash": self.query_ids_hash,
            "rows": self.rows,
            "queries": self.queries,
            "storage": self.storage,
        }


@dataclass
class DatasetLineage:
    """Train and eval versions scored by one run."""

    dataset: str
    spec_hash: str
    train: SplitLineage
    evaluation: SplitLineage

    def tags(self) -> dict[str, str]:
        tags = {
            "dataset_name": self.dataset,
            "dataset_spec_hash": self.spec_hash,
            "train_table": self.train.table,
            "train_delta_version": _version_text(self.train.delta_version),
            "train_content_hash": self.train.content_hash,
            "train_query_ids_hash": self.train.query_ids_hash,
            "eval_table": self.evaluation.table,
            "eval_delta_version": _version_text(self.evaluation.delta_version),
            "eval_content_hash": self.evaluation.content_hash,
            "eval_query_ids_hash": self.evaluation.query_ids_hash,
        }
        return {key: value for key, value in tags.items() if value}

    def as_dict(self) -> dict:
        return {
            "dataset": self.dataset,
            "spec_hash": self.spec_hash,
            "train": self.train.as_dict(),
            "evaluation": self.evaluation.as_dict(),
        }


def _version_text(version: int | None) -> str:
    return "" if version is None else str(version)


def load_rows(path: Path) -> list[dict]:
    rows = []
    for line in path.read_text().splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def load_rung(
    spec,
    manifest: dict,
    rung: str,
    *,
    catalog: str,
    schema: str,
    confirmation: bool,
    eval_split: str,
) -> tuple[RankingFrame, RankingFrame, DatasetLineage]:
    """Read one rung from volume Parquet, or from Spark when the file is absent.

    Holdouts are read only when confirming. Parquet lives at
    ``<volume>/<spec>/<spec_hash>/splits/<split>.parquet``. The Spark fallback
    uses ``toPandas`` rather than collecting rows one at a time, and pins the
    Delta version recorded on the manifest when the cache is missing.
    """

    from pyspark.sql import SparkSession

    spark = SparkSession.getActiveSession() or SparkSession.builder.getOrCreate()
    spec_hash = spec.spec_hash()

    def table(split: str) -> str:
        return f"`{catalog}`.`{schema}`.`{spec.name}_{split}`"

    def parquet(split: str) -> Path:
        return split_parquet_path(catalog, schema, spec.name, spec_hash, split)

    def choose(split: str, rule: dict) -> list:
        id_rows = spark.sql(
            f"SELECT query_id, min(query_ts) AS query_ts FROM {table(split)} GROUP BY query_id"
        ).collect()
        times = query_times([row.asDict(recursive=True) for row in id_rows])
        return choose_query_ids(times, rule)

    def pull(
        split: str, query_ids: list | None, query_hash: str
    ) -> tuple[RankingFrame, SplitLineage]:
        recorded = (manifest.get("splits") or {}).get(split) or {}
        version = recorded.get("delta_version")
        version_int = int(version) if version is not None else None
        path = parquet(split)
        frame = read_split_frame(
            spark, table(split), query_ids, path, delta_version=version_int
        )
        storage = "parquet" if path.is_file() else "delta"
        resolved = version_int
        if resolved is None and storage == "delta":
            resolved = _describe_delta_version(spark, table(split))
        lineage = SplitLineage(
            name=split,
            table=table(split),
            delta_version=resolved,
            content_hash=str(recorded.get("content_hash") or ""),
            query_ids_hash=query_hash,
            rows=recorded.get("rows"),
            queries=recorded.get("queries"),
            storage=storage,
        )
        return RankingFrame.from_pandas(frame), lineage

    if confirmation:
        train_frame, train_lineage = pull("train", None, "full")
        eval_frame, eval_lineage = pull(eval_split, None, "full")
    else:
        rules = manifest["samples"][rung]
        if rules["train"]["split"] != "train" or rules["eval"]["split"] != "search_val":
            raise ValueError("search rungs only sample train and search_val")
        train_ids = choose("train", rules["train"])
        eval_ids = choose("search_val", rules["eval"])
        if query_ids_hash(train_ids) != rules["train"]["query_ids_hash"]:
            raise RuntimeError("train sample does not match the dataset manifest")
        if query_ids_hash(eval_ids) != rules["eval"]["query_ids_hash"]:
            raise RuntimeError("eval sample does not match the dataset manifest")
        train_frame, train_lineage = pull("train", train_ids, rules["train"]["query_ids_hash"])
        eval_frame, eval_lineage = pull("search_val", eval_ids, rules["eval"]["query_ids_hash"])
    lineage = DatasetLineage(
        dataset=spec.name,
        spec_hash=str(manifest.get("spec_hash") or spec_hash),
        train=train_lineage,
        evaluation=eval_lineage,
    )
    return train_frame, eval_frame, lineage


def read_split_frame(
    spark,
    table: str,
    query_ids: list | None,
    parquet_path: Path,
    *,
    delta_version: int | None = None,
) -> pd.DataFrame:
    """Filtered split as a DataFrame. Volume Parquet wins over a Spark scan.

    A recorded Delta version is applied with ``VERSION AS OF`` when the Parquet
    cache is missing, so a later overwrite of the split table does not change
    the rows a rerun scores.
    """

    if parquet_path.is_file():
        frame = pd.read_parquet(parquet_path)
        if query_ids is None:
            return frame
        keep = {str(item) for item in query_ids}
        return frame.loc[frame["query_id"].astype(str).isin(keep)].reset_index(drop=True)
    version = delta_version if delta_version is not None else _describe_delta_version(spark, table)
    clause = f" VERSION AS OF {int(version)}" if version is not None else ""
    if query_ids is None:
        spark_frame = _sql_or_plain(
            spark, f"SELECT * FROM {table}{clause}", f"SELECT * FROM {table}"
        )
    else:
        spark.createDataFrame(
            [(str(query_id),) for query_id in query_ids],
            ["query_id"],
        ).createOrReplaceTempView("amt_keep_ids")
        pinned = (
            f"SELECT t.* FROM {table}{clause} AS t "
            "INNER JOIN amt_keep_ids AS k ON t.query_id = k.query_id"
        )
        plain = (
            f"SELECT t.* FROM {table} AS t INNER JOIN amt_keep_ids AS k ON t.query_id = k.query_id"
        )
        spark_frame = _sql_or_plain(spark, pinned, plain)
    return spark_frame.toPandas()


def _describe_delta_version(spark, table: str) -> int | None:
    try:
        rows = spark.sql(f"DESCRIBE HISTORY {table} LIMIT 1").collect()
    except Exception:
        return None
    if not rows:
        return None
    row = rows[0].asDict(recursive=True) if hasattr(rows[0], "asDict") else dict(rows[0])
    version = row.get("version")
    if version is None:
        return None
    try:
        return int(version)
    except (TypeError, ValueError):
        return None


def _sql_or_plain(spark, pinned: str, plain: str):
    if pinned == plain:
        return spark.sql(plain)
    try:
        return spark.sql(pinned)
    except Exception:
        return spark.sql(plain)


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
    if baseline in LADDER_IDS:
        return make_ladder_ranker(baseline)
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
    except UnsupportedDataset as exc:
        result.status = "unsupported"
        result.reason = str(exc)
        result.metrics = {}
        return result
    except Exception as exc:
        result.reason = str(exc)
        result.traceback = _traceback_tail(exc)
        result.metrics = {}
        return result


def candidate_failure_path(
    catalog: str,
    schema: str,
    run_id: str,
    node_id: str,
    rung: str,
    seed: int | str,
) -> Path:
    """``failure.json`` for one candidate job.

    ``AMT_TRAIN_FAILURE_ROOT`` redirects the write in tests. The cluster job
    leaves it unset and writes under the artifact volume.
    """

    override = os.environ.get("AMT_TRAIN_FAILURE_ROOT", "").strip()
    base = Path(override) if override else Path(f"/Volumes/{catalog}/{schema}/amt_artifacts")
    return base / "train_failures" / run_id / node_id / rung / str(seed) / "failure.json"


def write_candidate_failure(path: Path, payload: dict) -> None:
    """Best-effort ``failure.json``. A write error must not hide the exception."""

    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, indent=2, default=str) + "\n", encoding="utf-8")
    except OSError:
        return


def _traceback_tail(exc: BaseException) -> str:
    lines = traceback.format_exception(type(exc), exc, exc.__traceback__)
    return "\n".join("".join(lines).splitlines()[-40:])


def _record_candidate_failure(
    args: argparse.Namespace,
    model_path: Path | None,
    catalog: str,
    schema: str,
    payload: dict,
) -> None:
    node_id = str(args.node_id or args.candidate_id or "")
    run_id = str(args.run_id or args.batch or "")
    remote = candidate_failure_path(catalog, schema, run_id, node_id, args.rung, args.seed)
    write_candidate_failure(remote, payload)
    if model_path is not None and not args.baseline:
        write_candidate_failure(Path(model_path).parent / "failure.json", payload)


def _failure_payload(args: argparse.Namespace, exc: BaseException) -> dict:
    return {
        "exception_type": type(exc).__name__,
        "message": str(exc),
        "traceback": _traceback_tail(exc),
        "rung": str(getattr(args, "rung", "") or ""),
        "node_id": str(getattr(args, "node_id", "") or getattr(args, "candidate_id", "") or ""),
        "run_id": str(getattr(args, "run_id", "") or ""),
    }


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


def log_mlflow(
    result: TrainResult,
    experiment: str | None,
    *,
    ranker=None,
    baseline: str = "",
    model_path: Path | None = None,
    train: RankingFrame | None = None,
    evaluation: RankingFrame | None = None,
    lineage: DatasetLineage | None = None,
    rung: str = "",
    family: str = "",
    node_id: str = "",
    run_id: str = "",
    seed: int = 0,
    git_sha: str = "",
    eval_split: str = "search_val",
) -> dict[str, str | None]:
    """Log metrics, and at R3 the fitted ranker plus the dataset version."""

    empty = {
        "mlflow_run_id": None,
        "model_uri": None,
        "native_model_uri": None,
        "native_flavor": None,
    }
    if not experiment:
        return empty
    import mlflow

    from harness.mlflow_model import log_r3_models, should_log_r3_model

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
        logged = dict(empty)
        logged["mlflow_run_id"] = active.info.run_id
        if lineage is not None:
            mlflow.set_tags(lineage.tags())
        if (
            should_log_r3_model(rung, baseline, result.status)
            and ranker is not None
            and train is not None
            and evaluation is not None
            and lineage is not None
        ):
            models = log_r3_models(
                ranker=ranker,
                baseline=baseline,
                model_path=model_path,
                train=train,
                evaluation=evaluation,
                lineage=lineage,
                family=family,
                node_id=node_id,
                run_id=run_id,
                seed=seed,
                git_sha=git_sha,
                eval_split=eval_split,
            )
            logged.update(
                {
                    "model_uri": models.get("model_uri"),
                    "native_model_uri": models.get("native_model_uri"),
                    "native_flavor": models.get("native_flavor"),
                }
            )
        return logged


def _guard_confirm_token(confirmation: bool, token: str) -> None:
    """Holdout scoring runs only with a one-time token file."""

    if not confirmation:
        return
    if not token:
        raise SystemExit("confirmation requires --confirm-token")
    if not Path(token).is_file():
        raise SystemExit("confirmation token is missing or already used")


def _guard_confirmation_caller(confirmation: bool) -> None:
    """Backward-compatible name. Tokens replaced the cluster environment variable."""

    if confirmation:
        raise SystemExit("confirmation requires --confirm-token")


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
    parser.add_argument("--paired-reference", default="production")
    parser.add_argument("--write-scores", default="")
    parser.add_argument("--root", default="")
    parser.add_argument("--confirm-token", default="")
    parser.add_argument("--shared-cluster", default="")
    parser.add_argument("--matrix", default="")
    args = parser.parse_args(argv)
    if args.root:
        root_path = str(Path(args.root).resolve())
        if root_path not in sys.path:
            sys.path.insert(0, root_path)
    if args.matrix:
        _run_matrix(args)
        return
    _execute(args)


def _run_matrix(args: argparse.Namespace) -> None:
    cells = json.loads(Path(args.matrix).read_text(encoding="utf-8"))
    if not isinstance(cells, list):
        raise SystemExit("matrix file must be a JSON list")
    failures = 0
    for cell in cells:
        if not isinstance(cell, dict):
            raise SystemExit("matrix cell must be an object")
        cloned = argparse.Namespace(**vars(args))
        cloned.matrix = ""
        for key, value in cell.items():
            field = key.replace("-", "_")
            if hasattr(cloned, field):
                setattr(cloned, field, value)
        try:
            _execute(cloned)
        except SystemExit as exc:
            failures += 1 if exc.code else 0
    if failures:
        raise SystemExit(1)


def _execute(args: argparse.Namespace) -> None:
    args.seed = int(args.seed)
    _guard_confirm_token(args.confirmation, args.confirm_token)
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
        query_hash = "full" if args.confirmation else ""
        lineage = DatasetLineage(
            dataset=spec.name,
            spec_hash=str(manifest.get("spec_hash") or spec.spec_hash()),
            train=SplitLineage(name="train", query_ids_hash=query_hash, storage="jsonl"),
            evaluation=SplitLineage(
                name=eval_split, query_ids_hash=query_hash, storage="jsonl"
            ),
        )
    else:
        manifest = json.loads((dataset_dir / "manifest.json").read_text())
        train, evaluation, lineage = load_rung(
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
        else project_root() / "runs" / args.run_id / "candidates" / args.candidate_id / "model.py"
    )
    requirements_sha = ""
    install_s = 0.0
    if not args.baseline:
        requirements_sha, install_s = activate_candidate_requirements(model_path)
    try:
        ranker = make_ranker(args.baseline or None, model_path, args.reference_score)
    except Exception as exc:
        _record_candidate_failure(args, model_path, catalog, schema, _failure_payload(args, exc))
        raise
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
    if requirements_sha:
        tags["requirements_sha"] = requirements_sha
    from harness.scores_io import align_scores, read_scores, reference_vector, write_scores

    columns = getattr(evaluation, "columns", None)
    paired = reference_vector(evaluation, args.paired_reference) if columns else None
    keyed = None
    if args.baseline_scores:
        keyed = align_scores(evaluation, read_scores(Path(args.baseline_scores)))
    result = train_ranker(
        ranker=ranker,
        train=train,
        evaluation=evaluation,
        seed=args.seed,
        rung=args.rung,
        deadline_s=args.deadline_seconds or DEADLINES[args.rung],
        require_loss=not args.baseline and args.rung == "R0",
        check_determinism=args.rung == "R0",
        baseline_scores=None if paired is None else paired.tolist(),
        tags=tags,
    )
    if result.traceback:
        _record_candidate_failure(
            args,
            model_path,
            catalog,
            schema,
            {
                "exception_type": "CandidateError",
                "message": result.reason or "",
                "traceback": result.traceback,
                "rung": args.rung,
                "node_id": args.node_id or args.candidate_id,
                "run_id": args.run_id,
            },
        )
    if result.status == "ok" and keyed is not None:
        from harness.evaluate import paired_day_blocked_delta

        second = paired_day_blocked_delta(
            evaluation,
            np.asarray(ranker.score(evaluation.without_labels()), dtype=float),
            keyed,
            seed=args.seed,
        )
        result.metrics["delta_lgbm_point"] = second["point"]
        result.metrics["delta_lgbm_ci_lo"] = second["ci_lo"]
        result.metrics["delta_lgbm_ci_hi"] = second["ci_hi"]
    if result.status == "ok" and args.write_scores:
        write_scores(
            Path(args.write_scores),
            evaluation,
            np.asarray(ranker.score(evaluation.without_labels()), dtype=float),
        )
    if str(args.shared_cluster).lower() in {"1", "true", "yes"}:
        result.tags["shared_cluster"] = "true"
    if requirements_sha:
        result.metrics["install_s"] = round(install_s, 4)
        result.tags["requirements_sha"] = requirements_sha
    logged = log_mlflow(
        result,
        args.experiment_name or None,
        ranker=ranker,
        baseline=args.baseline or "",
        model_path=None if args.baseline else model_path,
        train=train,
        evaluation=evaluation,
        lineage=lineage,
        rung=args.rung,
        family=args.family,
        node_id=args.node_id or args.candidate_id,
        run_id=args.run_id or args.batch,
        seed=args.seed,
        git_sha=args.git_sha,
        eval_split=eval_split,
    )
    payload = {
        "status": result.status,
        "reason": result.reason,
        "metrics": result.metrics,
        "mlflow_run_id": logged.get("mlflow_run_id"),
        "model_uri": logged.get("model_uri"),
        "native_model_uri": logged.get("native_model_uri"),
        "native_flavor": logged.get("native_flavor"),
    }
    print(json.dumps(payload))
    if result.status == "ok" and args.confirmation and args.confirm_token:
        token_path = Path(args.confirm_token)
        if token_path.is_file():
            token_path.unlink()
    if result.status not in {"ok", "unsupported"}:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
