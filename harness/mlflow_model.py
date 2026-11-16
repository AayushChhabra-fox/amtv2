"""Log an R3 ranker as an end-to-end pyfunc plus a native flavor when one exists.

``runs:/<id>/model`` scores a raw split DataFrame through ``Ranker.score``.
``runs:/<id>/native_model`` is the inner LightGBM, XGBoost, CatBoost, PyTorch,
or scikit-learn estimator when the ranker wraps exactly one of those.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import tempfile
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

import mlflow.pyfunc
import numpy as np
import pandas as pd

from harness.data.contract import AUX_PREFIX
from harness.deps import parse_requirements
from harness.interface import RankingFrame, _sort_token
from harness.registry import BASE_PACKAGES

_NATIVE_NONE = "none"
_DATASET_CONTEXTS = {
    "search_val": "validation",
    "select_holdout": "testing",
    "final_holdout": "testing",
}


def should_log_r3_model(rung: str, baseline: str, status: str) -> bool:
    """Candidates and the LightGBM reference log weights at R3 only."""

    return status == "ok" and rung == "R3" and baseline in {"", "lgbm_reference"}


class AMTRankerPyFunc(mlflow.pyfunc.PythonModel):
    """MLflow pyfunc that restores caller row order after the contract sort."""

    def load_context(self, context) -> None:
        meta_path = Path(context.artifacts["meta"])
        self.meta = json.loads(meta_path.read_text(encoding="utf-8"))
        state = context.artifacts["ranker_state"]
        kind = str(self.meta.get("kind") or "")
        if kind == "lgbm_reference":
            from harness.baselines.lgbm_reference import LGBMReferenceRanker

            self.ranker = LGBMReferenceRanker.load(state)
            return
        model_py = Path(context.artifacts["model_py"])
        ranker_cls = _load_ranker_class(model_py)
        self.ranker = ranker_cls.load(state)

    def predict(self, context, model_input, params: dict | None = None):
        del context, params
        frame = model_input if isinstance(model_input, pd.DataFrame) else pd.DataFrame(model_input)
        drop = [
            name
            for name in frame.columns
            if name == "label" or str(name).startswith(AUX_PREFIX)
        ]
        clean = frame.drop(columns=drop).reset_index(drop=True)
        if "query_id" in clean.columns and "item_id" in clean.columns:
            ordered = clean.sort_values(
                by=["query_id", "item_id"],
                key=lambda column: column.map(_sort_token),
                kind="mergesort",
            )
            positions = ordered.index.to_numpy()
        else:
            positions = np.arange(len(clean))
        scored = np.asarray(
            self.ranker.score(RankingFrame.from_pandas(clean.iloc[positions]).without_labels()),
            dtype=float,
        )
        restored = np.empty(len(scored), dtype=float)
        restored[positions] = scored
        return restored


def log_r3_models(
    *,
    ranker,
    baseline: str,
    model_path: Path | None,
    train,
    evaluation,
    lineage,
    family: str,
    node_id: str,
    run_id: str,
    seed: int,
    git_sha: str,
    eval_split: str,
) -> dict[str, str | None]:
    """Log the pyfunc, the native estimator when present, and the dataset version."""

    import mlflow

    os.environ.setdefault("MLFLOW_LOG_UV_FILES", "false")
    kind = "lgbm_reference" if baseline == "lgbm_reference" else "candidate"
    requirements = _pip_requirements(model_path)
    native_flavor, native_logged = log_native_estimator(ranker, requirements, model_path)
    logged = _log_pyfunc(
        ranker=ranker,
        kind=kind,
        model_path=model_path,
        evaluation=evaluation,
        lineage=lineage,
        family=family,
        node_id=node_id,
        run_id=run_id,
        seed=seed,
        git_sha=git_sha,
        native_flavor=native_flavor,
        requirements=requirements,
    )
    log_dataset_inputs(lineage, train, evaluation, eval_split)
    mlflow.log_dict(lineage.as_dict(), "dataset_version.json")
    active = mlflow.active_run()
    mlflow_run_id = None if active is None else active.info.run_id
    tags = {
        "model_artifact_path": "model",
        "native_flavor": native_flavor,
        **lineage.tags(),
    }
    if native_logged:
        tags["native_model_artifact_path"] = "native_model"
    mlflow.set_tags(tags)
    model_uri = None if mlflow_run_id is None else f"runs:/{mlflow_run_id}/model"
    native_uri = (
        f"runs:/{mlflow_run_id}/native_model" if mlflow_run_id and native_logged else None
    )
    return {
        "mlflow_run_id": mlflow_run_id,
        "model_uri": model_uri,
        "native_model_uri": native_uri,
        "native_flavor": native_flavor,
        "logged_pyfunc": logged,
    }


def _call_log(function, **kwargs):
    """Accept MLflow 2 ``artifact_path`` and MLflow 3 skops trust lists."""

    try:
        return _invoke(function, kwargs)
    except Exception as exc:
        trusted = _untrusted_types(exc)
        if not trusted or kwargs.get("skops_trusted_types"):
            raise
        retried = dict(kwargs)
        retried["skops_trusted_types"] = trusted
        return _invoke(function, retried)


def _invoke(function, kwargs: dict):
    try:
        return function(**kwargs)
    except TypeError:
        if "artifact_path" not in kwargs:
            raise
        renamed = dict(kwargs)
        renamed["name"] = renamed.pop("artifact_path")
        return function(**renamed)


def _untrusted_types(exc: BaseException) -> list[str]:
    text = str(exc)
    if "Untrusted types" not in text:
        return []
    return re.findall(r"'([^']+)'", text)


def log_native_estimator(
    ranker, pip_requirements: list[str], model_path: Path | None
) -> tuple[str, bool]:
    """Log one native flavor. Ensembles and custom rankers stay pyfunc-only."""

    flavor, estimator = extract_native_estimator(ranker)
    if flavor == _NATIVE_NONE or estimator is None:
        return _NATIVE_NONE, False
    import mlflow

    common: dict[str, Any] = {
        "artifact_path": "native_model",
        "pip_requirements": pip_requirements,
    }
    if flavor in {"lightgbm", "sklearn"}:
        common["skops_trusted_types"] = [
            f"{type(estimator).__module__}.{type(estimator).__qualname__}"
        ]
    if flavor == "pytorch":
        common["serialization_format"] = "pickle"
    if flavor == "lightgbm":
        _call_log(mlflow.lightgbm.log_model, lgb_model=estimator, **common)
    elif flavor == "xgboost":
        _call_log(mlflow.xgboost.log_model, xgb_model=estimator, **common)
    elif flavor == "catboost":
        _call_log(mlflow.catboost.log_model, cb_model=estimator, **common)
    elif flavor == "pytorch":
        if model_path is not None and model_path.is_file():
            common["code_paths"] = [str(model_path)]
        _call_log(mlflow.pytorch.log_model, pytorch_model=estimator, **common)
    elif flavor == "sklearn":
        _call_log(mlflow.sklearn.log_model, sk_model=estimator, **common)
    else:
        return _NATIVE_NONE, False
    columns = _feature_columns(ranker)
    if columns:
        mlflow.log_dict(columns, "feature_columns.json")
    return flavor, True


def extract_native_estimator(ranker) -> tuple[str, Any]:
    """Return ``(flavor, estimator)`` when the ranker holds exactly one native model."""

    found: list[tuple[str, Any]] = []
    seen: set[int] = set()
    for value in vars(ranker).values():
        if value is None or id(value) in seen:
            continue
        flavor = _flavor_of(value)
        if not flavor:
            continue
        seen.add(id(value))
        found.append((flavor, value))
    if len(found) != 1:
        return _NATIVE_NONE, None
    return found[0]


def log_dataset_inputs(lineage, train_frame, eval_frame, eval_split: str) -> None:
    """Attach train and eval inputs. Spark uses the Delta version; local uses a sample."""

    import mlflow

    eval_context = _DATASET_CONTEXTS.get(eval_split, "validation")
    _log_one_input(lineage.train, train_frame, "training")
    _log_one_input(lineage.evaluation, eval_frame, eval_context)
    mlflow.set_tags(lineage.tags())


def _log_one_input(split, frame, context: str) -> None:
    import mlflow

    dataset = _spark_dataset(split)
    if dataset is None:
        sample = frame.to_pandas().head(20)
        dataset = mlflow.data.from_pandas(
            sample,
            name=split.table or split.name,
            digest=split.content_hash or None,
        )
    mlflow.log_input(dataset, context=context)


def _spark_dataset(split):
    if not split.table or split.storage == "jsonl":
        return None
    try:
        from pyspark.sql import SparkSession
    except ImportError:
        return None
    spark = SparkSession.getActiveSession()
    if spark is None:
        return None
    import mlflow

    table = split.table.replace("`", "")
    try:
        reader = spark.read
        if split.delta_version is not None:
            reader = reader.option("versionAsOf", str(split.delta_version))
        frame = reader.table(table)
        version = None if split.delta_version is None else str(split.delta_version)
        return mlflow.data.from_spark(frame, table_name=table, version=version)
    except Exception:
        return None


def _log_pyfunc(
    *,
    ranker,
    kind: str,
    model_path: Path | None,
    evaluation,
    lineage,
    family: str,
    node_id: str,
    run_id: str,
    seed: int,
    git_sha: str,
    native_flavor: str,
    requirements: list[str],
) -> bool:
    import mlflow
    from mlflow.models import infer_signature

    from harness.train import _bundle_root

    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        state = root / "ranker_state"
        ranker.save(str(state))
        meta = {
            "kind": kind,
            "family": family,
            "node_id": node_id,
            "run_id": run_id,
            "seed": seed,
            "git_sha": git_sha,
            "native_flavor": native_flavor,
            "dataset_lineage": lineage.as_dict(),
        }
        meta_path = root / "meta.json"
        meta_path.write_text(json.dumps(meta, default=str), encoding="utf-8")
        artifacts = {"ranker_state": str(state), "meta": str(meta_path)}
        if kind == "candidate" and model_path is not None and model_path.is_file():
            copied = root / "model.py"
            shutil.copy2(model_path, copied)
            artifacts["model_py"] = str(copied)
        sample = evaluation.without_labels().to_pandas().head(5)
        hidden = RankingFrame.from_pandas(sample)
        prediction = np.asarray(ranker.score(hidden.without_labels()), dtype=float)
        signature = infer_signature(sample, prediction)
        code_root = root / "code"
        shutil.copytree(
            _bundle_root() / "harness",
            code_root / "harness",
            ignore=shutil.ignore_patterns("__pycache__", "*.pyc"),
        )
        _call_log(
            mlflow.pyfunc.log_model,
            artifact_path="model",
            python_model=AMTRankerPyFunc(),
            artifacts=artifacts,
            code_paths=[str(code_root)],
            pip_requirements=requirements,
            signature=signature,
            input_example=sample,
        )
    return True


def _pip_requirements(model_path: Path | None) -> list[str]:
    pins: list[str] = []
    for name in sorted(BASE_PACKAGES):
        try:
            pins.append(f"{name}=={version(name)}")
        except PackageNotFoundError:
            continue
    if model_path is None:
        return pins
    requirements = model_path.parent / "requirements.txt"
    if not requirements.is_file():
        return pins
    try:
        extra = parse_requirements(requirements)
    except Exception:
        return pins
    pins.extend(f"{item.name}=={item.version}" for item in extra)
    return pins


def _feature_columns(ranker) -> dict[str, list[str]]:
    found: dict[str, list[str]] = {}
    for name in ("columns", "numeric_columns", "categorical_columns", "feature_names"):
        value = getattr(ranker, name, None)
        if isinstance(value, list) and value:
            found[name] = [str(item) for item in value]
    return found


def _flavor_of(value: Any) -> str:
    try:
        import lightgbm as lgb

        if isinstance(value, (lgb.Booster, lgb.LGBMModel)):
            return "lightgbm"
    except Exception:
        pass
    try:
        import xgboost as xgb

        if isinstance(value, (xgb.Booster, xgb.XGBModel)):
            return "xgboost"
    except Exception:
        pass
    try:
        import catboost

        if isinstance(value, catboost.CatBoost):
            return "catboost"
    except Exception:
        pass
    try:
        import torch

        if isinstance(value, torch.nn.Module):
            return "pytorch"
    except Exception:
        pass
    try:
        import sklearn.base

        if isinstance(value, sklearn.base.BaseEstimator):
            return "sklearn"
    except Exception:
        pass
    return ""


def _load_ranker_class(model_path: Path):
    import importlib.util

    spec = importlib.util.spec_from_file_location("amt_logged_ranker", model_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot import {model_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if not hasattr(module, "Ranker"):
        raise ImportError(f"{model_path} does not define Ranker")
    return module.Ranker
