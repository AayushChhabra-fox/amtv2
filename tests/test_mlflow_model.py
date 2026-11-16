"""R3 pyfunc and native-flavor logging, plus dataset version capture."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import mlflow
import numpy as np
import pandas as pd
import pytest

from amt.schemas import LedgerRow, export_json_schemas, json_schema_dir
from amt.scores import manifest_hash
from harness.data.build_dataset import _delta_history_head
from harness.interface import FitContext, RankingFrame, _sort_token
from harness.mlflow_model import extract_native_estimator, log_r3_models, should_log_r3_model
from harness.train import DatasetLineage, SplitLineage, read_split_frame


def test_should_log_only_r3_candidates_and_lgbm() -> None:
    assert should_log_r3_model("R3", "", "ok") is True
    assert should_log_r3_model("R3", "lgbm_reference", "ok") is True
    assert should_log_r3_model("R2", "", "ok") is False
    assert should_log_r3_model("R3", "random", "ok") is False
    assert should_log_r3_model("R3", "popularity", "ok") is False
    assert should_log_r3_model("R3", "reference_score", "ok") is False
    assert should_log_r3_model("R3", "lgbm_full", "ok") is False
    assert should_log_r3_model("R3", "", "failed") is False


def test_delta_history_head_reads_version() -> None:
    class _Executor:
        def execute(self, sql: str) -> list[dict]:
            assert sql.startswith("DESCRIBE HISTORY")
            return [{"version": "4", "timestamp": "2026-01-01 00:00:00"}]

    assert _delta_history_head(_Executor(), "`cat`.`sch`.`gold_train`") == {
        "version": 4,
        "timestamp": "2026-01-01 00:00:00",
    }

    class _Missing:
        def execute(self, sql: str) -> list[dict]:
            raise RuntimeError("not a delta table")

    assert _delta_history_head(_Missing(), "src") == {}


def test_read_split_frame_pins_delta_version(tmp_path: Path) -> None:
    class _Frame:
        def toPandas(self) -> pd.DataFrame:
            return pd.DataFrame({"query_id": ["q1"], "item_id": ["a"]})

    class _Spark:
        def __init__(self) -> None:
            self.sqls: list[str] = []

        def sql(self, text: str) -> _Frame:
            self.sqls.append(text)
            if "VERSION AS OF" not in text:
                raise AssertionError(text)
            return _Frame()

    spark = _Spark()
    frame = read_split_frame(
        spark,
        "`cat`.`sch`.`gold_train`",
        None,
        tmp_path / "missing.parquet",
        delta_version=7,
    )
    assert frame["query_id"].tolist() == ["q1"]
    assert "VERSION AS OF 7" in spark.sqls[0]


def test_manifest_hash_reads_the_ready_gate(tmp_path: Path) -> None:
    ready = tmp_path / "gates" / "ready" / "nfy_demo_v3"
    ready.mkdir(parents=True)
    (ready / "manifest.json").write_text(
        json.dumps({"dataset": "nfy_article_rerank_gold", "spec_hash": "abc123"}),
        encoding="utf-8",
    )
    assert manifest_hash(tmp_path, "nfy_article_rerank_gold") == "abc123"
    assert manifest_hash(tmp_path, "other") == "unhashed"


def test_ledger_row_accepts_a_row_without_model_fields() -> None:
    row = LedgerRow.model_validate(
        {
            "run_id": "toy",
            "node_id": "c01",
            "operator": "draft",
            "family": "linear",
            "claim": "ridge",
            "rung": "R1",
            "seed": 0,
            "status": "kept",
            "git_sha": "abc",
        }
    )
    assert row.model_uri is None
    assert row.dataset_version is None


@pytest.mark.parametrize(
    "flavor",
    ["lightgbm", "xgboost", "catboost", "sklearn", "pytorch", "numpy", "ensemble"],
)
def test_r3_logging_round_trip(tmp_path: Path, flavor: str) -> None:
    mlflow.set_tracking_uri(f"sqlite:///{tmp_path / 'mlflow.db'}")
    mlflow.set_experiment("amt-r3")
    ranker, baseline, model_path = _fit(tmp_path, flavor)
    train, evaluation = _frames()
    lineage = _lineage()
    shuffled = evaluation.to_pandas().sample(frac=1, random_state=1).reset_index(drop=True)
    shuffled["label"] = 0
    with mlflow.start_run():
        logged = log_r3_models(
            ranker=ranker,
            baseline=baseline,
            model_path=model_path,
            train=train,
            evaluation=evaluation,
            lineage=lineage,
            family=flavor,
            node_id="c01",
            run_id="toy",
            seed=0,
            git_sha="abc",
            eval_split="search_val",
        )
        run = mlflow.active_run()
        assert run is not None
        run_id = run.info.run_id
    loaded = mlflow.pyfunc.load_model(str(logged["model_uri"]))
    predicted = np.asarray(loaded.predict(shuffled), dtype=float)
    expected = _scores_in_input_order(ranker, shuffled.drop(columns=["label"]))
    assert predicted.shape == expected.shape
    assert np.allclose(predicted, expected)
    if flavor in {"numpy", "ensemble"}:
        assert logged["native_flavor"] == "none"
        assert logged["native_model_uri"] is None
        assert extract_native_estimator(ranker)[0] == "none"
    else:
        assert logged["native_flavor"] == flavor
        assert _load_native(flavor, str(logged["native_model_uri"])) is not None
        columns = mlflow.artifacts.load_dict(f"runs:/{run_id}/feature_columns.json")
        assert columns
    client = mlflow.MlflowClient()
    stored = client.get_run(run_id)
    assert stored.data.tags["dataset_spec_hash"] == "spec-hash"
    assert stored.data.tags["train_delta_version"] == "3"
    assert stored.data.tags["eval_delta_version"] == "4"
    assert stored.data.tags["model_artifact_path"] == "model"
    dataset = mlflow.artifacts.load_dict(f"runs:/{run_id}/dataset_version.json")
    assert dataset["spec_hash"] == "spec-hash"
    assert stored.inputs.dataset_inputs


def test_exported_schemas_include_model_uri() -> None:
    export_json_schemas(json_schema_dir())
    payload = json.loads((json_schema_dir() / "ledger_row.schema.json").read_text(encoding="utf-8"))
    assert "model_uri" in payload["properties"]
    assert "native_model_uri" in payload["properties"]
    assert "dataset_version" in payload["properties"]


def _load_native(flavor: str, uri: str):
    loaders = {
        "lightgbm": mlflow.lightgbm.load_model,
        "xgboost": mlflow.xgboost.load_model,
        "catboost": mlflow.catboost.load_model,
        "sklearn": mlflow.sklearn.load_model,
        "pytorch": mlflow.pytorch.load_model,
    }
    return loaders[flavor](uri)


def _scores_in_input_order(ranker, frame: pd.DataFrame) -> np.ndarray:
    ordered = frame.sort_values(
        by=["query_id", "item_id"],
        key=lambda column: column.map(_sort_token),
        kind="mergesort",
    )
    positions = ordered.index.to_numpy()
    scored = np.asarray(
        ranker.score(RankingFrame.from_pandas(frame.iloc[positions]).without_labels()),
        dtype=float,
    )
    restored = np.empty(len(scored), dtype=float)
    restored[positions] = scored
    return restored


def _frames() -> tuple[RankingFrame, RankingFrame]:
    rows = []
    for query_id, labels in (("q1", (3, 0)), ("q2", (2, 1))):
        for index, label in enumerate(labels):
            rows.append(
                {
                    "query_id": query_id,
                    "item_id": f"{query_id}-{index}",
                    "label": label,
                    "query_ts": "2026-01-01T00:00:00",
                    "event_day": "2026-01-01",
                    "feat__num__x": float(label),
                    "feat__num__y": float(index),
                }
            )
    frame = RankingFrame(rows)
    return frame, frame


def _lineage() -> DatasetLineage:
    return DatasetLineage(
        dataset="gold",
        spec_hash="spec-hash",
        train=SplitLineage(
            name="train",
            table="`cat`.`sch`.`gold_train`",
            delta_version=3,
            content_hash="train-hash",
            query_ids_hash="full",
            storage="jsonl",
        ),
        evaluation=SplitLineage(
            name="search_val",
            table="`cat`.`sch`.`gold_search_val`",
            delta_version=4,
            content_hash="eval-hash",
            query_ids_hash="full",
            storage="jsonl",
        ),
    )


def _fit(tmp_path: Path, flavor: str):
    train, _evaluation = _frames()
    context = FitContext(seed=0, deadline=1e12)
    if flavor == "lightgbm":
        from harness.baselines.lgbm_reference import LGBMReferenceRanker

        ranker = LGBMReferenceRanker(n_estimators=4)
        ranker.fit(train, train, context)
        return ranker, "lgbm_reference", None
    path = tmp_path / flavor / "model.py"
    path.parent.mkdir(parents=True)
    path.write_text(_MODEL[flavor], encoding="utf-8")
    spec = importlib.util.spec_from_file_location(f"ranker_{flavor}", path)
    module = importlib.util.module_from_spec(spec)
    assert spec is not None and spec.loader is not None
    spec.loader.exec_module(module)
    ranker = module.Ranker()
    ranker.fit(train, train, context)
    return ranker, "", path


_MODEL = {
    "numpy": """
import pickle

class Ranker:
    def fit(self, train, valid, ctx):
        del valid, ctx
        self.columns = train.numeric_feature_names()
    def score(self, frame):
        matrix, _ = frame.numeric_matrix(self.columns)
        return matrix[:, 0]
    def save(self, path):
        with open(path, "wb") as handle:
            pickle.dump({"columns": self.columns}, handle)
    @classmethod
    def load(cls, path):
        ranker = cls()
        with open(path, "rb") as handle:
            ranker.columns = pickle.load(handle)["columns"]
        return ranker
""",
    "sklearn": """
import pickle
from sklearn.linear_model import Ridge

class Ranker:
    def __init__(self):
        self.model = Ridge(alpha=1.0)
        self.columns = []
    def fit(self, train, valid, ctx):
        del valid, ctx
        matrix, self.columns = train.numeric_matrix()
        self.model.fit(matrix, train.labels())
    def score(self, frame):
        matrix, _ = frame.numeric_matrix(self.columns)
        return self.model.predict(matrix)
    def save(self, path):
        with open(path, "wb") as handle:
            pickle.dump({"columns": self.columns, "model": self.model}, handle)
    @classmethod
    def load(cls, path):
        ranker = cls()
        with open(path, "rb") as handle:
            payload = pickle.load(handle)
        ranker.columns = payload["columns"]
        ranker.model = payload["model"]
        return ranker
""",
    "xgboost": """
import pickle
import xgboost as xgb

class Ranker:
    def __init__(self):
        self.model = xgb.XGBRegressor(n_estimators=2, verbosity=0)
        self.columns = []
    def fit(self, train, valid, ctx):
        del valid, ctx
        matrix, self.columns = train.numeric_matrix()
        self.model.fit(matrix, train.labels())
    def score(self, frame):
        matrix, _ = frame.numeric_matrix(self.columns)
        return self.model.predict(matrix)
    def save(self, path):
        with open(path, "wb") as handle:
            pickle.dump({"columns": self.columns, "model": self.model}, handle)
    @classmethod
    def load(cls, path):
        ranker = cls()
        with open(path, "rb") as handle:
            payload = pickle.load(handle)
        ranker.columns = payload["columns"]
        ranker.model = payload["model"]
        return ranker
""",
    "catboost": """
import pickle
from catboost import CatBoostRegressor

class Ranker:
    def __init__(self):
        self.model = CatBoostRegressor(iterations=2, verbose=False, allow_writing_files=False)
        self.columns = []
    def fit(self, train, valid, ctx):
        del valid, ctx
        matrix, self.columns = train.numeric_matrix()
        self.model.fit(matrix, train.labels())
    def score(self, frame):
        matrix, _ = frame.numeric_matrix(self.columns)
        return self.model.predict(matrix)
    def save(self, path):
        with open(path, "wb") as handle:
            pickle.dump({"columns": self.columns, "model": self.model}, handle)
    @classmethod
    def load(cls, path):
        ranker = cls()
        with open(path, "rb") as handle:
            payload = pickle.load(handle)
        ranker.columns = payload["columns"]
        ranker.model = payload["model"]
        return ranker
""",
    "pytorch": """
import pickle
import torch

class Ranker:
    def fit(self, train, valid, ctx):
        del valid, ctx
        matrix, self.columns = train.numeric_matrix()
        self.model = torch.nn.Linear(matrix.shape[1], 1)
        with torch.no_grad():
            self.model.weight.fill_(1.0)
            self.model.bias.fill_(0.0)
    def score(self, frame):
        matrix, _ = frame.numeric_matrix(self.columns)
        with torch.no_grad():
            values = self.model(torch.tensor(matrix, dtype=torch.float32)).reshape(-1)
        return values.numpy()
    def save(self, path):
        with open(path, "wb") as handle:
            pickle.dump({"columns": self.columns, "state": self.model.state_dict()}, handle)
    @classmethod
    def load(cls, path):
        ranker = cls()
        with open(path, "rb") as handle:
            payload = pickle.load(handle)
        ranker.columns = payload["columns"]
        ranker.model = torch.nn.Linear(len(ranker.columns), 1)
        ranker.model.load_state_dict(payload["state"])
        return ranker
""",
    "ensemble": """
import pickle
from sklearn.linear_model import Ridge

class Ranker:
    def __init__(self):
        self.left = Ridge()
        self.right = Ridge()
        self.columns = []
    def fit(self, train, valid, ctx):
        del valid, ctx
        matrix, self.columns = train.numeric_matrix()
        labels = train.labels()
        self.left.fit(matrix, labels)
        self.right.fit(matrix, labels)
    def score(self, frame):
        matrix, _ = frame.numeric_matrix(self.columns)
        return 0.5 * (self.left.predict(matrix) + self.right.predict(matrix))
    def save(self, path):
        with open(path, "wb") as handle:
            pickle.dump({"columns": self.columns, "left": self.left, "right": self.right}, handle)
    @classmethod
    def load(cls, path):
        ranker = cls()
        with open(path, "rb") as handle:
            payload = pickle.load(handle)
        ranker.columns = payload["columns"]
        ranker.left = payload["left"]
        ranker.right = payload["right"]
        return ranker
""",
}
