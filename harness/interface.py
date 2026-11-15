"""Ranker protocol and the frame candidates are allowed to see.

``score`` receives a frame whose labels have been removed. ``FitContext``
accepts training logs only; evaluation metric names are rejected.

``RankingFrame`` stores one split as a pandas frame sorted by ``query_id``,
``item_id``. ``rows`` is a lazy list of dicts for callers that still walk
records. Numeric work goes through ``numeric_matrix``.
"""

from __future__ import annotations

import time
from typing import Any, Protocol

import numpy as np
import pandas as pd

from harness.data.contract import AUX_PREFIX, FEAT_PREFIX

_BANNED_LOG_TOKENS = ("ndcg", "mrr", "coverage", "holdout", "label")


class Ranker(Protocol):
    def fit(self, train: RankingFrame, valid: RankingFrame, ctx: FitContext) -> None: ...

    def score(self, frame: RankingFrame) -> np.ndarray: ...

    def save(self, path: str) -> None: ...

    @classmethod
    def load(cls, path: str) -> Ranker: ...


class MetricLogger(Protocol):
    def __call__(self, key: str, value: float) -> None: ...


class FitContext:
    """Seed, deadline, device, and a logger that only accepts training metrics."""

    def __init__(
        self,
        seed: int,
        deadline: float,
        device: str = "cpu",
        logger: MetricLogger | None = None,
    ) -> None:
        self.seed = seed
        self.deadline = deadline
        self.device = device
        self.history: dict[str, list[float]] = {}
        self._logger = logger

    def log(self, key: str, value: float) -> None:
        lowered = key.lower()
        if any(token in lowered for token in _BANNED_LOG_TOKENS):
            raise RuntimeError(f"fit cannot log evaluation metric {key}")
        numeric = float(value)
        self.history.setdefault(key, []).append(numeric)
        if self._logger is not None:
            self._logger(key, numeric)

    def expired(self) -> bool:
        return time.monotonic() > self.deadline

    def loss_decreased(self) -> bool:
        losses = self.history.get("train_loss") or []
        return len(losses) >= 2 and losses[-1] < losses[0]


class RankingFrame:
    """Columnar view of one contract-shaped split. Rows are grouped by query."""

    def __init__(self, rows: list[dict[str, Any]]) -> None:
        self._frame = _sort_frame(pd.DataFrame(rows))
        self._rows: list[dict[str, Any]] | None = None

    def __len__(self) -> int:
        return int(len(self._frame))

    @property
    def columns(self) -> set[str]:
        return set(self._frame.columns)

    def column_names(self) -> set[str]:
        return set(self.columns)

    @property
    def rows(self) -> list[dict[str, Any]]:
        """Records in frame order. Built on first access and then reused."""

        if self._rows is None:
            self._rows = [_record(row) for row in self._frame.to_dict(orient="records")]
        return self._rows

    @property
    def item_id(self) -> list[Any]:
        if "item_id" not in self._frame.columns:
            return [None] * len(self)
        return [_python(value) for value in self._frame["item_id"].tolist()]

    def to_pandas(self) -> pd.DataFrame:
        """Copy of the sorted frame. Mutations do not change this object."""

        return self._frame.copy()

    def labels(self) -> np.ndarray:
        if "label" not in self._frame.columns:
            raise KeyError("label")
        values = pd.to_numeric(self._frame["label"], errors="coerce")
        return values.to_numpy(dtype=float, na_value=np.nan)

    def numeric_matrix(self, columns: list[str] | None = None) -> tuple[np.ndarray, list[str]]:
        """Numeric columns as a float array. Missing values, including NaN, are 0.

        ``columns`` selects and orders the columns. A name that is absent is an
        all-zero column, so a model can score a frame that dropped a feature.
        """

        names = list(columns) if columns is not None else self.numeric_feature_names()
        width = len(names)
        if width == 0:
            return np.zeros((len(self), 0), dtype=float), []
        aligned = pd.DataFrame(
            {name: _numeric_series(self._frame, name) for name in names},
            index=self._frame.index,
        )
        return aligned.to_numpy(dtype=float, na_value=0.0), names

    def column(self, name: str) -> list[Any]:
        if name not in self._frame.columns:
            raise KeyError(name)
        return [_python(value) for value in self._frame[name].tolist()]

    def __getitem__(self, name: str) -> list[Any]:
        return self.column(name)

    def without_labels(self) -> RankingFrame:
        drop = [
            name
            for name in self._frame.columns
            if name == "label" or str(name).startswith(AUX_PREFIX)
        ]
        return RankingFrame.from_pandas(self._frame.drop(columns=drop))

    def groups(self) -> list[tuple[int, int, Any]]:
        if self._frame.empty or "query_id" not in self._frame.columns:
            return []
        ids = self._frame["query_id"].to_numpy()
        if len(ids) == 0:
            return []
        change = np.flatnonzero(ids[1:] != ids[:-1]) + 1
        starts = np.concatenate((np.array([0]), change))
        ends = np.concatenate((change, np.array([len(ids)])))
        spans = zip(starts, ends, strict=True)
        return [(_as_int(start), _as_int(end), _python(ids[start])) for start, end in spans]

    def group_sizes(self) -> list[int]:
        return [end - start for start, end, _query_id in self.groups()]

    def numeric_feature_names(self) -> list[str]:
        names = []
        for name in sorted(self._frame.columns, key=str):
            if not str(name).startswith(FEAT_PREFIX):
                continue
            if _numeric_column(self._frame[name]):
                names.append(str(name))
        return names

    def categorical_feature_names(self) -> list[str]:
        """``feat__`` columns that carry strings rather than numbers."""

        numeric = set(self.numeric_feature_names())
        names = []
        for name in sorted(self._frame.columns, key=str):
            text = str(name)
            if not text.startswith(FEAT_PREFIX) or text in numeric:
                continue
            if _has_value(self._frame[name]):
                names.append(text)
        return names

    @classmethod
    def from_rows(cls, rows: list[dict[str, Any]]) -> RankingFrame:
        return cls(rows)

    @classmethod
    def from_pandas(cls, frame: pd.DataFrame) -> RankingFrame:
        """Wrap a frame that is already the split, and sort it."""

        obj = cls.__new__(cls)
        obj._frame = _sort_frame(frame.reset_index(drop=True))
        obj._rows = None
        return obj


def _sort_frame(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty or "query_id" not in frame.columns or "item_id" not in frame.columns:
        return frame.reset_index(drop=True)
    ordered = frame.sort_values(
        by=["query_id", "item_id"],
        key=lambda column: column.map(_sort_token),
        kind="mergesort",
    )
    return ordered.reset_index(drop=True)


def _sort_token(value: Any) -> str:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return ""
    return str(value)


def _numeric_series(frame: pd.DataFrame, name: str) -> pd.Series:
    if name not in frame.columns:
        return pd.Series(np.zeros(len(frame)), index=frame.index, dtype=float)
    return pd.to_numeric(frame[name], errors="coerce")


def _numeric_column(series: pd.Series) -> bool:
    """True when every present value is a number. An all-null column is not."""

    if series.empty:
        return False
    if pd.api.types.is_bool_dtype(series) or pd.api.types.is_numeric_dtype(series):
        return bool(series.notna().any())
    present = series.notna()
    if not bool(present.any()):
        return False
    coerced = pd.to_numeric(series, errors="coerce")
    failed = present & coerced.isna()
    return not bool(failed.any())


def _has_value(series: pd.Series) -> bool:
    return bool(series.notna().any())


def _python(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, float) and np.isnan(value):
        return None
    if isinstance(value, np.generic):
        item = value.item()
        if isinstance(item, float) and np.isnan(item):
            return None
        return item
    return value


def _record(row: dict[str, Any]) -> dict[str, Any]:
    return {key: _python(value) for key, value in row.items()}


def _as_int(value: Any) -> int:
    return int(value)
