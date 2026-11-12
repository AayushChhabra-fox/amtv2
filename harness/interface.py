"""Ranker protocol and the frame candidates are allowed to see.

``score`` receives a frame whose labels have been removed. ``FitContext``
accepts training logs only; evaluation metric names are rejected.
"""

from __future__ import annotations

import time
from typing import Any, Protocol

import numpy as np

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
        self.rows = sorted(rows, key=lambda row: (str(row["query_id"]), str(row["item_id"])))
        self.columns = set().union(*(row.keys() for row in self.rows)) if self.rows else set()

    def __len__(self) -> int:
        return len(self.rows)

    def column_names(self) -> set[str]:
        return set(self.columns)

    @property
    def item_id(self) -> list[Any]:
        return [row.get("item_id") for row in self.rows]

    def labels(self) -> np.ndarray:
        if "label" not in self.columns:
            raise KeyError("label")
        return np.asarray([row["label"] for row in self.rows], dtype=float)

    def numeric_matrix(self) -> tuple[np.ndarray, list[str]]:
        names = self.numeric_feature_names()
        matrix = np.zeros((len(self), len(names)), dtype=float)
        for column_index, name in enumerate(names):
            for row_index, row in enumerate(self.rows):
                raw = row.get(name)
                if raw is None:
                    continue
                matrix[row_index, column_index] = float(raw)
        return matrix, names

    def column(self, name: str) -> list[Any]:
        if name not in self.columns and name not in {"query_id", "item_id", "label"}:
            raise KeyError(name)
        if name not in self.columns:
            raise KeyError(name)
        return [row.get(name) for row in self.rows]

    def __getitem__(self, name: str) -> list[Any]:
        return self.column(name)

    def without_labels(self) -> RankingFrame:
        hidden: list[dict[str, Any]] = []
        for row in self.rows:
            copied = {
                key: value
                for key, value in row.items()
                if key != "label" and not key.startswith(AUX_PREFIX)
            }
            hidden.append(copied)
        return RankingFrame(hidden)

    def groups(self) -> list[tuple[int, int, Any]]:
        if not self.rows:
            return []
        spans: list[tuple[int, int, Any]] = []
        start = 0
        current = self.rows[0]["query_id"]
        for index, row in enumerate(self.rows):
            if row["query_id"] != current:
                spans.append((start, index, current))
                start = index
                current = row["query_id"]
        spans.append((start, len(self.rows), current))
        return spans

    def group_sizes(self) -> list[int]:
        return [end - start for start, end, _query_id in self.groups()]

    def numeric_feature_names(self) -> list[str]:
        names = []
        for name in sorted(self.columns):
            if not name.startswith(FEAT_PREFIX):
                continue
            if _numeric_column(self.rows, name):
                names.append(name)
        return names

    @classmethod
    def from_rows(cls, rows: list[dict[str, Any]]) -> RankingFrame:
        return cls(rows)


def _numeric_column(rows: list[dict[str, Any]], name: str) -> bool:
    seen = False
    for row in rows:
        value = row.get(name)
        if value is None:
            continue
        seen = True
        if isinstance(value, bool):
            continue
        if isinstance(value, (int, float)):
            if isinstance(value, float) and np.isnan(value):
                continue
            continue
        try:
            float(value)
        except (TypeError, ValueError):
            return False
    return seen
