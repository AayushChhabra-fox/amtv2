"""LightGBM lambdarank over numeric ``feat.*`` columns. Prediction is single-threaded."""

from __future__ import annotations

import numpy as np

from harness.baselines.persist import load_ranker, save_ranker
from harness.interface import FitContext, RankingFrame


class LGBMReferenceRanker:
    def __init__(
        self,
        n_estimators: int = 50,
        num_leaves: int = 15,
        learning_rate: float = 0.1,
        min_child_samples: int = 20,
    ) -> None:
        self.n_estimators = n_estimators
        self.num_leaves = num_leaves
        self.learning_rate = learning_rate
        self.min_child_samples = min_child_samples
        self.columns: list[str] = []
        self.model = None

    def fit(self, train: RankingFrame, valid: RankingFrame, ctx: FitContext) -> None:
        import lightgbm as lgb

        matrix, names = train.numeric_matrix()
        if matrix.shape[1] == 0:
            raise ValueError("lgbm_reference needs at least one numeric feat.* column")
        self.columns = names
        self.model = lgb.LGBMRanker(
            objective="lambdarank",
            n_estimators=self.n_estimators,
            num_leaves=self.num_leaves,
            learning_rate=self.learning_rate,
            min_child_samples=self.min_child_samples,
            random_state=int(ctx.seed),
            n_jobs=1,
            verbosity=-1,
            deterministic=True,
            force_col_wise=True,
        )
        labels = train.labels().astype(int)
        groups = train.group_sizes()
        kwargs = {}
        if len(valid.groups()) >= 1 and valid.numeric_matrix()[1]:
            kwargs["eval_X"] = _align(valid, self.columns)
            kwargs["eval_y"] = valid.labels().astype(int)
            kwargs["eval_group"] = [valid.group_sizes()]
        try:
            self.model.fit(matrix, labels, group=groups, **kwargs)
        except Exception:
            if not kwargs:
                raise
            self.model.fit(matrix, labels, group=groups)

    def score(self, frame: RankingFrame) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("lgbm_reference has not been fit")
        return np.asarray(self.model.predict(_align(frame, self.columns)), dtype=float)

    def save(self, path: str) -> None:
        save_ranker(self, path)

    @classmethod
    def load(cls, path: str) -> LGBMReferenceRanker:
        return load_ranker(path, cls)


def _align(frame: RankingFrame, columns: list[str]) -> np.ndarray:
    matrix = np.zeros((len(frame), len(columns)), dtype=float)
    for column_index, name in enumerate(columns):
        for row_index, row in enumerate(frame.rows):
            raw = row.get(name)
            if raw is None:
                continue
            matrix[row_index, column_index] = float(raw)
    return matrix
