"""Score with one ``ref.*`` column the dataset spec declares."""

from __future__ import annotations

import numpy as np

from harness.baselines.persist import load_ranker, save_ranker
from harness.data.contract import ref_column
from harness.interface import FitContext, RankingFrame


class ReferenceScoreRanker:
    def __init__(self, score: str) -> None:
        self.score_name = score
        self.column = ref_column(score)

    def fit(self, train: RankingFrame, valid: RankingFrame, ctx: FitContext) -> None:
        del train, valid, ctx

    def score(self, frame: RankingFrame) -> np.ndarray:
        if self.column not in frame.column_names():
            raise KeyError(self.column)
        values = np.zeros(len(frame), dtype=float)
        for index, row in enumerate(frame.rows):
            raw = row.get(self.column)
            values[index] = 0.0 if raw is None else float(raw)
        return values

    def save(self, path: str) -> None:
        save_ranker(self, path)

    @classmethod
    def load(cls, path: str) -> ReferenceScoreRanker:
        return load_ranker(path, cls)
