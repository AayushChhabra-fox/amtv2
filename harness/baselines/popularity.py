"""Item click rate estimated on the train split only."""

from __future__ import annotations

import numpy as np

from harness.baselines.persist import load_ranker, save_ranker
from harness.interface import FitContext, RankingFrame


class PopularityRanker:
    def __init__(self) -> None:
        self.rates: dict = {}
        self.global_rate = 0.0

    def fit(self, train: RankingFrame, valid: RankingFrame, ctx: FitContext) -> None:
        del valid, ctx
        labels = train.labels()
        clicks: dict = {}
        counts: dict = {}
        for item_id, label in zip(train.item_id, labels, strict=True):
            counts[item_id] = counts.get(item_id, 0) + 1
            if float(label) > 0:
                clicks[item_id] = clicks.get(item_id, 0) + 1
        self.rates = {item_id: clicks.get(item_id, 0) / count for item_id, count in counts.items()}
        self.global_rate = float(np.mean(labels > 0)) if len(labels) else 0.0

    def score(self, frame: RankingFrame) -> np.ndarray:
        return np.asarray(
            [self.rates.get(item_id, self.global_rate) for item_id in frame.item_id],
            dtype=float,
        )

    def save(self, path: str) -> None:
        save_ranker(self, path)

    @classmethod
    def load(cls, path: str) -> PopularityRanker:
        return load_ranker(path, cls)
