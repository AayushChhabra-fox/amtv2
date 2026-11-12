"""Uniform scores. A fixed seed and row count always repeat."""

from __future__ import annotations

import numpy as np

from harness.baselines.persist import load_ranker, save_ranker
from harness.interface import FitContext, RankingFrame


class RandomRanker:
    def __init__(self) -> None:
        self.seed = 0

    def fit(self, train: RankingFrame, valid: RankingFrame, ctx: FitContext) -> None:
        del train, valid
        self.seed = int(ctx.seed)

    def score(self, frame: RankingFrame) -> np.ndarray:
        return np.random.default_rng(self.seed).random(len(frame))

    def save(self, path: str) -> None:
        save_ranker(self, path)

    @classmethod
    def load(cls, path: str) -> RandomRanker:
        return load_ranker(path, cls)
