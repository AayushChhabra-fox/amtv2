"""Ranker protocol. Dataset-specific code does not belong here."""

from __future__ import annotations

from typing import Protocol

import numpy as np


class RankingFrame(Protocol):
    def __getitem__(self, key: str): ...


class FitContext(Protocol):
    seed: int


class Ranker(Protocol):
    def fit(self, train: RankingFrame, valid: RankingFrame, ctx: FitContext) -> None: ...

    def score(self, frame: RankingFrame) -> np.ndarray: ...

    def save(self, path: str) -> None: ...

    @classmethod
    def load(cls, path: str) -> Ranker: ...
