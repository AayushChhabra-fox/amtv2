"""Pickle a baseline next to its scores."""

from __future__ import annotations

import pickle
from pathlib import Path


def save_ranker(ranker: object, path: str) -> None:
    destination = Path(path)
    destination.mkdir(parents=True, exist_ok=True)
    with (destination / "model.pkl").open("wb") as handle:
        pickle.dump(ranker, handle)


def load_ranker(path: str, kind: type):
    with (Path(path) / "model.pkl").open("rb") as handle:
        ranker = pickle.load(handle)
    if not isinstance(ranker, kind):
        raise TypeError(f"expected {kind.__name__} in {path}")
    return ranker
