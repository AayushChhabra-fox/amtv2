"""Known-good and known-bad rankers for the fidelity ladder.

The four families are the section 5.4 pairs. Each ranker reads only the
contract columns it is given:

- LightGBM on every numeric ``feat.*`` column, or on columns whose names
  contain ``recency`` or ``category``.
- A small MLP, trained on the labels or on a seed-shuffled copy of them.
- Dot-product similarity of ``feat__profile__emb*`` and ``feat__item__emb*``,
  with a fitted diagonal, or after a seeded permutation of the item side.
- A linear reweighting of ``feat__prod_signals__*``, or the ``ref.production``
  column left as it is.

Callers pass ``search_val`` frames. These classes do not open a holdout.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

from harness.baselines.lgbm_reference import LGBMReferenceRanker
from harness.baselines.persist import load_ranker, save_ranker
from harness.data.contract import ref_column
from harness.interface import FitContext, RankingFrame

Role = Literal["good", "bad"]


@dataclass(frozen=True)
class LadderDesign:
    """One side of a known-good / known-bad pair."""

    design_id: str
    family: str
    role: Role
    claim: str


LADDER_DESIGNS: tuple[LadderDesign, ...] = (
    LadderDesign(
        "lgbm_full",
        "lgbm",
        "good",
        "LightGBM on all numeric features",
    ),
    LadderDesign(
        "lgbm_recency_category",
        "lgbm",
        "bad",
        "LightGBM on recency and category only",
    ),
    LadderDesign(
        "neural_normal",
        "neural",
        "good",
        "MLP trained on the real labels",
    ),
    LadderDesign(
        "neural_label_shuffled",
        "neural",
        "bad",
        "MLP trained on shuffled labels",
    ),
    LadderDesign(
        "embedding_trained",
        "embedding",
        "good",
        "Fitted similarity of profile and item embeddings",
    ),
    LadderDesign(
        "embedding_random_projection",
        "embedding",
        "bad",
        "Similarity after a seeded permutation of the item embedding",
    ),
    LadderDesign(
        "linear_fitted",
        "linear_reweight",
        "good",
        "Linear weights fit on the production signals",
    ),
    LadderDesign(
        "linear_production_weights",
        "linear_reweight",
        "bad",
        "The production score, with its weights left unchanged",
    ),
)

LADDER_IDS: frozenset[str] = frozenset(design.design_id for design in LADDER_DESIGNS)


def ladder_families() -> list[str]:
    """Family ids in catalog order, without duplicates."""

    seen: list[str] = []
    for design in LADDER_DESIGNS:
        if design.family not in seen:
            seen.append(design.family)
    return seen


def family_key() -> str:
    """Stable directory key for this family set."""

    return "+".join(sorted(ladder_families()))


def designs_by_family() -> dict[str, tuple[LadderDesign, LadderDesign]]:
    """Map each family to its ``(good, bad)`` pair."""

    grouped: dict[str, dict[str, LadderDesign]] = {}
    for design in LADDER_DESIGNS:
        grouped.setdefault(design.family, {})[design.role] = design
    pairs: dict[str, tuple[LadderDesign, LadderDesign]] = {}
    for family, roles in grouped.items():
        pairs[family] = (roles["good"], roles["bad"])
    return pairs


def make_ladder_ranker(design_id: str):
    """Build the ranker for one ladder design id."""

    kind = _RANKERS.get(design_id)
    if kind is None:
        known = ", ".join(sorted(LADDER_IDS))
        raise KeyError(f"unknown ladder design {design_id}; expected one of {known}")
    return kind()


class LGBMFullRanker:
    """LightGBM lambdarank over every numeric feature column."""

    def __init__(self) -> None:
        self.model = LGBMReferenceRanker(n_estimators=40, num_leaves=7, min_child_samples=1)

    def fit(self, train: RankingFrame, valid: RankingFrame, ctx: FitContext) -> None:
        self.model.fit(train, valid, ctx)

    def score(self, frame: RankingFrame) -> np.ndarray:
        return self.model.score(frame)

    def save(self, path: str) -> None:
        save_ranker(self, path)

    @classmethod
    def load(cls, path: str) -> LGBMFullRanker:
        return load_ranker(path, cls)


class LGBMRecencyCategoryRanker:
    """LightGBM restricted to recency and category columns."""

    def __init__(self) -> None:
        self.model = LGBMReferenceRanker(n_estimators=40, num_leaves=7, min_child_samples=1)
        self.columns: list[str] = []

    def fit(self, train: RankingFrame, valid: RankingFrame, ctx: FitContext) -> None:
        self.columns = _recency_category(train.numeric_feature_names())
        self.model.fit(_select(train, self.columns), _select(valid, self.columns), ctx)

    def score(self, frame: RankingFrame) -> np.ndarray:
        if not self.columns:
            raise RuntimeError("lgbm_recency_category has not been fit")
        return self.model.score(_select(frame, self.columns))

    def save(self, path: str) -> None:
        save_ranker(self, path)

    @classmethod
    def load(cls, path: str) -> LGBMRecencyCategoryRanker:
        return load_ranker(path, cls)


class NeuralNormalRanker:
    """Pointwise MLP on numeric features."""

    def __init__(self) -> None:
        self._net = _NumpyMLP(shuffle_labels=False)

    def fit(self, train: RankingFrame, valid: RankingFrame, ctx: FitContext) -> None:
        del valid
        self._net.fit(train, ctx)

    def score(self, frame: RankingFrame) -> np.ndarray:
        return self._net.score(frame)

    def save(self, path: str) -> None:
        save_ranker(self, path)

    @classmethod
    def load(cls, path: str) -> NeuralNormalRanker:
        return load_ranker(path, cls)


class NeuralShuffledRanker:
    """The same MLP, fit on a permutation of the training labels."""

    def __init__(self) -> None:
        self._net = _NumpyMLP(shuffle_labels=True)

    def fit(self, train: RankingFrame, valid: RankingFrame, ctx: FitContext) -> None:
        del valid
        self._net.fit(train, ctx)

    def score(self, frame: RankingFrame) -> np.ndarray:
        return self._net.score(frame)

    def save(self, path: str) -> None:
        save_ranker(self, path)

    @classmethod
    def load(cls, path: str) -> NeuralShuffledRanker:
        return load_ranker(path, cls)


class EmbeddingTrainedRanker:
    """Diagonal weights on the elementwise product of the two embeddings."""

    def __init__(self) -> None:
        self.pairs: list[tuple[str, str]] = []
        self.weights: np.ndarray | None = None

    def fit(self, train: RankingFrame, valid: RankingFrame, ctx: FitContext) -> None:
        del valid, ctx
        self.pairs = _embedding_pairs(sorted(train.column_names()))
        products = _products(train, self.pairs)
        self.weights = _ridge(products, train.labels())

    def score(self, frame: RankingFrame) -> np.ndarray:
        if self.weights is None:
            raise RuntimeError("embedding_trained has not been fit")
        return _products(frame, self.pairs) @ self.weights

    def save(self, path: str) -> None:
        save_ranker(self, path)

    @classmethod
    def load(cls, path: str) -> EmbeddingTrainedRanker:
        return load_ranker(path, cls)


class EmbeddingRandomProjectionRanker:
    """Dot product after a seeded permutation of the item embedding.

    The permutation moves the first axis when a draw would have left it in
    place, so the bad arm is not a copy of the aligned similarity.
    """

    def __init__(self) -> None:
        self.pairs: list[tuple[str, str]] = []
        self.permutation: np.ndarray | None = None

    def fit(self, train: RankingFrame, valid: RankingFrame, ctx: FitContext) -> None:
        del valid
        self.pairs = _embedding_pairs(sorted(train.column_names()))
        self.permutation = _permutation(len(self.pairs), int(ctx.seed))

    def score(self, frame: RankingFrame) -> np.ndarray:
        if self.permutation is None:
            raise RuntimeError("embedding_random_projection has not been fit")
        user, item = _sides(frame, self.pairs)
        return np.sum(user * item[:, self.permutation], axis=1)

    def save(self, path: str) -> None:
        save_ranker(self, path)

    @classmethod
    def load(cls, path: str) -> EmbeddingRandomProjectionRanker:
        return load_ranker(path, cls)


class LinearFittedRanker:
    """Least-squares weights on the production-signal columns."""

    def __init__(self) -> None:
        self.columns: list[str] = []
        self.weights: np.ndarray | None = None

    def fit(self, train: RankingFrame, valid: RankingFrame, ctx: FitContext) -> None:
        del valid, ctx
        self.columns = _prod_signals(train.numeric_feature_names())
        self.weights = _ridge(_align(train, self.columns), train.labels())

    def score(self, frame: RankingFrame) -> np.ndarray:
        if self.weights is None:
            raise RuntimeError("linear_fitted has not been fit")
        return _align(frame, self.columns) @ self.weights

    def save(self, path: str) -> None:
        save_ranker(self, path)

    @classmethod
    def load(cls, path: str) -> LinearFittedRanker:
        return load_ranker(path, cls)


class LinearProductionRanker:
    """Score with the production reference column. Nothing is fit."""

    def __init__(self) -> None:
        self.column = ref_column("production")

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
    def load(cls, path: str) -> LinearProductionRanker:
        return load_ranker(path, cls)


_RANKERS: dict[str, type] = {
    "lgbm_full": LGBMFullRanker,
    "lgbm_recency_category": LGBMRecencyCategoryRanker,
    "neural_normal": NeuralNormalRanker,
    "neural_label_shuffled": NeuralShuffledRanker,
    "embedding_trained": EmbeddingTrainedRanker,
    "embedding_random_projection": EmbeddingRandomProjectionRanker,
    "linear_fitted": LinearFittedRanker,
    "linear_production_weights": LinearProductionRanker,
}


class _NumpyMLP:
    """One hidden layer, ReLU, mean-squared error. Small on purpose."""

    def __init__(self, shuffle_labels: bool, hidden: int = 8, steps: int = 160) -> None:
        self.shuffle_labels = shuffle_labels
        self.hidden = hidden
        self.steps = steps
        self.columns: list[str] = []
        self.mean: np.ndarray | None = None
        self.std: np.ndarray | None = None
        self.w1: np.ndarray | None = None
        self.b1: np.ndarray | None = None
        self.w2: np.ndarray | None = None
        self.b2: float = 0.0

    def fit(self, train: RankingFrame, ctx: FitContext) -> None:
        matrix, names = train.numeric_matrix()
        if matrix.shape[1] == 0:
            raise ValueError("neural ladder needs at least one numeric feat.* column")
        self.columns = names
        labels = train.labels().astype(float).copy()
        rng = np.random.default_rng(int(ctx.seed) + (17 if self.shuffle_labels else 0))
        if self.shuffle_labels:
            labels = rng.permutation(labels)
        mean = matrix.mean(axis=0)
        std = matrix.std(axis=0)
        std = np.where(std < 1e-8, 1.0, std)
        self.mean = mean
        self.std = std
        scaled = (matrix - mean) / std
        width = scaled.shape[1]
        w1 = rng.normal(0.0, 0.2, size=(width, self.hidden))
        b1 = np.zeros(self.hidden)
        w2 = rng.normal(0.0, 0.2, size=(self.hidden,))
        b2 = 0.0
        count = float(scaled.shape[0])
        first_loss: float | None = None
        last_loss = 0.0
        learning_rate = 0.15
        for step in range(self.steps):
            if ctx.expired():
                break
            pre = scaled @ w1 + b1
            activated = np.maximum(pre, 0.0)
            pred = activated @ w2 + b2
            residual = pred - labels
            last_loss = float(np.mean(residual**2))
            if step == 0:
                first_loss = last_loss
            err = residual / count
            grad_w2 = activated.T @ err
            grad_b2 = float(np.sum(err))
            grad_hidden = np.outer(err, w2) * (pre > 0.0)
            w2 = w2 - learning_rate * grad_w2
            b2 = b2 - learning_rate * grad_b2
            w1 = w1 - learning_rate * (scaled.T @ grad_hidden)
            b1 = b1 - learning_rate * grad_hidden.sum(axis=0)
        self.w1, self.b1, self.w2, self.b2 = w1, b1, w2, b2
        if first_loss is not None:
            ctx.log("train_loss", first_loss)
            ctx.log("train_loss", last_loss)

    def score(self, frame: RankingFrame) -> np.ndarray:
        if self.w1 is None or self.mean is None or self.std is None or self.w2 is None:
            raise RuntimeError("neural ladder has not been fit")
        scaled = (_align(frame, self.columns) - self.mean) / self.std
        activated = np.maximum(scaled @ self.w1 + self.b1, 0.0)
        return activated @ self.w2 + self.b2


def _recency_category(names: list[str]) -> list[str]:
    kept = [name for name in names if "recency" in name or "category" in name]
    if not kept:
        raise ValueError(
            "lgbm_recency_category needs a numeric feature whose name contains "
            "recency or category"
        )
    return kept


def _prod_signals(names: list[str]) -> list[str]:
    kept = [name for name in names if name.startswith("feat__prod_signals__")]
    if not kept:
        raise ValueError("linear reweighting needs feat__prod_signals__* columns")
    return kept


def _embedding_pairs(names: list[str]) -> list[tuple[str, str]]:
    profile: dict[str, str] = {}
    item: dict[str, str] = {}
    for name in names:
        if "__emb" not in name:
            continue
        suffix = name.split("__emb", 1)[1]
        if name.startswith("feat__profile__"):
            profile[suffix] = name
        elif name.startswith("feat__item__"):
            item[suffix] = name
    keys = sorted(set(profile) & set(item))
    if not keys:
        raise ValueError("embedding ladder needs feat__profile__emb* and feat__item__emb*")
    return [(profile[key], item[key]) for key in keys]


def _sides(
    frame: RankingFrame, pairs: list[tuple[str, str]]
) -> tuple[np.ndarray, np.ndarray]:
    user = np.zeros((len(frame), len(pairs)), dtype=float)
    item = np.zeros((len(frame), len(pairs)), dtype=float)
    for column, (left, right) in enumerate(pairs):
        for index, row in enumerate(frame.rows):
            raw_user = row.get(left)
            raw_item = row.get(right)
            if raw_user is not None:
                user[index, column] = float(raw_user)
            if raw_item is not None:
                item[index, column] = float(raw_item)
    return user, item


def _products(frame: RankingFrame, pairs: list[tuple[str, str]]) -> np.ndarray:
    user, item = _sides(frame, pairs)
    return user * item


def _permutation(dim: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed + 91)
    order = rng.permutation(dim)
    if dim > 1 and int(order[0]) == 0:
        order = np.roll(order, 1)
    return order


def _select(frame: RankingFrame, columns: list[str]) -> RankingFrame:
    keep = set(columns)
    keep.update({"query_id", "item_id", "label", "event_day", "query_ts"})
    rows = [{key: value for key, value in row.items() if key in keep} for row in frame.rows]
    return RankingFrame(rows)


def _align(frame: RankingFrame, columns: list[str]) -> np.ndarray:
    matrix = np.zeros((len(frame), len(columns)), dtype=float)
    for column_index, name in enumerate(columns):
        for row_index, row in enumerate(frame.rows):
            raw = row.get(name)
            if raw is None:
                continue
            matrix[row_index, column_index] = float(raw)
    return matrix


def _ridge(features: np.ndarray, labels: np.ndarray, l2: float = 1e-3) -> np.ndarray:
    width = features.shape[1]
    gram = features.T @ features + l2 * np.eye(width)
    return np.linalg.solve(gram, features.T @ labels)
