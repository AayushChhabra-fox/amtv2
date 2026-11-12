"""Dataset-agnostic baselines.

``reference_score`` is one ranker per ``ref.*`` column. ``popularity``,
``random`` and ``lgbm_reference`` do not read the dataset name.
"""

from __future__ import annotations

from harness.baselines.lgbm_reference import LGBMReferenceRanker
from harness.baselines.popularity import PopularityRanker
from harness.baselines.random import RandomRanker
from harness.baselines.reference_score import ReferenceScoreRanker
from harness.data.contract import DatasetSpec

__all__ = [
    "LGBMReferenceRanker",
    "PopularityRanker",
    "RandomRanker",
    "ReferenceScoreRanker",
    "iter_baselines",
]


def iter_baselines(spec: DatasetSpec, **lgbm_kwargs: object):
    for name in spec.reference_scores:
        yield ReferenceScoreRanker(name)
    yield PopularityRanker()
    yield RandomRanker()
    yield LGBMReferenceRanker(**lgbm_kwargs)  # type: ignore[arg-type]
