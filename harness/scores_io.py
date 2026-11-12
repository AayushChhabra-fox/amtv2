"""Keyed score files. Rows are ``query_id``, ``item_id``, ``score``.

Parquet is preferred. JSON is the fallback when pyarrow is not installed, and
readers accept either suffix.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from harness.interface import RankingFrame


def write_scores(path: Path, frame: RankingFrame, scores: np.ndarray) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    values = np.asarray(scores, dtype=float)
    for index, row in enumerate(frame.rows):
        rows.append(
            {
                "query_id": str(row["query_id"]),
                "item_id": str(row["item_id"]),
                "score": float(values[index]),
            }
        )
    if path.suffix == ".parquet":
        try:
            import pyarrow as pa
            import pyarrow.parquet as pq

            pq.write_table(pa.Table.from_pylist(rows), path)
            return path
        except ImportError:
            path = path.with_suffix(".json")
    path.write_text(json.dumps(rows), encoding="utf-8")
    return path


def read_scores(path: Path) -> list[dict]:
    if not path.is_file():
        alternate = path.with_suffix(".json" if path.suffix == ".parquet" else ".parquet")
        if alternate.is_file():
            path = alternate
        else:
            raise FileNotFoundError(path)
    if path.suffix == ".parquet":
        import pyarrow.parquet as pq

        return pq.read_table(path).to_pylist()
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError(f"{path} is not a list of scores")
    return payload


def align_scores(frame: RankingFrame, rows: list[dict]) -> np.ndarray:
    """Join baseline scores onto the frame. Missing keys are a hard failure."""

    index = {
        (str(row["query_id"]), str(row["item_id"])): float(row["score"]) for row in rows
    }
    aligned: list[float] = []
    missing = 0
    for row in frame.rows:
        key = (str(row["query_id"]), str(row["item_id"]))
        if key not in index:
            missing += 1
            aligned.append(float("nan"))
            continue
        aligned.append(index[key])
    if missing:
        raise RuntimeError(
            f"baseline scores cover {len(frame) - missing}/{len(frame)} rows"
        )
    return np.asarray(aligned, dtype=float)


def reference_vector(frame: RankingFrame, name: str) -> np.ndarray | None:
    column = f"ref__{name}"
    if column not in frame.columns:
        return None
    return np.asarray([float(row.get(column) or 0.0) for row in frame.rows], dtype=float)
