"""p95 latency for one request of 100 candidates, single-threaded.

Features are already materialised on ``frame``. The benchmark does not start
threads. The train job pins the CPU node type (``m5d.xlarge``) so runs are
comparable; this function times the process it is called in.
"""

from __future__ import annotations

import time

import numpy as np

from harness.interface import RankingFrame

LATENCY_CANDIDATES = 100
LATENCY_NODE_TYPE = "m5d.xlarge"


def latency_p95_ms(
    ranker,
    frame: RankingFrame,
    *,
    n_candidates: int = LATENCY_CANDIDATES,
    n_requests: int = 50,
    seed: int = 0,
) -> float:
    request = one_request(frame, n_candidates=n_candidates, seed=seed).without_labels()
    for _ in range(3):
        ranker.score(request)
    samples = np.empty(n_requests, dtype=float)
    for index in range(n_requests):
        start = time.perf_counter()
        ranker.score(request)
        samples[index] = (time.perf_counter() - start) * 1000.0
    return float(np.quantile(samples, 0.95))


def one_request(frame: RankingFrame, *, n_candidates: int, seed: int) -> RankingFrame:
    if len(frame) == 0:
        raise ValueError("latency probe needs at least one row")
    del seed
    chosen = [index % len(frame) for index in range(n_candidates)]
    rows = []
    for slot, index in enumerate(chosen):
        row = dict(frame.rows[index])
        row["query_id"] = "latency-probe"
        row["item_id"] = f"c{slot:03d}"
        rows.append(row)
    return RankingFrame(rows)
