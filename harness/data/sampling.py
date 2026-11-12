"""Seeded query-level samples for the smoke rung and R1-R3.

Train rungs take the most recent fraction of train queries. Eval caps are a
seeded sample of search_val. Holdout query ids are never sampled here.
"""

from __future__ import annotations

import hashlib
import math
import random
from datetime import datetime
from typing import Any

from harness.data.contract import DatasetBuildError, DatasetSpec, as_datetime


def scaled_count(n_queries: int, fraction: float) -> int:
    if n_queries <= 0:
        return 0
    if fraction >= 1:
        return n_queries
    return max(1, int(math.floor(n_queries * fraction + 0.5)))


def sample_rules(spec: DatasetSpec) -> dict[str, dict[str, dict[str, Any]]]:
    """Rung -> role -> rule. Roles are ``train`` and ``eval``."""

    seed = spec.sampling.seed
    rules: dict[str, dict[str, dict[str, Any]]] = {
        "R0": {
            "train": {
                "split": "train",
                "mode": "seeded_cap",
                "n": spec.sampling.smoke.train_queries,
                "seed": seed,
            },
            "eval": {
                "split": "search_val",
                "mode": "seeded_cap",
                "n": spec.sampling.smoke.eval_queries,
                "seed": seed,
            },
        }
    }
    for rung, rung_spec in spec.sampling.rungs.items():
        if rung_spec.eval_queries == "all":
            eval_rule: dict[str, Any] = {
                "split": "search_val",
                "mode": "all",
                "seed": seed,
            }
        else:
            eval_rule = {
                "split": "search_val",
                "mode": "seeded_cap",
                "n": int(rung_spec.eval_queries),
                "seed": seed,
            }
        rules[rung] = {
            "train": {
                "split": "train",
                "mode": "most_recent_fraction",
                "fraction": rung_spec.train_fraction,
                "seed": seed,
            },
            "eval": eval_rule,
        }
    return rules


def query_times(rows: list[dict]) -> dict[str, datetime]:
    """Latest query_ts per query id. Ids are strings so samples survive JSON."""

    times: dict[str, datetime] = {}
    for row in rows:
        query_id = str(row["query_id"])
        stamp = as_datetime(row.get("query_ts"))
        if stamp is None:
            continue
        current = times.get(query_id)
        if current is None or stamp > current:
            times[query_id] = stamp
    return times


def query_ids_hash(query_ids: list[str]) -> str:
    digest = hashlib.sha256()
    for query_id in sorted(str(query_id) for query_id in query_ids):
        digest.update(query_id.encode())
        digest.update(b"\n")
    return digest.hexdigest()


def choose_query_ids(query_ts: dict[str, datetime], rule: dict[str, Any]) -> list[str]:
    """Return the query ids for one sample rule. The result is deterministic."""

    mode = rule["mode"]
    if mode == "all":
        return _by_recency(query_ts)
    if mode == "most_recent_fraction":
        ordered = _by_recency(query_ts)
        return ordered[: scaled_count(len(ordered), float(rule["fraction"]))]
    if mode == "seeded_cap":
        ordered = sorted(query_ts)
        limit = rule.get("n")
        if limit is None or int(limit) >= len(ordered):
            return ordered
        drawn = random.Random(int(rule["seed"])).sample(ordered, int(limit))
        return sorted(drawn)
    raise DatasetBuildError("contract", f"unknown sample mode {mode}")


def _by_recency(query_ts: dict[str, datetime]) -> list[str]:
    return sorted(query_ts, key=lambda query_id: (-query_ts[query_id].timestamp(), query_id))
