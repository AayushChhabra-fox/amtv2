"""DBU pool and per-candidate caps.

Live spend is wall-clock times the cluster DBU rate. A submission that would
push the pool below zero, or a candidate past its cap, is refused. Reconciliation
against ``system.billing.usage`` stays a later stretch.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from amt.schemas import Brief, LedgerRow

# Jobs Compute DBU per node-hour for the pinned CPU node. Billing reconciliation
# replaces this table; callers can pass an explicit rate until then.
DEFAULT_NODE_TYPE = "m5d.xlarge"
CLUSTER_DBU_PER_HOUR: dict[str, float] = {
    "m5d.xlarge": 0.75,
}


class BudgetExceeded(RuntimeError):
    pass


def pool_dbu(brief: Brief) -> float:
    return float(brief.budget.total_dbu)


def candidate_cap_dbu(brief: Brief) -> float | None:
    cap = brief.efficiency.train_dbu_per_candidate
    if cap is None:
        return None
    return float(cap)


def estimate_dbu(
    wall_min: float,
    *,
    node_type: str = DEFAULT_NODE_TYPE,
    workers: int = 0,
    dbu_per_hour: float | None = None,
) -> float:
    """DBUs for one single-node or small cluster from its wall clock."""

    if wall_min < 0:
        raise ValueError("wall_min must be >= 0")
    if workers < 0:
        raise ValueError("workers must be >= 0")
    rate = CLUSTER_DBU_PER_HOUR[node_type] if dbu_per_hour is None else dbu_per_hour
    return (wall_min / 60.0) * rate * (workers + 1)


def node_committed_dbu(rows: list[LedgerRow], run_id: str, node_id: str) -> float:
    """DBUs already written to the ledger for one node."""

    total = 0.0
    for row in rows:
        if row.run_id == run_id and row.node_id == node_id and row.dbu:
            total += float(row.dbu)
    return total


def inflight_dbu(
    submitted_jobs: Mapping[str, Mapping[str, Any]],
    node_id: str | None = None,
) -> float:
    """Reservations for jobs submitted and not yet pruned into the ledger."""

    total = 0.0
    prefix = None if node_id is None else f"{node_id}/"
    for key, record in submitted_jobs.items():
        if record.get("pruned"):
            continue
        if prefix is not None and not str(key).startswith(prefix):
            continue
        total += float(record.get("dbu_est") or 0)
    return total


def assert_can_spend(
    brief: Brief,
    spent_dbu: float,
    next_dbu: float,
    *,
    candidate_spent_dbu: float = 0.0,
) -> None:
    """Refuse a submission that would drop the pool below zero or pass a candidate cap.

    Spending the pool or the cap down to exactly zero is allowed. ``spent_dbu``
    should already include in-flight reservations.
    """

    if next_dbu < 0:
        raise BudgetExceeded("refusing submit: next spend is negative")
    pool = pool_dbu(brief)
    if spent_dbu + next_dbu > pool:
        raise BudgetExceeded(
            f"refusing submit: spent {spent_dbu} + {next_dbu} exceeds pool {pool}"
        )
    cap = candidate_cap_dbu(brief)
    if cap is not None and candidate_spent_dbu + next_dbu > cap:
        raise BudgetExceeded(
            f"refusing submit: candidate spent {candidate_spent_dbu} + {next_dbu} "
            f"exceeds cap {cap}"
        )
