"""DBU pool. Live reconciliation against system.billing is a Phase 1 stretch."""

from __future__ import annotations

from amt.schemas import Brief


class BudgetExceeded(RuntimeError):
    pass


def pool_dbu(brief: Brief) -> float:
    return float(brief.budget.total_dbu)


def assert_can_spend(brief: Brief, spent_dbu: float, next_dbu: float) -> None:
    pool = pool_dbu(brief)
    if spent_dbu + next_dbu > pool:
        raise BudgetExceeded(f"refusing submit: spent {spent_dbu} + {next_dbu} exceeds pool {pool}")
