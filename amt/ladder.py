"""Ladder validation for gate G2.

Each known-good / known-bad pair is scored at R1–R3 with two seeds on
``search_val``. Within a family the good design must rank above the bad one
at every rung. Kendall tau-b between the R1 and R3 scores of all eight
designs must be at least 0.6. A failure only at R1 can be re-validated by
skipping R1 for that family. A low tau with the within-family order intact
is re-validated by raising R1's train fraction. The signed result is reused
for later runs on the same dataset and family set.
"""

from __future__ import annotations

import json
import math
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from amt.noise import SEARCH_SPLIT, GateDataError, TrainJob
from harness.ladder import (
    LADDER_DESIGNS,
    LadderDesign,
    designs_by_family,
    family_key,
    ladder_families,
)

KENDALL_MIN = 0.6
LADDER_RUNGS: tuple[str, ...] = ("R1", "R2", "R3")
LADDER_SEEDS: tuple[int, ...] = (0, 1)
_HOLDOUTS = frozenset({"select_holdout", "final_holdout"})


class LadderCell(BaseModel):
    """One design seed on ``search_val``."""

    model_config = ConfigDict(extra="forbid")

    design_id: str = Field(min_length=1)
    rung: Literal["R1", "R2", "R3"]
    seed: int = Field(ge=0, le=1)
    split: str = SEARCH_SPLIT
    metric: float


class DesignScore(BaseModel):
    model_config = ConfigDict(extra="forbid")

    design_id: str
    family: str
    role: str
    rung: str
    metric: float


class FamilyOrder(BaseModel):
    model_config = ConfigDict(extra="forbid")

    family: str
    rung: str
    good_id: str
    bad_id: str
    good_metric: float
    bad_metric: float
    holds: bool
    checked: bool


class LadderReport(BaseModel):
    """The document a human signs at gate G2."""

    model_config = ConfigDict(extra="forbid")

    dataset: str
    family_set: list[str]
    family_key: str
    status: Literal["pass", "revise", "fail", "incomplete"]
    kendall_tau: float | None = None
    kendall_min: float = KENDALL_MIN
    kendall_n: int = 0
    skip_r1: list[str] = Field(default_factory=list)
    r1_train_fraction: float | None = None
    suggested_r1_train_fraction: float | None = None
    remediation: list[str] = Field(default_factory=list)
    orders: list[FamilyOrder] = Field(default_factory=list)
    scores: list[DesignScore] = Field(default_factory=list)
    missing: list[str] = Field(default_factory=list)
    expected_cells: int = 0
    observed_cells: int = 0


def ladder_jobs() -> list[TrainJob]:
    """R1–R3 for every ladder design, two seeds, always on ``search_val``."""

    jobs: list[TrainJob] = []
    for design in LADDER_DESIGNS:
        for rung in LADDER_RUNGS:
            for seed in LADDER_SEEDS:
                jobs.append(
                    TrainJob(
                        candidate_id=design.design_id,
                        baseline=design.design_id,
                        reference_score="production",
                        family=design.family,
                        rung=rung,  # type: ignore[arg-type]
                        seed=seed,
                        split=SEARCH_SPLIT,
                    )
                )
    return jobs


def load_cells(payload: Any) -> list[LadderCell]:
    rows = payload.get("observations") if isinstance(payload, dict) else payload
    if not isinstance(rows, list):
        raise GateDataError("observations must be a list")
    known = {design.design_id for design in LADDER_DESIGNS}
    cells: list[LadderCell] = []
    seen: set[tuple[str, str, int]] = set()
    for index, raw in enumerate(rows):
        if not isinstance(raw, dict):
            raise GateDataError(f"observation {index} is not an object")
        try:
            cell = LadderCell.model_validate(raw)
        except ValidationError as exc:
            raise GateDataError(f"observation {index} is invalid: {exc}") from exc
        if cell.design_id not in known:
            raise GateDataError(f"unknown ladder design {cell.design_id}")
        if cell.split in _HOLDOUTS:
            raise GateDataError(
                f"{cell.design_id} is scored on {cell.split}; the ladder reads search_val only"
            )
        if cell.split != SEARCH_SPLIT:
            raise GateDataError(f"{cell.design_id} split must be {SEARCH_SPLIT}")
        if not math.isfinite(cell.metric):
            raise GateDataError(f"{cell.design_id}/{cell.rung}/{cell.seed} metric is not finite")
        key = (cell.design_id, cell.rung, cell.seed)
        if key in seen:
            raise GateDataError(f"duplicate observation {cell.design_id}/{cell.rung}/{cell.seed}")
        seen.add(key)
        cells.append(cell)
    return cells


def build_ladder_report(
    cells: list[LadderCell],
    *,
    dataset: str,
    skip_r1: list[str] | None = None,
    r1_train_fraction: float | None = None,
) -> LadderReport:
    """Check within-family order and the R1–R3 Kendall tau."""

    skipped = _clean_skip(skip_r1 or [])
    families = ladder_families()
    pairs = designs_by_family()
    wanted = [
        (design.design_id, rung, seed)
        for design in LADDER_DESIGNS
        for rung in LADDER_RUNGS
        for seed in LADDER_SEEDS
    ]
    by_key = {(cell.design_id, cell.rung, cell.seed): cell for cell in cells}
    extra = sorted(set(by_key) - set(wanted))
    if extra:
        shown = ", ".join(f"{name}/{rung}/{seed}" for name, rung, seed in extra[:8])
        raise GateDataError(f"unexpected ladder observation: {shown}")
    missing = [
        f"{name}/{rung}/{seed}"
        for name, rung, seed in wanted
        if (name, rung, seed) not in by_key
    ]

    means = _means(cells)
    scores = [
        DesignScore(
            design_id=design.design_id,
            family=design.family,
            role=design.role,
            rung=rung,
            metric=_round(means[(design.design_id, rung)]),
        )
        for design in LADDER_DESIGNS
        for rung in LADDER_RUNGS
        if (design.design_id, rung) in means
    ]
    orders = _orders(pairs, means, skipped) if not missing else []

    tau: float | None = None
    kendall_n = 0
    if not missing:
        ranked = [design for design in LADDER_DESIGNS if design.family not in skipped]
        kendall_n = len(ranked)
        if kendall_n >= 2:
            tau = _round(
                kendall_tau_b(
                    [means[(design.design_id, "R1")] for design in ranked],
                    [means[(design.design_id, "R3")] for design in ranked],
                )
            )

    remediation, status = _judge(
        missing=missing,
        orders=orders,
        tau=tau,
        kendall_n=kendall_n,
        ranked_ids=[design.design_id for design in LADDER_DESIGNS if design.family not in skipped],
        means=means,
        r1_train_fraction=r1_train_fraction,
    )
    return LadderReport(
        dataset=dataset,
        family_set=families,
        family_key=family_key(),
        status=status,
        kendall_tau=tau,
        kendall_n=kendall_n,
        skip_r1=skipped,
        r1_train_fraction=r1_train_fraction,
        suggested_r1_train_fraction=_suggest_fraction(r1_train_fraction),
        remediation=remediation,
        orders=orders,
        scores=scores,
        missing=missing,
        expected_cells=len(wanted),
        observed_cells=len(wanted) - len(missing),
    )


def kendall_tau_b(xs: list[float], ys: list[float]) -> float:
    """Kendall tau-b. Ties in either ranking are excluded from the numerator."""

    count = len(xs)
    if count != len(ys) or count < 2:
        raise GateDataError("Kendall tau needs at least two designs scored at both rungs")
    pairs = count * (count - 1) / 2
    concordant = 0
    discordant = 0
    for left in range(count):
        for right in range(left + 1, count):
            dx = xs[left] - xs[right]
            dy = ys[left] - ys[right]
            if dx == 0 or dy == 0:
                continue
            if dx * dy > 0:
                concordant += 1
            else:
                discordant += 1
    denominator = math.sqrt((pairs - _tie_pairs(xs)) * (pairs - _tie_pairs(ys)))
    if denominator == 0:
        return 0.0
    return (concordant - discordant) / denominator


def render_ladder_markdown(report: LadderReport) -> str:
    lines = [
        f"# Ladder report: {report.dataset}",
        "",
        f"Status: **{report.status}**",
        "",
        f"Family set `{', '.join(report.family_set)}`.",
        f"Kendall tau between R1 and R3: "
        f"{'n/a' if report.kendall_tau is None else f'{report.kendall_tau:.4f}'}"
        f" (minimum {report.kendall_min:.1f}, n={report.kendall_n}).",
        "",
    ]
    if report.skip_r1:
        lines.append(f"R1 skipped for: {', '.join(report.skip_r1)}.")
        lines.append("")
    if report.r1_train_fraction is not None:
        suggestion = (
            f", suggested next fraction {report.suggested_r1_train_fraction:.2f}"
            if report.suggested_r1_train_fraction is not None
            else ""
        )
        lines.append(f"R1 train fraction {report.r1_train_fraction:.2f}{suggestion}.")
        lines.append("")
    if report.orders:
        lines.append("| Family | Rung | Good | Bad | Holds |")
        lines.append("| --- | --- | --- | --- | --- |")
        for order in report.orders:
            if not order.checked:
                holds = "skipped"
            else:
                holds = "yes" if order.holds else "no"
            lines.append(
                f"| {order.family} | {order.rung} | {order.good_metric:.4f} | "
                f"{order.bad_metric:.4f} | {holds} |"
            )
        lines.append("")
    if report.remediation:
        lines.append("## Remediation")
        lines.append("")
        for item in report.remediation:
            lines.append(f"- {item}")
        lines.append("")
    lines.append(
        "Gate G2 signs this report once per dataset and family set. "
        "Later runs on the same pair reuse the signature."
    )
    lines.append("")
    return "\n".join(lines)


def _judge(
    *,
    missing: list[str],
    orders: list[FamilyOrder],
    tau: float | None,
    kendall_n: int,
    ranked_ids: list[str],
    means: dict[tuple[str, str], float],
    r1_train_fraction: float | None,
) -> tuple[list[str], Literal["pass", "revise", "fail", "incomplete"]]:
    if missing:
        return (
            [
                f"{len(missing)} ladder cells are missing. "
                "Score every design at R1–R3 with seeds 0 and 1 on search_val."
            ],
            "incomplete",
        )
    high_rung = [
        order
        for order in orders
        if order.checked and not order.holds and order.rung in {"R2", "R3"}
    ]
    r1_failures = [
        order for order in orders if order.checked and not order.holds and order.rung == "R1"
    ]
    notes: list[str] = []
    if high_rung:
        for order in high_rung:
            notes.append(
                f"{order.family} ordering failed at {order.rung}: "
                f"{order.good_id} mean {order.good_metric:.6f} is not above "
                f"{order.bad_id} mean {order.bad_metric:.6f}."
            )
        notes.append(
            "Skipping R1 does not repair a pair that fails at R2 or R3. "
            "The known-good and known-bad designs do not separate."
        )
        return notes, "fail"
    if kendall_n < 2 or tau is None:
        notes.append(
            "Skipping R1 for every family leaves no R1-versus-R3 ranking to validate."
        )
        return notes, "fail"
    if r1_failures:
        families = sorted({order.family for order in r1_failures})
        for order in r1_failures:
            notes.append(
                f"{order.family} ordering failed at R1: "
                f"{order.good_id} mean {order.good_metric:.6f} is not above "
                f"{order.bad_id} mean {order.bad_metric:.6f}."
            )
        notes.append(
            "Skip R1 for "
            + ", ".join(families)
            + " and re-validate. Ordering held at R2 and R3."
        )
        return notes, "revise"
    if tau + 1e-12 < KENDALL_MIN:
        notes.append(f"Kendall tau between R1 and R3 is {tau:.4f}, below {KENDALL_MIN:.1f}.")
        suggestion = _suggest_fraction(r1_train_fraction)
        if suggestion is not None and r1_train_fraction is not None:
            notes.append(
                f"Raise R1's train fraction from {r1_train_fraction:.2f} to {suggestion:.2f} "
                "and re-validate."
            )
        elif r1_train_fraction is not None:
            notes.append(
                "R1 already uses the full train split. Add seeds, or skip R1 for the "
                "families whose rank moves the most, and re-validate."
            )
        else:
            notes.append("Raise R1's train fraction, or skip R1 for a family, and re-validate.")
        movers = _rank_movers(ranked_ids, means)
        if movers:
            notes.append("Largest R1-to-R3 rank moves: " + ", ".join(movers) + ".")
        return notes, "revise"
    return notes, "pass"


def _orders(
    pairs: dict[str, tuple[LadderDesign, LadderDesign]],
    means: dict[tuple[str, str], float],
    skipped: list[str],
) -> list[FamilyOrder]:
    orders: list[FamilyOrder] = []
    for family in ladder_families():
        good, bad = pairs[family]
        for rung in LADDER_RUNGS:
            checked = not (rung == "R1" and family in skipped)
            good_metric = means[(good.design_id, rung)]
            bad_metric = means[(bad.design_id, rung)]
            orders.append(
                FamilyOrder(
                    family=family,
                    rung=rung,
                    good_id=good.design_id,
                    bad_id=bad.design_id,
                    good_metric=_round(good_metric),
                    bad_metric=_round(bad_metric),
                    holds=good_metric > bad_metric,
                    checked=checked,
                )
            )
    return orders


def _means(cells: list[LadderCell]) -> dict[tuple[str, str], float]:
    grouped: dict[tuple[str, str], list[float]] = {}
    for cell in cells:
        grouped.setdefault((cell.design_id, cell.rung), []).append(cell.metric)
    return {key: sum(values) / len(values) for key, values in grouped.items()}


def _clean_skip(skip_r1: list[str]) -> list[str]:
    known = set(ladder_families())
    cleaned: list[str] = []
    for family in skip_r1:
        if family not in known:
            raise GateDataError(f"cannot skip R1 for unknown family {family}")
        if family not in cleaned:
            cleaned.append(family)
    return cleaned


def _suggest_fraction(current: float | None) -> float | None:
    if current is None:
        return None
    if current < 0.4 - 1e-12:
        return 0.4
    if current < 1.0 - 1e-12:
        return 1.0
    return None


def _rank_movers(
    ranked_ids: list[str],
    means: dict[tuple[str, str], float],
) -> list[str]:
    r1 = _average_ranks([means[(design_id, "R1")] for design_id in ranked_ids])
    r3 = _average_ranks([means[(design_id, "R3")] for design_id in ranked_ids])
    by_family: dict[str, float] = {}
    catalog = {design.design_id: design.family for design in LADDER_DESIGNS}
    for index, design_id in enumerate(ranked_ids):
        family = catalog[design_id]
        by_family[family] = max(by_family.get(family, 0.0), abs(r1[index] - r3[index]))
    ordered = sorted(by_family, key=lambda family: (-by_family[family], family))
    return [f"{family} ({by_family[family]:.1f})" for family in ordered if by_family[family] >= 2]


def _average_ranks(values: list[float]) -> list[float]:
    order = sorted(range(len(values)), key=lambda index: (-values[index], index))
    ranks = [0.0] * len(values)
    start = 0
    while start < len(order):
        end = start + 1
        while end < len(order) and values[order[end]] == values[order[start]]:
            end += 1
        average = (start + 1 + end) / 2
        for cursor in range(start, end):
            ranks[order[cursor]] = average
        start = end
    return ranks


def _tie_pairs(values: list[float]) -> float:
    counts: dict[float, int] = {}
    for value in values:
        counts[value] = counts.get(value, 0) + 1
    return sum(count * (count - 1) / 2 for count in counts.values() if count > 1)


def _round(value: float) -> float:
    return round(float(value), 6)


def dumps_report(report: LadderReport) -> str:
    return json.dumps(report.model_dump(mode="json"), indent=2, sort_keys=True) + "\n"
