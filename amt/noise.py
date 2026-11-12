"""Round-0 noise floor for gate G1.

Every reference score named by the dataset spec, plus popularity, random, and
the LightGBM reference, is scored at R0–R3 with seeds 0, 1 and 2 on
``search_val``. Each non-production baseline carries a paired delta against
the same-rung production scores. The binding half-width at a rung is the
larger of the day-blocked bootstrap half-width and the three-seed t interval.
``min_effect`` must be at least twice the R3 half-width. The proposed starting
value is 0.010. Whether the LightGBM reference beats production is recorded
either way.
"""

from __future__ import annotations

import json
import math
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from amt.schemas import Brief, Rung

PROPOSED_MIN_EFFECT = 0.010
MIN_EFFECT_MULTIPLE = 2.0
ROUND0_RUNGS: tuple[Rung, ...] = ("R0", "R1", "R2", "R3")
ROUND0_SEEDS: tuple[int, ...] = (0, 1, 2)
GENERIC_BASELINES: tuple[str, ...] = ("popularity", "random", "lgbm_reference")
PRODUCTION_BASELINE = "production"
SEARCH_SPLIT = "search_val"
# Student-t 0.975 quantile at df=2. Round 0 always uses three seeds.
_T_975_DF2 = 4.302652729911275
_HOLDOUTS = frozenset({"select_holdout", "final_holdout"})


class GateDataError(ValueError):
    """Round-0 observations cannot be turned into a noise report."""


class TrainJob(BaseModel):
    """One ``train`` job submission. The split is always ``search_val``."""

    model_config = ConfigDict(extra="forbid")

    candidate_id: str
    baseline: str
    reference_score: str = PRODUCTION_BASELINE
    family: str
    rung: Rung
    seed: int = Field(ge=0)
    split: str = SEARCH_SPLIT

    def params(self) -> dict[str, str]:
        if self.split != SEARCH_SPLIT:
            raise GateDataError(f"{self.candidate_id} must score {SEARCH_SPLIT}")
        return {
            "candidate_id": self.candidate_id,
            "baseline": self.baseline,
            "reference_score": self.reference_score,
            "family": self.family,
            "rung": self.rung,
            "seed": str(self.seed),
            "split": self.split,
            "model": "",
        }


class Round0Cell(BaseModel):
    """One baseline seed. Non-production cells include the paired delta."""

    model_config = ConfigDict(extra="forbid")

    baseline_id: str = Field(min_length=1)
    rung: Rung
    seed: int = Field(ge=0, le=2)
    split: str = SEARCH_SPLIT
    metric: float
    delta_point: float | None = None
    delta_ci_lo: float | None = None
    delta_ci_hi: float | None = None


class BaselineNoise(BaseModel):
    """Paired-delta summary for one baseline at one rung."""

    model_config = ConfigDict(extra="forbid")

    baseline_id: str
    rung: Rung
    metric: float
    paired_delta: float
    bootstrap_half_width: float
    seed_half_width: float
    half_width: float


class LGBMOutcome(BaseModel):
    """Recorded at R3. Either result is acceptable."""

    model_config = ConfigDict(extra="forbid")

    rung: Rung = "R3"
    lgbm_metric: float
    production_metric: float
    paired_delta_point: float
    beats_production: bool


class NoiseReport(BaseModel):
    """The document a human signs at gate G1."""

    model_config = ConfigDict(extra="forbid")

    brief_name: str
    dataset: str
    primary_metric: str
    proposed_min_effect: float = PROPOSED_MIN_EFFECT
    min_effect_multiple: float = MIN_EFFECT_MULTIPLE
    brief_min_effect: float
    budget_total_dbu: float
    train_dbu_per_candidate: float | None = None
    r3_half_width: float | None = None
    required_min_effect: float | None = None
    status: Literal["pass", "revise", "incomplete"]
    remediation: list[str] = Field(default_factory=list)
    half_widths: dict[str, float] = Field(default_factory=dict)
    baselines: list[BaselineNoise] = Field(default_factory=list)
    lgbm_vs_production: LGBMOutcome | None = None
    expected_cells: int
    observed_cells: int
    missing: list[str] = Field(default_factory=list)
    evaluation: str = ""


def baseline_ids(reference_scores: list[str] | dict[str, Any]) -> list[str]:
    """Production, the other reference scores, then the generic baselines."""

    names = list(reference_scores)
    if PRODUCTION_BASELINE not in names:
        raise GateDataError(
            "round 0 pairs every baseline against the production reference score"
        )
    ordered = [PRODUCTION_BASELINE]
    ordered.extend(name for name in names if name != PRODUCTION_BASELINE)
    ordered.extend(name for name in GENERIC_BASELINES if name not in ordered)
    return ordered


def round0_jobs(reference_scores: list[str] | dict[str, Any]) -> list[TrainJob]:
    """The R0–R3 matrix. Every job scores ``search_val``."""

    jobs: list[TrainJob] = []
    for name in baseline_ids(reference_scores):
        if name in GENERIC_BASELINES:
            baseline = name
            family = name
            reference_score = PRODUCTION_BASELINE
        else:
            baseline = "reference_score"
            family = f"ref.{name}"
            reference_score = name
        for rung in ROUND0_RUNGS:
            for seed in ROUND0_SEEDS:
                jobs.append(
                    TrainJob(
                        candidate_id=name,
                        baseline=baseline,
                        reference_score=reference_score,
                        family=family,
                        rung=rung,
                        seed=seed,
                        split=SEARCH_SPLIT,
                    )
                )
    return jobs


def expected_cells(reference_scores: list[str] | dict[str, Any]) -> list[tuple[str, str, int]]:
    return [
        (job.candidate_id, job.rung, job.seed) for job in round0_jobs(reference_scores)
    ]


def load_cells(payload: Any) -> list[Round0Cell]:
    """Parse a JSON list of round-0 observations."""

    rows = payload.get("observations") if isinstance(payload, dict) else payload
    if not isinstance(rows, list):
        raise GateDataError("observations must be a list")
    cells: list[Round0Cell] = []
    seen: set[tuple[str, str, int]] = set()
    for index, raw in enumerate(rows):
        if not isinstance(raw, dict):
            raise GateDataError(f"observation {index} is not an object")
        try:
            cell = Round0Cell.model_validate(_normalize_cell(raw))
        except ValidationError as exc:
            raise GateDataError(f"observation {index} is invalid: {exc}") from exc
        _check_cell(cell)
        key = (cell.baseline_id, cell.rung, cell.seed)
        if key in seen:
            raise GateDataError(f"duplicate observation {cell.baseline_id}/{cell.rung}/{cell.seed}")
        seen.add(key)
        cells.append(cell)
    return cells


def build_noise_report(
    cells: list[Round0Cell],
    brief: Brief,
    *,
    dataset: str,
    reference_scores: list[str] | dict[str, Any],
) -> NoiseReport:
    """Aggregate paired-delta half-widths and compare them with ``min_effect``."""

    names = baseline_ids(reference_scores)
    wanted = expected_cells(reference_scores)
    by_key = {(cell.baseline_id, cell.rung, cell.seed): cell for cell in cells}
    extra = sorted(set(by_key) - set(wanted))
    if extra:
        shown = ", ".join(f"{name}/{rung}/{seed}" for name, rung, seed in extra[:8])
        raise GateDataError(f"unexpected round-0 observation: {shown}")
    missing = [
        f"{name}/{rung}/{seed}" for name, rung, seed in wanted if (name, rung, seed) not in by_key
    ]
    grouped: dict[tuple[str, str], list[Round0Cell]] = {}
    for cell in cells:
        grouped.setdefault((cell.baseline_id, cell.rung), []).append(cell)

    baselines: list[BaselineNoise] = []
    half_widths: dict[str, float] = {}
    if not missing:
        for rung in ROUND0_RUNGS:
            rung_width = 0.0
            for name in names:
                if name == PRODUCTION_BASELINE:
                    continue
                summary = _summarize(grouped[(name, rung)])
                baselines.append(summary)
                rung_width = max(rung_width, summary.half_width)
            half_widths[rung] = _round(rung_width)

    outcome = None
    if not missing:
        outcome = _lgbm_outcome(grouped)

    r3_width = half_widths.get("R3")
    required = None if r3_width is None else _round(MIN_EFFECT_MULTIPLE * r3_width)
    remediation: list[str] = []
    if missing:
        status: Literal["pass", "revise", "incomplete"] = "incomplete"
        remediation.append(
            f"{len(missing)} round-0 cells are missing. Score every baseline at R0–R3 "
            "with seeds 0, 1 and 2 on search_val."
        )
    elif required is not None and brief.min_effect + 1e-12 < required:
        status = "revise"
        remediation.append(
            f"min_effect {brief.min_effect:.6f} is below {MIN_EFFECT_MULTIPLE:.0f}× the R3 "
            f"paired-delta half-width {r3_width:.6f} (required {required:.6f})."
        )
        remediation.append(
            "Raise min_effect to at least the required value, or lengthen search_val "
            "or add seeds and re-run round 0."
        )
        if required > PROPOSED_MIN_EFFECT:
            remediation.append(
                "The proposed starting value "
                f"{PROPOSED_MIN_EFFECT:.3f} is below the measured floor."
            )
    else:
        status = "pass"

    metric = brief.problem.metric
    evaluation = ""
    if metric is not None and metric.name:
        evaluation = metric.name
        if metric.gains:
            rendered = ", ".join(f"{key}={value}" for key, value in metric.gains.items())
            evaluation = f"{evaluation} ({rendered})"

    return NoiseReport(
        brief_name=brief.name or dataset,
        dataset=dataset,
        primary_metric=metric.name if metric is not None and metric.name else "",
        brief_min_effect=brief.min_effect,
        budget_total_dbu=brief.budget.total_dbu,
        train_dbu_per_candidate=brief.efficiency.train_dbu_per_candidate,
        r3_half_width=r3_width,
        required_min_effect=required,
        status=status,
        remediation=remediation,
        half_widths=half_widths,
        baselines=baselines,
        lgbm_vs_production=outcome,
        expected_cells=len(wanted),
        observed_cells=len(wanted) - len(missing),
        missing=missing,
        evaluation=evaluation,
    )


def render_noise_markdown(report: NoiseReport) -> str:
    """Human-readable copy of the noise report."""

    lines = [
        f"# Noise report: {report.brief_name}",
        "",
        f"Status: **{report.status}**",
        "",
        f"Dataset `{report.dataset}`. "
        f"Primary metric `{report.evaluation or report.primary_metric}`.",
        f"Budget {report.budget_total_dbu:g} DBU"
        + (
            f", {report.train_dbu_per_candidate:g} DBU per candidate."
            if report.train_dbu_per_candidate is not None
            else "."
        ),
        f"Brief min_effect {report.brief_min_effect:.6f}. "
        f"Proposed starting value {report.proposed_min_effect:.3f}.",
        "",
        "Half-widths are the binding paired-delta half-width at each rung: the larger of "
        "the day-blocked bootstrap half-width and the three-seed t interval. "
        f"min_effect must be at least {report.min_effect_multiple:.0f}× the R3 value.",
        "",
    ]
    if report.r3_half_width is not None and report.required_min_effect is not None:
        lines.append(
            f"R3 half-width {report.r3_half_width:.6f}. "
            f"Required min_effect {report.required_min_effect:.6f}."
        )
        lines.append("")
    if report.half_widths:
        lines.append("| Rung | Half-width |")
        lines.append("| --- | --- |")
        for rung in ROUND0_RUNGS:
            if rung in report.half_widths:
                lines.append(f"| {rung} | {report.half_widths[rung]:.6f} |")
        lines.append("")
    if report.lgbm_vs_production is not None:
        outcome = report.lgbm_vs_production
        relation = "beats" if outcome.beats_production else "does not beat"
        lines.append(
            f"LightGBM reference {relation} production at R3 "
            f"(paired delta {outcome.paired_delta_point:.6f}, "
            f"LightGBM {outcome.lgbm_metric:.6f}, production {outcome.production_metric:.6f})."
        )
        lines.append("")
    if report.remediation:
        lines.append("## Remediation")
        lines.append("")
        for item in report.remediation:
            lines.append(f"- {item}")
        lines.append("")
    if report.missing:
        lines.append(f"Missing cells: {len(report.missing)}.")
        lines.append("")
    lines.append(
        "Gate G1 signs this report together with the brief, the dataset spec, "
        "the manifest, the evaluation definition, min_effect, and the budget."
    )
    lines.append("")
    return "\n".join(lines)


def _summarize(seeds: list[Round0Cell]) -> BaselineNoise:
    seeds = sorted(seeds, key=lambda cell: cell.seed)
    points = [float(cell.delta_point) for cell in seeds]
    widths = [
        (float(cell.delta_ci_hi) - float(cell.delta_ci_lo)) / 2.0  # type: ignore[arg-type]
        for cell in seeds
    ]
    point = sum(points) / len(points)
    bootstrap = max(widths)
    seed_width = _seed_half_width(points)
    cell = seeds[0]
    return BaselineNoise(
        baseline_id=cell.baseline_id,
        rung=cell.rung,
        metric=_round(sum(cell.metric for cell in seeds) / len(seeds)),
        paired_delta=_round(point),
        bootstrap_half_width=_round(bootstrap),
        seed_half_width=_round(seed_width),
        half_width=_round(max(bootstrap, seed_width)),
    )


def _seed_half_width(points: list[float]) -> float:
    count = len(points)
    if count < 2:
        raise GateDataError("the seed interval needs at least two seeds")
    mean = sum(points) / count
    variance = sum((point - mean) ** 2 for point in points) / (count - 1)
    std = math.sqrt(variance)
    return _T_975_DF2 * std / math.sqrt(count)


def _lgbm_outcome(grouped: dict[tuple[str, str], list[Round0Cell]]) -> LGBMOutcome:
    lgbm = grouped[("lgbm_reference", "R3")]
    production = grouped[(PRODUCTION_BASELINE, "R3")]
    delta = sum(float(cell.delta_point) for cell in lgbm) / len(lgbm)
    lgbm_metric = sum(cell.metric for cell in lgbm) / len(lgbm)
    production_metric = sum(cell.metric for cell in production) / len(production)
    return LGBMOutcome(
        lgbm_metric=_round(lgbm_metric),
        production_metric=_round(production_metric),
        paired_delta_point=_round(delta),
        beats_production=delta > 0,
    )


def _normalize_cell(raw: dict[str, Any]) -> dict[str, Any]:
    payload = dict(raw)
    delta = payload.pop("delta", None)
    if isinstance(delta, dict):
        payload.setdefault("delta_point", delta.get("point"))
        payload.setdefault("delta_ci_lo", delta.get("ci_lo"))
        payload.setdefault("delta_ci_hi", delta.get("ci_hi"))
    payload.pop("observations", None)
    return payload


def _check_cell(cell: Round0Cell) -> None:
    if cell.split in _HOLDOUTS:
        raise GateDataError(
            f"{cell.baseline_id} is scored on {cell.split}; round 0 reads search_val only"
        )
    if cell.split != SEARCH_SPLIT:
        raise GateDataError(f"{cell.baseline_id} split must be {SEARCH_SPLIT}")
    if not math.isfinite(cell.metric):
        raise GateDataError(f"{cell.baseline_id}/{cell.rung}/{cell.seed} metric is not finite")
    needs_delta = cell.baseline_id != PRODUCTION_BASELINE
    present = (cell.delta_point, cell.delta_ci_lo, cell.delta_ci_hi)
    if needs_delta and any(value is None for value in present):
        raise GateDataError(
            f"{cell.baseline_id}/{cell.rung}/{cell.seed} needs a paired delta against production"
        )
    if any(value is not None for value in present):
        point, low, high = present
        if point is None or low is None or high is None or not all(
            math.isfinite(value) for value in (point, low, high)
        ):
            raise GateDataError(
                f"{cell.baseline_id}/{cell.rung}/{cell.seed} has a non-finite paired delta"
            )
        if high < low:
            raise GateDataError(
                f"{cell.baseline_id}/{cell.rung}/{cell.seed} has ci_hi below ci_lo"
            )


def _round(value: float) -> float:
    return round(float(value), 6)


def dumps_report(report: NoiseReport) -> str:
    return json.dumps(report.model_dump(mode="json"), indent=2, sort_keys=True) + "\n"
