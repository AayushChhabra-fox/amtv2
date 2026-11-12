"""Contracts for briefs, agent outputs, the ledger, and run state.

JSON Schemas exported to ``amt/json_schemas/`` are the copies agent prompts
should follow. ``parse_agent_output`` turns a malformed planner or critic
payload into one revision the orchestrator hands back.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

ApprovalMode = Literal["none", "every_stage", "round_shortlist"]
ScaleMode = Literal["search", "calibration", "small", "full"]
Operator = Literal["draft", "improve", "debug", "ablate", "combine"]
Kind = Literal["model", "baseline"]
Rung = Literal["R0", "R1", "R2", "R3"]
LedgerStatus = Literal["kept", "killed", "failed", "promoted"]
Stage = Literal[
    "PLAN",
    "CRITIQUE",
    "SPEC",
    "IMPLEMENT",
    "PREFLIGHT",
    "SMOKE",
    "DEBUG",
    "SUBMIT",
    "WAIT",
    "PRUNE",
    "CONFIRM",
    "REPORT",
    "ESCALATE",
]

# Set on every MLflow run. parent_id is an empty string when the node has none.
MLFLOW_TAG_KEYS: tuple[str, ...] = (
    "run_id",
    "node_id",
    "parent_id",
    "family",
    "operator",
    "rung",
    "seed",
    "batch",
    "split",
    "git_sha",
)


class MetricSpec(BaseModel):
    """Primary metric the brief asks the loop to move."""

    model_config = ConfigDict(extra="forbid")

    name: str = Field(min_length=1)
    gains: dict[str, int] | None = None
    why_it_tracks_outcome: str | None = None


class Problem(BaseModel):
    """Human-readable problem statement. ``dataset_spec`` may live here."""

    model_config = ConfigDict(extra="forbid")

    surface: str | None = None
    user_outcome: str | None = None
    metric: MetricSpec | None = None
    dataset_spec: str | None = None
    data: dict[str, Any] = Field(default_factory=dict)


class TaskBlock(BaseModel):
    """Task definition the critic checks proposals against."""

    model_config = ConfigDict(extra="forbid")

    query_unit: str = ""
    candidates: str | None = None
    include_query_if: str | None = None
    serve_time_inputs: list[str] = Field(default_factory=list)
    forbidden: list[str] = Field(default_factory=list)


class Reference(BaseModel):
    """Production reference. Extra keys hold measured ranges and notes."""

    model_config = ConfigDict(extra="allow")

    name: str | None = None


class Budget(BaseModel):
    """DBU pool for one brief."""

    model_config = ConfigDict(extra="forbid")

    total_dbu: float = Field(ge=0)
    max_concurrent_jobs: int | None = Field(default=None, ge=1)


class ServingLimit(BaseModel):
    """Serving budget recorded on the brief. The harness enforces it later."""

    model_config = ConfigDict(extra="forbid")

    latency_p95_ms_per_100_candidates: int | None = Field(default=None, ge=0)
    cost: str | None = None


class Efficiency(BaseModel):
    """Placeholders until gate G1 replaces them."""

    model_config = ConfigDict(extra="forbid")

    train_dbu_per_candidate: float | None = Field(default=None, ge=0)
    wall_clock_per_rung: dict[str, str] = Field(default_factory=dict)
    serving: ServingLimit | None = None


class EvalBlock(BaseModel):
    """Where the brief says scores come from. Splits live in the dataset spec."""

    model_config = ConfigDict(extra="forbid")

    split: str | None = None
    final_holdout: str | None = None


class DesignSpace(BaseModel):
    """Families the planner may draft from."""

    model_config = ConfigDict(extra="forbid")

    families: list[str] = Field(default_factory=list)
    out_of_scope_for_mvp: list[str] = Field(default_factory=list)
    data: str | None = None


class BriefCandidate(BaseModel):
    """Single node pinned by a walking-skeleton brief. Search briefs omit this."""

    model_config = ConfigDict(extra="forbid")

    id: str = "c00"
    family: str = "linear"
    rung: Rung = "R0"
    seed: int = Field(default=0, ge=0)
    batch: str = "toy"
    claim: str = "logistic regression"
    est_dbu: float = Field(default=0.1, ge=0)


class Brief(BaseModel):
    """One search brief. ``task`` and ``dataset_spec`` are the contract."""

    model_config = ConfigDict(extra="forbid", validate_assignment=True)

    name: str | None = None
    run_id: str | None = None
    approval_mode: ApprovalMode = "none"
    scale: ScaleMode = Field(
        default="search",
        description=(
            "search is the agent loop. calibration is the one-time G3 run. "
            "small is the K=4 unattended run. full only records Phase 2 entry."
        ),
    )
    dataset_spec: str | None = Field(
        default=None,
        description="Path of the dataset spec folder, for example datasets/nfy_article_rerank.",
    )
    problem: Problem = Field(default_factory=Problem)
    task: TaskBlock = Field(default_factory=TaskBlock)
    reference: Reference | None = None
    min_effect: float = 0.01
    efficiency: Efficiency = Field(default_factory=Efficiency)
    eval: EvalBlock | None = None
    budget: Budget
    k: int = Field(default=4, ge=1, description="Nodes kept per round.")
    mode: str | None = None
    design_space: DesignSpace | None = None
    off_limits: list[str] = Field(default_factory=list)
    candidate: BriefCandidate | None = None

    @model_validator(mode="after")
    def _dataset_specs_agree(self) -> Brief:
        top = self.dataset_spec
        nested = self.problem.dataset_spec
        if top and nested and top != nested:
            raise ValueError("dataset_spec and problem.dataset_spec disagree")
        return self

    def resolved_dataset_spec(self) -> str | None:
        if self.dataset_spec:
            return self.dataset_spec
        return self.problem.dataset_spec


class Candidate(BaseModel):
    """Planner proposal for one node."""

    model_config = ConfigDict(extra="forbid")

    id: str = Field(min_length=1, max_length=64, pattern=r"^[A-Za-z][A-Za-z0-9_-]*$")
    parent_id: str | None = Field(
        default=None,
        description="Set for improve, debug, ablate, and combine. Omit on a draft.",
    )
    operator: Operator
    kind: Kind = "model"
    family: str = Field(min_length=1)
    claim: str = Field(min_length=1)
    fit_argument: str = Field(min_length=1)
    change: str | None = Field(
        default=None, description="The one atomic change. Required for improve."
    )
    cheap_test: str = Field(min_length=1)
    kill_if: str = Field(min_length=1)
    est_cost: float = Field(ge=0, description="Estimated DBU for the next rung.")

    @model_validator(mode="after")
    def _operator_links(self) -> Candidate:
        needs_parent = {"improve", "debug", "ablate", "combine"}
        if self.operator in needs_parent and not self.parent_id:
            raise ValueError(f"{self.operator} requires parent_id")
        if self.operator == "improve" and not (self.change and self.change.strip()):
            raise ValueError("improve requires change")
        if self.operator == "draft" and self.parent_id:
            raise ValueError("draft must not set parent_id")
        return self


class Verdict(BaseModel):
    """Critic decision for one candidate."""

    model_config = ConfigDict(extra="forbid")

    candidate_id: str = Field(min_length=1)
    decision: Literal["PASS", "REVISE", "DROP"]
    reason: str = Field(min_length=1)


class ProposalBatch(BaseModel):
    """Planner output: one or more candidates."""

    model_config = ConfigDict(extra="forbid")

    candidates: list[Candidate] = Field(min_length=1)

    @model_validator(mode="before")
    @classmethod
    def _accept_proposals_key(cls, data: Any) -> Any:
        if isinstance(data, dict) and "candidates" not in data and "proposals" in data:
            renamed = dict(data)
            renamed["candidates"] = renamed.pop("proposals")
            return renamed
        return data


class VerdictBatch(BaseModel):
    """Critic output: one verdict per candidate."""

    model_config = ConfigDict(extra="forbid")

    verdicts: list[Verdict] = Field(min_length=1)

    @model_validator(mode="before")
    @classmethod
    def _accept_single_verdict(cls, data: Any) -> Any:
        if isinstance(data, dict) and "verdicts" not in data and "decision" in data:
            return {"verdicts": [data]}
        return data


class DeltaVsBaseline(BaseModel):
    """Paired difference against the same-rung baseline."""

    model_config = ConfigDict(extra="forbid")

    point: float
    ci_lo: float
    ci_hi: float


class LedgerRow(BaseModel):
    """One append-only result. Measurements stay null until a rung finishes."""

    model_config = ConfigDict(extra="forbid")

    run_id: str = Field(min_length=1)
    node_id: str = Field(min_length=1)
    parent_id: str | None = None
    round: int = Field(default=0, ge=0)
    operator: Operator
    kind: Kind = "model"
    family: str = Field(min_length=1)
    claim: str = Field(min_length=1)
    rung: Rung
    seed: int = Field(ge=0)
    metric: float | None = None
    delta_vs_baseline: DeltaVsBaseline | None = None
    dbu: float | None = Field(default=None, ge=0)
    wall_min: float | None = Field(default=None, ge=0)
    latency_p95_ms: float | None = Field(default=None, ge=0)
    status: LedgerStatus
    kill_reason: str | None = None
    git_sha: str = Field(min_length=1)
    mlflow_run_id: str | None = None
    job_run_id: str | None = None

    @model_validator(mode="after")
    def _kill_reason_matches_status(self) -> LedgerRow:
        if self.status in {"killed", "failed"} and not self.kill_reason:
            raise ValueError(f"{self.status} requires kill_reason")
        if self.status in {"kept", "promoted"} and self.kill_reason:
            raise ValueError(f"{self.status} must not set kill_reason")
        return self


class NodeWork(BaseModel):
    """One node inside a search round. The walking skeleton leaves this empty."""

    model_config = ConfigDict(extra="forbid")

    node_id: str
    family: str
    operator: Operator = "draft"
    parent_id: str | None = None
    parents: list[str] = Field(default_factory=list)
    claim: str = ""
    change: str | None = None
    fit_argument: str = ""
    cheap_test: str = ""
    kill_if: str = ""
    est_cost: float = Field(default=0.1, ge=0)
    rung: Rung = "R1"
    seed: int = Field(default=0, ge=0)
    search_round: int = Field(default=1, ge=0)
    status: Literal[
        "pending",
        "implemented",
        "debugging",
        "ready",
        "submitted",
        "kept",
        "killed",
        "failed",
    ] = "pending"
    debug_attempts: int = Field(default=0, ge=0)
    preflight: str | None = None
    review: str | None = None
    delta_point: float | None = None
    delta_lo: float | None = None
    delta_hi: float | None = None


class RunState(BaseModel):
    """Resumable loop state. Settings are K, budget, and approval_mode."""

    model_config = ConfigDict(extra="forbid", validate_assignment=True)

    run_id: str
    brief_path: str
    stage: Stage = "PLAN"
    round: int = Field(default=0, ge=0)
    live_nodes: list[str] = Field(default_factory=list)
    submitted_jobs: dict[str, dict[str, Any]] = Field(default_factory=dict)
    spent_dbu: float = Field(default=0.0, ge=0)
    k: int = Field(default=4, ge=1)
    budget_dbu: float = Field(default=0.0, ge=0)
    approval_mode: ApprovalMode = "none"
    scale: ScaleMode = "search"
    revisions: int = Field(default=0, ge=0, description="Agent revisions already spent.")
    candidate_id: str
    family: str
    rung: Rung
    seed: int = Field(ge=0)
    batch: str
    git_sha: str = "unknown"
    stop_requested: bool = False
    nodes: list[NodeWork] = Field(default_factory=list)
    awaiting_approval: bool = False
    approval_gate: str | None = None
    round_operator: Operator = "draft"
    critic_revisions: int = Field(default=0, ge=0)
    outer_retries: int = Field(default=0, ge=0)
    promoted: list[str] = Field(default_factory=list)
    rung_index: int = Field(default=0, ge=0)
    improvement_gains: list[float] = Field(default_factory=list)


class AgentRevision(BaseModel):
    """One validation failure returned to the agent. It spends a single revision."""

    model_config = ConfigDict(extra="forbid")

    role: str
    schema_name: str
    revision_count: Literal[1] = 1
    errors: list[str]
    message: str


@dataclass(frozen=True)
class AgentParse:
    value: ProposalBatch | VerdictBatch | None
    revision: AgentRevision | None


ROLE_OUTPUTS: dict[str, type[BaseModel]] = {
    "planner": ProposalBatch,
    "critic": VerdictBatch,
}

SCHEMA_MODELS: dict[str, type[BaseModel]] = {
    "brief": Brief,
    "candidate": Candidate,
    "verdict": Verdict,
    "ledger_row": LedgerRow,
    "run_state": RunState,
    "proposal_batch": ProposalBatch,
    "verdict_batch": VerdictBatch,
}


def json_schema_dir() -> Path:
    return Path(__file__).resolve().parent / "json_schemas"


def export_json_schemas(directory: Path | None = None) -> Path:
    """Write one JSON Schema per contract model. Re-run when models change."""
    target = directory or json_schema_dir()
    target.mkdir(parents=True, exist_ok=True)
    for name, model in SCHEMA_MODELS.items():
        payload = model.model_json_schema()
        text = json.dumps(payload, indent=2) + "\n"
        (target / f"{name}.schema.json").write_text(text)
    return target


def load_brief(path: Path) -> Brief:
    payload = yaml.safe_load(path.read_text())
    if not isinstance(payload, dict):
        raise ValueError(f"{path} did not contain a brief mapping")
    return Brief.model_validate(payload)


def mlflow_tags(
    *,
    run_id: str,
    node_id: str,
    parent_id: str | None,
    family: str,
    operator: str,
    rung: str,
    seed: int | str,
    batch: str,
    split: str,
    git_sha: str,
) -> dict[str, str]:
    """Tags set on every training run. Values are strings, as MLflow stores them."""
    tags = {
        "run_id": str(run_id),
        "node_id": str(node_id),
        "parent_id": "" if parent_id is None else str(parent_id),
        "family": str(family),
        "operator": str(operator),
        "rung": str(rung),
        "seed": str(seed),
        "batch": str(batch),
        "split": str(split),
        "git_sha": str(git_sha),
    }
    missing = [key for key in MLFLOW_TAG_KEYS if key not in tags]
    if missing:
        raise ValueError(f"mlflow tags missing {missing}")
    blank = [key for key in MLFLOW_TAG_KEYS if key != "parent_id" and tags[key] == ""]
    if blank:
        raise ValueError(f"mlflow tags empty: {blank}")
    return tags


def parse_agent_output(role: str, payload: Any) -> AgentParse:
    """Validate planner or critic output.

    A failure returns ``revision`` with ``revision_count`` 1 and every field
    error in ``message``. The caller gives that message back to the agent.
    """
    model = ROLE_OUTPUTS.get(role)
    if model is None:
        raise ValueError(f"no output schema for role {role!r}")
    schema_name = _schema_name(model)
    try:
        data = _normalize_role_payload(role, _load_payload(payload))
        value = model.model_validate(data)
    except ValidationError as exc:
        revision = _revision(role, schema_name, _format_errors(exc))
        return AgentParse(value=None, revision=revision)
    except ValueError as exc:
        revision = _revision(role, schema_name, [f"(root): {exc}"])
        return AgentParse(value=None, revision=revision)
    return AgentParse(value=value, revision=None)


def _schema_name(model: type[BaseModel]) -> str:
    chars: list[str] = []
    for index, char in enumerate(model.__name__):
        if char.isupper() and index:
            chars.append("_")
        chars.append(char.lower())
    return "".join(chars)


def _load_payload(payload: Any) -> Any:
    if not isinstance(payload, str):
        return payload
    text = payload.strip()
    if not text:
        raise ValueError("agent output was empty")
    try:
        return json.loads(text)
    except json.JSONDecodeError as json_exc:
        try:
            loaded = yaml.safe_load(text)
        except yaml.YAMLError as yaml_exc:
            raise ValueError(f"agent output was not JSON or YAML ({json_exc.msg})") from yaml_exc
        if loaded is None:
            raise ValueError("agent output was empty")
        return loaded


def _normalize_role_payload(role: str, data: Any) -> Any:
    if role == "planner" and isinstance(data, list):
        return {"candidates": data}
    if role == "critic" and isinstance(data, list):
        return {"verdicts": data}
    return data


def _format_errors(exc: ValidationError) -> list[str]:
    lines: list[str] = []
    for error in exc.errors():
        location = ".".join(str(part) for part in error["loc"]) or "(root)"
        lines.append(f"{location}: {error['msg']}")
    return lines or ["(root): validation failed"]


def _revision(role: str, schema_name: str, errors: list[str]) -> AgentRevision:
    body = "\n".join(f"- {line}" for line in errors)
    message = (
        f"Validation failed for {role} output against {schema_name}. "
        "This counts as 1 revision.\n"
        f"{body}\n"
        "Return one corrected object that matches the schema.\n"
    )
    return AgentRevision(role=role, schema_name=schema_name, errors=errors, message=message)
