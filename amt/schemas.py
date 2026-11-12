"""Pydantic contracts used by the Phase 0 loop.

The full validation round-trip (malformed agent output returned as one
revision) lands in Phase 1 M1. These models are the ones the skeleton persists.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


class Brief(BaseModel):
    model_config = ConfigDict(extra="allow")

    name: str
    run_id: str | None = None
    approval_mode: Literal["none", "every_stage", "round_shortlist"] = "none"
    dataset_spec: str | None = None
    problem: dict[str, Any] = Field(default_factory=dict)
    task: dict[str, Any] = Field(default_factory=dict)
    min_effect: float = 0.01
    budget: dict[str, Any] = Field(default_factory=dict)
    efficiency: dict[str, Any] = Field(default_factory=dict)
    candidate: dict[str, Any] = Field(default_factory=dict)


class Candidate(BaseModel):
    model_config = ConfigDict(extra="allow")

    id: str
    parent_id: str | None = None
    operator: str = "draft"
    kind: str = "model"
    family: str
    claim: str
    fit_argument: str | None = None
    change: str | None = None
    cheap_test: str | None = None
    kill_if: str | None = None
    est_cost: float | None = None


class Verdict(BaseModel):
    candidate_id: str
    decision: Literal["PASS", "REVISE", "DROP"]
    reason: str


class LedgerRow(BaseModel):
    run_id: str
    node_id: str
    parent_id: str | None = None
    round: int = 0
    operator: str
    kind: str = "model"
    family: str
    claim: str
    rung: str
    seed: int
    metric: float | None = None
    delta_vs_baseline: dict[str, float] | None = None
    dbu: float | None = None
    wall_min: float | None = None
    latency_p95_ms: float | None = None
    status: str
    kill_reason: str | None = None
    git_sha: str
    mlflow_run_id: str | None = None
    job_run_id: str | None = None


class RunState(BaseModel):
    run_id: str
    brief_path: str
    stage: str
    approval_mode: Literal["none", "every_stage", "round_shortlist"] = "none"
    candidate_id: str
    family: str
    rung: str
    seed: int
    batch: str
    submitted_jobs: dict[str, dict[str, Any]] = Field(default_factory=dict)
    spent_dbu: float = 0.0
    git_sha: str = "unknown"
    stop_requested: bool = False
