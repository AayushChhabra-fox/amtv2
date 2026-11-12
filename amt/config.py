"""Process settings. A real environment variable wins, then `.env`, then defaults.

`.env` is loaded by ``amt.envutil`` before this is read. A limit here can only
tighten a brief: the effective concurrency is the smaller of the two caps.
Budgets stay on the brief.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from amt.schemas import Brief


def _text(name: str, default: str = "") -> str:
    return os.environ.get(name, default).strip()


def _int(name: str, default: int) -> int:
    raw = _text(name)
    if not raw:
        return default
    return int(raw)


def _float(name: str, default: float = 0.0) -> float:
    raw = _text(name)
    if not raw:
        return default
    return float(raw)


def _bool(name: str, default: bool) -> bool:
    raw = _text(name).lower()
    if not raw:
        return default
    return raw in {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class Settings:
    """One snapshot of the environment for a command."""

    host: str
    catalog: str
    schema_name: str
    artifact_volume: str
    experiment_root: str
    bundle_target: str
    cluster_id: str
    cluster_autostart: bool
    gpu_cluster_id: str
    gpu_node_type: str
    node_type: str
    spark_version: str
    max_concurrent_jobs: int
    dbu_per_hour: float
    gpu_dbu_per_hour: float
    job_wait_timeout_s: int
    jobs_backend: str

    def experiment_for(self, brief_name: str) -> str:
        root = self.experiment_root.rstrip("/")
        return f"{root}/{brief_name}"

    @property
    def gpu_available(self) -> bool:
        return bool(self.gpu_cluster_id or self.gpu_node_type)

    @property
    def fixed_cluster(self) -> bool:
        return bool(self.cluster_id)

    def volume(self, *parts: str) -> str:
        base = self.artifact_volume.rstrip("/")
        extra = "/".join(part.strip("/") for part in parts if part)
        return f"{base}/{extra}" if extra else base

    def dbu_rate(self, *, gpu: bool = False) -> float:
        if gpu and self.gpu_dbu_per_hour > 0:
            return self.gpu_dbu_per_hour
        return self.dbu_per_hour


def load_settings(root: Path | None = None) -> Settings:
    """Read settings. ``root`` is accepted so callers share one signature."""

    del root
    catalog = _text("AMT_CATALOG", "product_api_dev")
    schema = _text("AMT_SCHEMA", "amt")
    volume = _text("AMT_ARTIFACT_VOLUME", f"/Volumes/{catalog}/{schema}/amt_artifacts")
    return Settings(
        host=_text("DATABRICKS_HOST", "https://fox-product-eng-dev-nation.cloud.databricks.com"),
        catalog=catalog,
        schema_name=schema,
        artifact_volume=volume,
        experiment_root=_text("AMT_MLFLOW_EXPERIMENT_ROOT", "/Shared/amt"),
        bundle_target=_text("AMT_BUNDLE_TARGET", "dev"),
        cluster_id=_text("AMT_CLUSTER_ID"),
        cluster_autostart=_bool("AMT_CLUSTER_AUTOSTART", True),
        gpu_cluster_id=_text("AMT_GPU_CLUSTER_ID"),
        gpu_node_type=_text("AMT_GPU_NODE_TYPE"),
        node_type=_text("AMT_NODE_TYPE", "m5d.xlarge"),
        spark_version=_text("AMT_SPARK_VERSION", "15.4.x-scala2.12"),
        max_concurrent_jobs=_int("AMT_MAX_CONCURRENT_JOBS", 2),
        dbu_per_hour=_float("AMT_DBU_PER_HOUR"),
        gpu_dbu_per_hour=_float("AMT_GPU_DBU_PER_HOUR"),
        job_wait_timeout_s=_int("AMT_JOB_WAIT_TIMEOUT_S", 16200),
        jobs_backend=_text("AMT_JOBS_BACKEND", "databricks").lower(),
    )


def effective_concurrency(brief: Brief, settings: Settings | None = None) -> int:
    """The smaller of the brief cap and the environment cap."""

    settings = settings or load_settings()
    env_cap = max(1, settings.max_concurrent_jobs)
    brief_cap = brief.budget.max_concurrent_jobs
    if brief_cap is None:
        return env_cap
    return max(1, min(int(brief_cap), env_cap))


# Set by the process or by tests. Not part of .env.example.
INTERNAL_ENV = frozenset(
    {
        "AMT_ROOT",
        "AMT_GIT_SHA",
        "AMT_STOP_AFTER",
        "AMT_ENV_ROOT",
        "AMT_ROLE",
        "AMT_JOBS_BACKEND",
        "AMT_HELLO_TRAIN_JOB_ID",
        "AMT_TRAIN_JOB_ID",
        "AMT_CONFIRM_JOB_ID",
        "AMT_FINAL_HOLDOUT_JOB_ID",
        "AMT_BUILD_DATASET_JOB_ID",
        "AMT_REQUIRE_BASELINES",
        "AMT_LIVE_DATABRICKS",
    }
)
