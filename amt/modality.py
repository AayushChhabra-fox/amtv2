"""Column modalities and the paradigms a brief is allowed to train.

The dataset manifest records counts and column names. Prep intersects that
profile with the brief's latency, GPU, weight, and license constraints. The
ready gate locks the ladder families that survive both checks.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from amt.schemas import Brief

_FEAT = "feat__"
_EMB = "__emb"
_PROD = "feat__prod_signals__"
_TEXT_TOKENS = ("__text", "__title", "__body", "__headline", "__caption")
_NUMERIC_TYPES = {
    "int",
    "integer",
    "bigint",
    "smallint",
    "tinyint",
    "float",
    "double",
    "long",
    "short",
    "decimal",
    "number",
    "numeric",
}


class ModalityProfile(BaseModel):
    """Column summary stored on ``manifest.json`` under ``modality``."""

    model_config = ConfigDict(extra="forbid")

    numeric_count: int = Field(default=0, ge=0)
    categorical_columns: list[str] = Field(default_factory=list)
    embedding_groups: list[str] = Field(default_factory=list)
    text_columns: list[str] = Field(default_factory=list)
    prod_signal_columns: list[str] = Field(default_factory=list)
    row_count: int = Field(default=0, ge=0)
    query_count: int = Field(default=0, ge=0)


class Paradigm(BaseModel):
    """One model family after the dataset and the brief constraints are applied."""

    model_config = ConfigDict(extra="forbid")

    name: str
    ladder_family: str | None = None
    viable: bool
    reason: str


def summarize_rows(rows_by_split: dict[str, list[dict]]) -> dict[str, Any]:
    """Modality block for a local build that already holds the split rows."""

    seen: dict[str, Any] = {}
    query_ids: set[Any] = set()
    row_count = 0
    for rows in rows_by_split.values():
        row_count += len(rows)
        for row in rows:
            query_id = row.get("query_id")
            if query_id is not None:
                query_ids.add(query_id)
            for key, value in row.items():
                if key not in seen or (seen[key] is None and value is not None):
                    seen[key] = value
    fields = [(name, _python_type(value)) for name, value in seen.items()]
    return summarize_fields(fields, row_count=row_count, query_count=len(query_ids))


def summarize_fields(
    fields: list[tuple[str, str]],
    *,
    row_count: int,
    query_count: int,
) -> dict[str, Any]:
    """Modality block from ``(column, type)`` pairs. Embedding dims are not scalars."""

    numeric = 0
    categorical: list[str] = []
    groups: set[str] = set()
    text: list[str] = []
    prod: list[str] = []
    for name, type_name in fields:
        if not str(name).startswith(_FEAT):
            continue
        column = str(name)
        if _is_prod(column):
            prod.append(column)
        group = _embedding_group(column)
        if group is not None:
            groups.add(group)
            continue
        if _is_text_name(column):
            text.append(column)
            continue
        if _is_numeric_type(type_name):
            numeric += 1
        else:
            categorical.append(column)
    profile = ModalityProfile(
        numeric_count=numeric,
        categorical_columns=sorted(categorical),
        embedding_groups=sorted(groups),
        text_columns=sorted(text),
        prod_signal_columns=sorted(prod),
        row_count=row_count,
        query_count=query_count,
    )
    return profile.model_dump(mode="json")


def profile_from_manifest(manifest: dict[str, Any]) -> ModalityProfile | None:
    """``None`` when the manifest has no modality block."""

    raw = manifest.get("modality")
    if not isinstance(raw, dict):
        return None
    return ModalityProfile.model_validate(raw)


def paradigm_decisions(profile: ModalityProfile, brief: Brief) -> list[Paradigm]:
    """Ladder families plus text and cross-encoder, each with a reason."""

    latency = _latency_ms(brief)
    decisions: list[Paradigm] = []
    for rule in _RULES:
        reason = _blocked(rule, profile, brief, latency)
        decisions.append(
            Paradigm(
                name=rule.name,
                ladder_family=rule.ladder_family,
                viable=reason is None,
                reason=reason or _fit_reason(rule),
            )
        )
    return decisions


def ladder_blockers(profile: ModalityProfile, brief: Brief) -> dict[str, str]:
    """Ladder family ids the dataset or the brief constraints cannot run."""

    blocked: dict[str, str] = {}
    for decision in paradigm_decisions(profile, brief):
        if decision.ladder_family and not decision.viable:
            blocked[decision.ladder_family] = decision.reason
    return blocked


class _Rule:
    def __init__(
        self,
        name: str,
        *,
        ladder_family: str | None,
        needs: str,
        gpu: bool = False,
        external_weights: bool = False,
        min_latency_ms: int = 0,
    ) -> None:
        self.name = name
        self.ladder_family = ladder_family
        self.needs = needs
        self.gpu = gpu
        self.external_weights = external_weights
        self.min_latency_ms = min_latency_ms


_RULES: tuple[_Rule, ...] = (
    _Rule("lgbm", ladder_family="lgbm", needs="numeric"),
    _Rule("neural", ladder_family="neural", needs="numeric", min_latency_ms=5),
    _Rule("embedding", ladder_family="embedding", needs="embedding_pair"),
    _Rule("linear_reweight", ladder_family="linear_reweight", needs="prod_signals"),
    _Rule(
        "text",
        ladder_family=None,
        needs="text",
        external_weights=True,
        min_latency_ms=20,
    ),
    _Rule(
        "cross_encoder",
        ladder_family=None,
        needs="text",
        gpu=True,
        external_weights=True,
        min_latency_ms=50,
    ),
)


def _blocked(
    rule: _Rule,
    profile: ModalityProfile,
    brief: Brief,
    latency: int | None,
) -> str | None:
    missing = _missing_inputs(rule, profile)
    if missing is not None:
        return missing
    if rule.gpu and not brief.constraints.allow_gpu_serving:
        return "GPU serving is not allowed"
    if rule.external_weights and not brief.constraints.allow_external_weights:
        return "external weights are not allowed"
    if rule.external_weights and brief.constraints.license_policy == "commercial_only":
        return "license_policy is commercial_only"
    if latency is not None and rule.min_latency_ms and latency < rule.min_latency_ms:
        return (
            f"serving latency budget {latency}ms is below {rule.min_latency_ms}ms "
            f"for {rule.name}"
        )
    return None


def _missing_inputs(rule: _Rule, profile: ModalityProfile) -> str | None:
    if rule.needs == "numeric" and profile.numeric_count <= 0:
        return "needs numeric feat__ columns"
    if rule.needs == "prod_signals" and not profile.prod_signal_columns:
        return "linear reweighting needs feat__prod_signals__* columns"
    if rule.needs == "embedding_pair":
        groups = set(profile.embedding_groups)
        if "profile" not in groups or "item" not in groups:
            return "embedding ladder needs feat__profile__emb* and feat__item__emb*"
    if rule.needs == "text" and not profile.text_columns:
        return "needs text feat__ columns"
    return None


def _fit_reason(rule: _Rule) -> str:
    return f"{rule.name} fits the dataset and the brief constraints"


def _latency_ms(brief: Brief) -> int | None:
    serving = brief.efficiency.serving
    if serving is None:
        return None
    return serving.latency_p95_ms_per_100_candidates


def _embedding_group(name: str) -> str | None:
    if not name.startswith(_FEAT) or _EMB not in name:
        return None
    body = name[len(_FEAT) :]
    group, _rest = body.split("__", 1)
    return group or None


def _is_prod(name: str) -> bool:
    return name.startswith(_PROD)


def _is_text_name(name: str) -> bool:
    lowered = name.lower()
    return any(token in lowered for token in _TEXT_TOKENS)


def _is_numeric_type(type_name: str) -> bool:
    base = type_name.lower().split("(", 1)[0].strip()
    if base.startswith("decimal"):
        return True
    return base in _NUMERIC_TYPES


def _python_type(value: Any) -> str:
    if isinstance(value, bool):
        return "string"
    if isinstance(value, (int, float)):
        return "double"
    return "string"
