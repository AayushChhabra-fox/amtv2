"""Output contract and dataset-spec models.

Every dataset the builder writes has the same columns. Evaluation, baselines,
and the pruner read this contract and never a surface-specific schema.

Physical column names use ``__`` because Delta and sqlite treat dots specially.
The logical names in the plan map as follows:

- ``aux_label.*`` -> ``aux_label__*``
- ``feat.<group>.*`` -> ``feat__<group>__*``
- ``ref.*`` -> ``ref__*``
"""

from __future__ import annotations

import fnmatch
import hashlib
import math
import re
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

CONTRACT_COLUMNS: tuple[str, ...] = (
    "query_id",
    "item_id",
    "query_ts",
    "event_day",
    "label",
)
AUX_PREFIX = "aux_label__"
FEAT_PREFIX = "feat__"
REF_PREFIX = "ref__"
SPLIT_NAMES: tuple[str, ...] = (
    "train",
    "search_val",
    "select_holdout",
    "final_holdout",
)
HOLDOUT_SPLITS: frozenset[str] = frozenset({"select_holdout", "final_holdout"})
SEARCH_SPLITS: frozenset[str] = frozenset({"train", "search_val"})
_IDENT = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
_WINDOW = re.compile(r"^(\d+)([mhd])$")
_INCLUDE = re.compile(r"^max\(label\)\s*>=\s*(-?\d+(?:\.\d+)?)$")
_DATE_SPAN = re.compile(r"^(\d{4}-\d{2}-\d{2})\.\.(\d{4}-\d{2}-\d{2})$")


class DatasetBuildError(ValueError):
    """A spec or an assembled frame violated the contract.

    ``kind`` is the revision the builder returns: ``point_in_time``,
    ``label_window``, ``forbidden_column``, ``contract``, ``split``, or ``check``.
    """

    def __init__(self, kind: str, message: str, examples: list[dict] | None = None) -> None:
        self.kind = kind
        self.examples = examples or []
        text = message
        if self.examples:
            text = f"{message}: {self.examples[:5]}"
        super().__init__(text)


class SourceSpec(BaseModel):
    """One warehouse table. Large tables declare a partition filter."""

    model_config = ConfigDict(extra="forbid")

    table: str = Field(min_length=1)
    partition_filter: str | None = None
    date_col: str | None = None
    unpartitioned: bool = False

    @model_validator(mode="after")
    def _filter_or_explicit(self) -> SourceSpec:
        if not self.partition_filter and not self.unpartitioned:
            raise ValueError(f"{self.table} needs partition_filter or unpartitioned: true")
        if self.partition_filter and "{start}" not in self.partition_filter:
            raise ValueError("partition_filter must include {start}")
        if self.partition_filter and "{end}" not in self.partition_filter:
            raise ValueError("partition_filter must include {end}")
        return self


class FeatureGroupSpec(BaseModel):
    """A SQL fragment joined on ``keys``. ``available_at`` is a column the fragment emits."""

    model_config = ConfigDict(extra="forbid")

    sql: str = Field(min_length=1)
    keys: list[str] = Field(min_length=1)
    available_at: str = Field(min_length=1)
    lookback_days: int = Field(default=0, ge=0)

    @field_validator("keys")
    @classmethod
    def _keys(cls, keys: list[str]) -> list[str]:
        allowed = {"query_id", "item_id"}
        if any(key not in allowed for key in keys):
            raise ValueError("feature keys must be query_id and/or item_id")
        return keys

    @field_validator("available_at")
    @classmethod
    def _available(cls, name: str) -> str:
        if not _IDENT.match(name):
            raise ValueError("available_at must be a plain column name")
        return name


class LabelSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")

    sql: str = Field(min_length=1)
    label_window: str
    aux_labels: list[str] = Field(default_factory=list)

    @field_validator("label_window")
    @classmethod
    def _window(cls, value: str) -> str:
        parse_label_window(value)
        return value

    @field_validator("aux_labels")
    @classmethod
    def _aux(cls, names: list[str]) -> list[str]:
        for name in names:
            if not _IDENT.match(name):
                raise ValueError(f"aux label {name!r} is not an identifier")
        return names


class SplitSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")

    time_col: str = "query_ts"
    gap_days: int = Field(ge=0)
    train: str
    search_val: str
    select_holdout: str
    final_holdout: str

    @field_validator("time_col")
    @classmethod
    def _time_col(cls, value: str) -> str:
        if value != "query_ts":
            raise ValueError("MVP splits on query_ts")
        return value


class SmokeSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")

    train_queries: int = Field(ge=1)
    eval_queries: int = Field(ge=1)


class RungSampleSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")

    train_fraction: float = Field(gt=0, le=1)
    eval_queries: int | Literal["all"]

    @field_validator("eval_queries")
    @classmethod
    def _eval(cls, value: int | str) -> int | str:
        if isinstance(value, int) and value < 1:
            raise ValueError("eval_queries must be positive or 'all'")
        return value


class SamplingSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")

    seed: int = Field(default=0, ge=0)
    smoke: SmokeSpec
    rungs: dict[str, RungSampleSpec]

    @model_validator(mode="after")
    def _rungs(self) -> SamplingSpec:
        missing = {"R1", "R2", "R3"} - set(self.rungs)
        if missing:
            raise ValueError(f"sampling.rungs missing {sorted(missing)}")
        return self


class FilterSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")

    include_query_if: str | None = None

    @field_validator("include_query_if")
    @classmethod
    def _include(cls, value: str | None) -> str | None:
        if value is not None and not _INCLUDE.match(value.strip()):
            raise ValueError("include_query_if must look like max(label) >= 1")
        return value


class CandidateBounds(BaseModel):
    model_config = ConfigDict(extra="forbid")

    min: int = Field(default=1, ge=1)
    max: int | None = Field(default=None, ge=1)


class SpecCheck(BaseModel):
    """One assertion from the spec's ``checks`` list."""

    model_config = ConfigDict(extra="forbid")

    name: str
    score: str | None = None
    metric: str | None = None
    split: str | None = None
    expected: str | None = None
    tol: float | None = None
    min: float | None = None
    between: list[float] | None = None

    @model_validator(mode="after")
    def _shape(self) -> SpecCheck:
        if self.between is not None and len(self.between) != 2:
            raise ValueError("between must be [lo, hi]")
        if self.name == "prod_ndcg_reproduces":
            if not self.expected or self.tol is None or not self.split or not self.score:
                raise ValueError("prod_ndcg_reproduces needs score, split, expected, and tol")
        return self


class OutputSpec(BaseModel):
    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    catalog: str = "product_api_dev"
    schema_name: str = Field(default="amt", alias="schema")


class DatasetSpec(BaseModel):
    """Executable dataset definition. Adding a surface is a new folder of these."""

    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    name: str = Field(min_length=1)
    task_type: Literal["ranking"] = "ranking"
    entity_type: str | None = None
    sources: dict[str, SourceSpec]
    queries: str
    candidates: str
    labels: LabelSpec
    feature_groups: dict[str, FeatureGroupSpec]
    reference_scores: dict[str, str] = Field(default_factory=dict)
    filters: FilterSpec = Field(default_factory=FilterSpec)
    forbidden_columns: list[str] = Field(default_factory=list)
    split: SplitSpec
    sampling: SamplingSpec
    checks: list[SpecCheck] = Field(default_factory=list)
    candidate_bounds: CandidateBounds = Field(default_factory=CandidateBounds)
    output: OutputSpec = Field(default_factory=OutputSpec)
    primary_metric: str | None = None
    root: Path | None = Field(default=None, exclude=True)

    @field_validator("name", "entity_type")
    @classmethod
    def _names(cls, value: str | None) -> str | None:
        if value is not None and not _IDENT.match(value):
            raise ValueError("name and entity_type must be identifiers")
        return value

    @field_validator("sources", "feature_groups")
    @classmethod
    def _idents(cls, value: dict) -> dict:
        for key in value:
            if not _IDENT.match(key):
                raise ValueError(f"{key!r} is not an identifier")
        return value

    @field_validator("reference_scores")
    @classmethod
    def _refs(cls, value: dict[str, str]) -> dict[str, str]:
        for name, expr in value.items():
            if not _IDENT.match(name):
                raise ValueError(f"reference score {name!r} is not an identifier")
            column = expr.strip()[1:].strip() if expr.strip().startswith("-") else expr.strip()
            if not _IDENT.match(column):
                raise ValueError(f"reference {name} must be a column or -<column>, got {expr!r}")
        return value

    @model_validator(mode="after")
    def _windows(self) -> DatasetSpec:
        validate_windows(self.split)
        return self

    def split_ranges(self) -> list[tuple[str, date, date]]:
        ranges = []
        for name in SPLIT_NAMES:
            start, end = parse_date_span(getattr(self.split, name))
            ranges.append((name, start, end))
        return ranges

    def include_min_label(self) -> float | None:
        if not self.filters.include_query_if:
            return None
        return float(_INCLUDE.match(self.filters.include_query_if.strip()).group(1))

    def label_window_delta(self) -> timedelta:
        return parse_label_window(self.labels.label_window)

    def sql_text(self, relative: str) -> str:
        if self.root is None:
            raise DatasetBuildError("contract", "spec root is not set")
        path = (self.root / relative).resolve()
        if not path.is_file():
            raise DatasetBuildError("contract", f"missing SQL fragment {relative}")
        return path.read_text()

    def fragment_paths(self) -> list[Path]:
        if self.root is None:
            return []
        paths = [self.root / "dataset.yaml", self.root / self.queries, self.root / self.candidates]
        paths.append(self.root / self.labels.sql)
        for group in self.feature_groups.values():
            paths.append(self.root / group.sql)
        for check in self.checks:
            if check.expected:
                paths.append(self.root / check.expected)
        return paths

    def spec_hash(self) -> str:
        digest = hashlib.sha256()
        for path in self.fragment_paths():
            digest.update(str(path.relative_to(self.root)).encode())
            digest.update(b"\0")
            digest.update(path.read_bytes())
            digest.update(b"\0")
        return digest.hexdigest()


def parse_label_window(value: str) -> timedelta:
    match = _WINDOW.match(value.strip())
    if not match:
        raise ValueError(f"label_window must look like 24h, 30m, or 1d, got {value!r}")
    count = int(match.group(1))
    unit = match.group(2)
    if unit == "m":
        return timedelta(minutes=count)
    if unit == "h":
        return timedelta(hours=count)
    return timedelta(days=count)


def parse_date_span(value: str) -> tuple[date, date]:
    match = _DATE_SPAN.match(value.strip())
    if not match:
        raise ValueError(f"date window must look like 2026-09-14..2026-10-18, got {value!r}")
    start = date.fromisoformat(match.group(1))
    end = date.fromisoformat(match.group(2))
    if end < start:
        raise ValueError(f"window ends before it starts: {value}")
    return start, end


def validate_windows(split: SplitSpec) -> None:
    ranges = []
    for name in SPLIT_NAMES:
        start, end = parse_date_span(getattr(split, name))
        ranges.append((name, start, end))
    for (left_name, left_start, left_end), (right_name, right_start, _right_end) in zip(
        ranges, ranges[1:], strict=False
    ):
        if right_start <= left_end:
            raise ValueError(f"{left_name} overlaps {right_name}")
        gap = (right_start - left_end).days - 1
        if gap < split.gap_days:
            raise ValueError(
                f"gap between {left_name} and {right_name} is {gap} days, "
                f"spec requires {split.gap_days}"
            )
        if left_start > left_end:
            raise ValueError(f"{left_name} is empty")


def load_spec(path: str | Path) -> DatasetSpec:
    """Load ``dataset.yaml`` or a spec directory."""

    spec_path = Path(path)
    if spec_path.is_dir():
        spec_path = spec_path / "dataset.yaml"
    root = spec_path.parent.resolve()
    payload = yaml.safe_load(spec_path.read_text())
    if not isinstance(payload, dict):
        raise DatasetBuildError("contract", f"{spec_path} is not a mapping")
    spec = DatasetSpec.model_validate(payload)
    spec.root = root
    for relative in [spec.queries, spec.candidates, spec.labels.sql]:
        if not (root / relative).is_file():
            raise DatasetBuildError("contract", f"missing SQL fragment {relative}")
    for group in spec.feature_groups.values():
        if not (root / group.sql).is_file():
            raise DatasetBuildError("contract", f"missing SQL fragment {group.sql}")
    for check in spec.checks:
        if check.expected and not (root / check.expected).is_file():
            raise DatasetBuildError("contract", f"missing check SQL {check.expected}")
    return spec


def as_datetime(value: Any) -> datetime | None:
    if value is None:
        return None
    if isinstance(value, datetime):
        return value.replace(tzinfo=None)
    if isinstance(value, date) and not isinstance(value, datetime):
        return datetime(value.year, value.month, value.day)
    text = str(value).strip().replace("Z", "")
    if not text or text.lower() == "none":
        return None
    text = text.replace(" ", "T")
    try:
        return datetime.fromisoformat(text).replace(tzinfo=None)
    except ValueError as exc:
        raise DatasetBuildError("contract", f"could not parse timestamp {value!r}") from exc


def as_day(value: Any) -> date:
    parsed = as_datetime(value)
    if parsed is None:
        raise DatasetBuildError("contract", "missing query_ts")
    return parsed.date()


def assign_split(spec: DatasetSpec, query_ts: Any) -> str | None:
    day = as_day(query_ts)
    for name, start, end in spec.split_ranges():
        if start <= day <= end:
            return name
    return None


def ref_column(name: str) -> str:
    if name.startswith(REF_PREFIX):
        return name
    if name.startswith("ref."):
        return REF_PREFIX + name[4:]
    return REF_PREFIX + name


def feat_column(group: str, column: str) -> str:
    return f"{FEAT_PREFIX}{group}__{column}"


def aux_column(name: str) -> str:
    if name.startswith(AUX_PREFIX):
        return name
    return AUX_PREFIX + name


def forbidden_hit(column: str, patterns: list[str]) -> str | None:
    for pattern in patterns:
        if fnmatch.fnmatch(column, pattern):
            return pattern
    return None


def _examples(rows: list[dict], limit: int = 5) -> list[dict]:
    exported = []
    for row in rows[:limit]:
        exported.append({key: _plain(value) for key, value in row.items()})
    return exported


def _plain(value: Any) -> Any:
    if isinstance(value, datetime):
        return value.isoformat(sep=" ")
    if isinstance(value, date):
        return value.isoformat()
    if isinstance(value, float) and math.isnan(value):
        return None
    return value


def _require_columns(rows: list[dict], columns: set[str], what: str) -> None:
    if not rows:
        raise DatasetBuildError("contract", f"{what} produced no rows")
    present: set[str] = set()
    for row in rows:
        present.update(row)
    missing = columns - present
    if missing:
        raise DatasetBuildError("contract", f"{what} is missing columns {sorted(missing)}")


def _scan_forbidden(rows: list[dict], patterns: list[str], what: str) -> None:
    columns: set[str] = set()
    for row in rows:
        columns.update(row)
    hits = []
    for column in sorted(columns):
        pattern = forbidden_hit(column, patterns)
        if pattern is not None:
            hits.append({"column": column, "pattern": pattern, "fragment": what})
    if hits:
        raise DatasetBuildError(
            "forbidden_column",
            f"{what} contains forbidden columns",
            hits,
        )


def _index_unique(rows: list[dict], keys: list[str], what: str) -> dict[tuple, dict]:
    index: dict[tuple, dict] = {}
    duplicates = []
    for row in rows:
        identity = tuple(row.get(key) for key in keys)
        if any(part is None for part in identity):
            raise DatasetBuildError("contract", f"{what} has a null key {keys}")
        if identity in index:
            duplicates.append({key: row.get(key) for key in keys})
            continue
        index[identity] = row
    if duplicates:
        raise DatasetBuildError("contract", f"{what} has duplicate keys {keys}", duplicates)
    return index


def _number(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return float(value)
    if isinstance(value, (int, float)):
        if isinstance(value, float) and math.isnan(value):
            return None
        return float(value)
    text = str(value).strip()
    if not text:
        return None
    try:
        return float(text)
    except ValueError:
        return None


def _feature_values(row: dict, group: FeatureGroupSpec) -> dict[str, Any]:
    skip = set(group.keys) | {group.available_at}
    return {key: value for key, value in row.items() if key not in skip}


def _resolve_ref(expr: str, sources: list[dict]) -> Any:
    text = expr.strip()
    negative = text.startswith("-")
    column = text[1:].strip() if negative else text
    for source in sources:
        if column in source and source[column] is not None:
            value = _number(source[column])
            if value is None:
                raise DatasetBuildError("contract", f"reference column {column} is not numeric")
            return -value if negative else value
    return None


def assemble_rows(
    *,
    queries: list[dict],
    candidates: list[dict],
    labels: list[dict],
    features: dict[str, list[dict]],
    spec: DatasetSpec,
) -> list[dict]:
    """Join fragment results and reject leaks, bad labels, and forbidden columns."""

    _scan_forbidden(queries, spec.forbidden_columns, "queries")
    _scan_forbidden(candidates, spec.forbidden_columns, "candidates")
    _scan_forbidden(labels, spec.forbidden_columns, "labels")
    for name, rows in features.items():
        _scan_forbidden(rows, spec.forbidden_columns, f"features/{name}")

    if not queries or not candidates:
        return []
    _require_columns(queries, {"query_id", "query_ts"}, "queries")
    _require_columns(candidates, {"query_id", "item_id"}, "candidates")
    if not labels:
        raise DatasetBuildError("contract", "labels produced no rows")
    _require_columns(labels, {"query_id", "item_id", "label", "label_event_ts"}, "labels")

    query_index = _index_unique(queries, ["query_id"], "queries")
    label_index = _index_unique(labels, ["query_id", "item_id"], "labels")
    feature_index: dict[str, dict[tuple, dict]] = {}
    for name, group in spec.feature_groups.items():
        rows = features.get(name, [])
        if not rows:
            feature_index[name] = {}
            continue
        _require_columns(rows, set(group.keys) | {group.available_at}, f"features/{name}")
        feature_index[name] = _index_unique(rows, list(group.keys), f"features/{name}")

    window = spec.label_window_delta()
    pair_rows: list[dict] = []
    seen_pairs: set[tuple] = set()
    missing_labels = []
    window_breaks = []
    pit_breaks = []

    for candidate in candidates:
        query_id = candidate.get("query_id")
        item_id = candidate.get("item_id")
        if (query_id,) not in query_index:
            continue
        pair = (query_id, item_id)
        if pair in seen_pairs:
            raise DatasetBuildError(
                "contract",
                "candidates has duplicate (query_id, item_id)",
                [{"query_id": query_id, "item_id": item_id}],
            )
        seen_pairs.add(pair)
        query = query_index[(query_id,)]
        query_ts = as_datetime(query.get("query_ts"))
        if query_ts is None:
            raise DatasetBuildError("contract", f"query {query_id} has no query_ts")
        label_row = label_index.get(pair)
        if label_row is None:
            missing_labels.append({"query_id": query_id, "item_id": item_id})
            continue
        label_value = _number(label_row.get("label"))
        if label_value is None:
            raise DatasetBuildError("contract", f"label is null for {pair}")
        label_ts = as_datetime(label_row.get("label_event_ts"))
        if label_ts is None or label_ts < query_ts or label_ts > query_ts + window:
            window_breaks.append(
                {
                    "query_id": query_id,
                    "item_id": item_id,
                    "query_ts": query_ts,
                    "label_event_ts": label_row.get("label_event_ts"),
                }
            )
            continue

        output: dict[str, Any] = {
            "query_id": query_id,
            "item_id": item_id,
            "query_ts": query_ts,
            "event_day": query_ts.date().isoformat(),
            "label": int(label_value),
        }
        for aux in spec.labels.aux_labels:
            if aux not in label_row:
                raise DatasetBuildError("contract", f"labels fragment has no aux column {aux}")
            output[aux_column(aux)] = _number(label_row.get(aux))
        raw_sources = [candidate]
        for name, group in spec.feature_groups.items():
            identity = tuple(query_id if key == "query_id" else item_id for key in group.keys)
            feature_row = feature_index.get(name, {}).get(identity)
            if feature_row is None:
                continue
            raw_sources.append(feature_row)
            values = _feature_values(feature_row, group)
            available = as_datetime(feature_row.get(group.available_at))
            populated = [key for key, value in values.items() if value is not None]
            if populated and (available is None or available > query_ts):
                pit_breaks.append(
                    {
                        "group": name,
                        "query_id": query_id,
                        "item_id": item_id,
                        "available_at": feature_row.get(group.available_at),
                        "query_ts": query_ts,
                    }
                )
                continue
            for key, value in values.items():
                output[feat_column(name, key)] = value
        for ref_name, expr in spec.reference_scores.items():
            output[ref_column(ref_name)] = _resolve_ref(expr, raw_sources)
        pair_rows.append(output)

    if missing_labels:
        raise DatasetBuildError(
            "contract",
            "every candidate needs exactly one label",
            _examples(missing_labels),
        )
    if window_breaks:
        raise DatasetBuildError(
            "label_window",
            f"label_event_ts falls outside {spec.labels.label_window} after query_ts",
            _examples(window_breaks),
        )
    if pit_breaks:
        raise DatasetBuildError(
            "point_in_time",
            "a feature is populated with available_at after query_ts",
            _examples(pit_breaks),
        )

    minimum = spec.include_min_label()
    if minimum is not None:
        best: dict[Any, float] = {}
        for row in pair_rows:
            best[row["query_id"]] = max(best.get(row["query_id"], minimum - 1), row["label"])
        pair_rows = [row for row in pair_rows if best[row["query_id"]] >= minimum]

    counts: dict[Any, int] = {}
    for row in pair_rows:
        counts[row["query_id"]] = counts.get(row["query_id"], 0) + 1
    bound_breaks = []
    for query_id, count in counts.items():
        if count < spec.candidate_bounds.min:
            bound_breaks.append({"query_id": query_id, "n": count})
        elif spec.candidate_bounds.max is not None and count > spec.candidate_bounds.max:
            bound_breaks.append({"query_id": query_id, "n": count})
    if bound_breaks:
        raise DatasetBuildError(
            "contract",
            "candidates per query are outside candidate_bounds",
            bound_breaks[:5],
        )

    _scan_forbidden(pair_rows, spec.forbidden_columns, "assembled")
    pair_rows.sort(key=lambda row: (str(row["query_id"]), str(row["item_id"])))
    return pair_rows


def content_hash(rows: list[dict]) -> str:
    """Stable hash of a split. Rebuilds of the same rows match."""

    digest = hashlib.sha256()
    ordered = sorted(rows, key=lambda row: (str(row["query_id"]), str(row["item_id"])))
    columns = _output_columns(ordered)
    digest.update("\t".join(columns).encode())
    digest.update(b"\n")
    for row in ordered:
        parts = [_hash_token(row.get(column)) for column in columns]
        digest.update("\t".join(parts).encode())
        digest.update(b"\n")
    return digest.hexdigest()


def _output_columns(rows: list[dict]) -> list[str]:
    if not rows:
        return list(CONTRACT_COLUMNS)
    present = set(rows[0])
    for row in rows:
        present.update(row)
    head = [column for column in CONTRACT_COLUMNS if column in present]
    rest = sorted(present - set(head))
    return head + rest


def _hash_token(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        if math.isnan(value):
            return "nan"
        return f"{value:.6g}"
    if isinstance(value, datetime):
        return value.isoformat(sep=" ")
    if isinstance(value, date):
        return value.isoformat()
    return str(value)


def split_summary(rows: list[dict]) -> dict[str, Any]:
    if not rows:
        return {
            "rows": 0,
            "queries": 0,
            "min_query_ts": None,
            "max_query_ts": None,
            "min_day": None,
            "max_day": None,
            "content_hash": content_hash([]),
            "content_hash_alg": "sha256-canonical-rows",
        }
    stamps = [as_datetime(row["query_ts"]) for row in rows]
    days = sorted({row["event_day"] for row in rows})
    return {
        "rows": len(rows),
        "queries": len({row["query_id"] for row in rows}),
        "min_query_ts": min(stamps).isoformat(sep=" ") if stamps else None,
        "max_query_ts": max(stamps).isoformat(sep=" ") if stamps else None,
        "min_day": days[0] if days else None,
        "max_day": days[-1] if days else None,
        "content_hash": content_hash(rows),
        "content_hash_alg": "sha256-canonical-rows",
    }
