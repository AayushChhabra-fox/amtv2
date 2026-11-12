"""SQL rendering shared by the local builder and the Spark job.

Placeholders are ``{name}`` identifiers from the spec. Fragment text is
trusted repository code; names are restricted to identifiers before they are
interpolated.
"""

from __future__ import annotations

import re
from datetime import date, timedelta

from harness.data.contract import (
    _IDENT,
    AUX_PREFIX,
    DatasetBuildError,
    DatasetSpec,
    FeatureGroupSpec,
    feat_column,
    ref_column,
)

_PLACEHOLDER = re.compile(r"\{([A-Za-z_][A-Za-z0-9_]*)\}")


def render(template: str, params: dict[str, str]) -> str:
    def replace(match: re.Match[str]) -> str:
        key = match.group(1)
        if key not in params:
            raise DatasetBuildError("contract", f"unknown placeholder {{{key}}}")
        return str(params[key])

    return _PLACEHOLDER.sub(replace, template)


def quote_table(name: str) -> str:
    if _IDENT.match(name):
        return name
    parts = name.split(".")
    if not all(_IDENT.match(part) for part in parts):
        raise DatasetBuildError("contract", f"unsafe table name {name}")
    return ".".join(f"`{part}`" for part in parts)


def quote_column(name: str) -> str:
    if not _IDENT.match(name):
        raise DatasetBuildError("contract", f"unsafe column name {name}")
    return f"`{name}`"


def scan_window(spec: DatasetSpec, start: date, end: date) -> tuple[date, date]:
    lookback = max((group.lookback_days for group in spec.feature_groups.values()), default=0)
    pad_after_days = max(1, int((spec.label_window_delta().total_seconds() + 86399) // 86400))
    return start - timedelta(days=lookback), end + timedelta(days=pad_after_days)


def render_params(spec: DatasetSpec, start: date, end: date) -> dict[str, str]:
    scan_start, scan_end = scan_window(spec, start, end)
    params = {
        "start": start.isoformat(),
        "end": end.isoformat(),
        "scan_start": scan_start.isoformat(),
        "scan_end": scan_end.isoformat(),
        "entity_type": spec.entity_type or "",
    }
    for name in spec.sources:
        params[name] = f"amt_src_{name}"
    return params


def source_view_sql(spec: DatasetSpec, start: date, end: date) -> list[tuple[str, str]]:
    """``(view_name, select_sql)`` for each source, partition-pruned on the scan window."""

    scan_start, scan_end = scan_window(spec, start, end)
    views = []
    filter_params = {
        "start": scan_start.isoformat(),
        "end": scan_end.isoformat(),
    }
    for name, source in spec.sources.items():
        table = quote_table(source.table)
        if source.partition_filter:
            if source.date_col:
                filter_params["date_col"] = source.date_col
            predicate = render(source.partition_filter, filter_params)
            body = f"SELECT * FROM {table} WHERE {predicate}"
        else:
            body = f"SELECT * FROM {table}"
        views.append((f"amt_src_{name}", body))
    return views


def fragment_selects(spec: DatasetSpec, params: dict[str, str]) -> list[tuple[str, str]]:
    fragments = [
        ("queries", render(spec.sql_text(spec.queries), params)),
        ("candidates", render(spec.sql_text(spec.candidates), params)),
        ("labels", render(spec.sql_text(spec.labels.sql), params)),
    ]
    for name, group in spec.feature_groups.items():
        fragments.append((name, render(spec.sql_text(group.sql), params)))
    return fragments


def _alias(fragment: str) -> str:
    if fragment == "queries":
        return "q"
    if fragment == "candidates":
        return "c"
    if fragment == "labels":
        return "l"
    return f"g_{fragment}"


def _join_keys(group: FeatureGroupSpec, alias: str) -> str:
    parts = []
    for key in group.keys:
        other = "q" if key == "query_id" else "c"
        parts.append(f"{alias}.{quote_column(key)} = {other}.{quote_column(key)}")
    return " AND ".join(parts)


def assemble_select_sql(spec: DatasetSpec, columns: dict[str, list[str]]) -> str:
    """Contract SELECT over fragment views ``amt_frag_<name>``."""

    selects = [
        "q.query_id AS query_id",
        "c.item_id AS item_id",
        "q.query_ts AS query_ts",
        "to_date(q.query_ts) AS event_day",
        "CAST(l.label AS INT) AS label",
    ]
    label_columns = set(columns.get("labels", []))
    for aux in spec.labels.aux_labels:
        if aux not in label_columns:
            raise DatasetBuildError("contract", f"labels fragment has no aux column {aux}")
        selects.append(f"l.{quote_column(aux)} AS {aux_column_sql(aux)}")
    joins = [
        "FROM amt_frag_queries q",
        "INNER JOIN amt_frag_candidates c ON q.query_id = c.query_id",
        "INNER JOIN amt_frag_labels l ON l.query_id = c.query_id AND l.item_id = c.item_id",
    ]
    owners: dict[str, str] = {}
    for fragment, names in columns.items():
        for name in names:
            owners.setdefault(name, fragment)
    for name, group in spec.feature_groups.items():
        alias = _alias(name)
        joins.append(f"LEFT JOIN amt_frag_{name} {alias} ON {_join_keys(group, alias)}")
        skip = set(group.keys) | {group.available_at}
        for column in columns.get(name, []):
            if column in skip:
                continue
            selects.append(
                f"{alias}.{quote_column(column)} AS {quote_column(feat_column(name, column))}"
            )
    for ref_name, expr in spec.reference_scores.items():
        selects.append(f"{_ref_sql(expr, owners)} AS {quote_column(ref_column(ref_name))}")
    return "SELECT\n  " + ",\n  ".join(selects) + "\n" + "\n".join(joins)


def aux_column_sql(name: str) -> str:
    return quote_column(f"{AUX_PREFIX}{name}")


def _ref_sql(expr: str, owners: dict[str, str]) -> str:
    text = expr.strip()
    negative = text.startswith("-")
    column = text[1:].strip() if negative else text
    owner = owners.get(column)
    if owner is None:
        raise DatasetBuildError("contract", f"reference column {column} is not in any fragment")
    qualified = f"{_alias(owner)}.{quote_column(column)}"
    if negative:
        return f"(-1 * {qualified})"
    return qualified


def filtered_sql(select_sql: str, spec: DatasetSpec, start: date, end: date) -> str:
    predicates = [
        f"to_date(query_ts) BETWEEN DATE '{start.isoformat()}' AND DATE '{end.isoformat()}'"
    ]
    minimum = spec.include_min_label()
    if minimum is not None:
        predicates.append(
            "query_id IN (SELECT query_id FROM assembled GROUP BY query_id "
            f"HAVING MAX(label) >= {minimum})"
        )
    where = " AND ".join(predicates)
    return f"WITH assembled AS (\n{select_sql}\n)\nSELECT * FROM assembled WHERE {where}"


def violation_sqls(
    spec: DatasetSpec,
    columns: dict[str, list[str]],
) -> list[tuple[str, str]]:
    """Queries that return example rows when the contract is violated."""

    seconds = int(spec.label_window_delta().total_seconds())
    statements: list[tuple[str, str]] = [
        (
            "contract",
            """
            SELECT query_id, COUNT(*) AS n
            FROM amt_frag_queries
            GROUP BY query_id
            HAVING COUNT(*) > 1
            LIMIT 5
            """.strip(),
        ),
        (
            "contract",
            """
            SELECT query_id, item_id, COUNT(*) AS n
            FROM amt_frag_labels
            GROUP BY query_id, item_id
            HAVING COUNT(*) > 1
            LIMIT 5
            """.strip(),
        ),
        (
            "contract",
            """
            SELECT c.query_id, c.item_id
            FROM amt_frag_candidates c
            LEFT JOIN amt_frag_labels l
              ON l.query_id = c.query_id AND l.item_id = c.item_id
            WHERE l.query_id IS NULL
            LIMIT 5
            """.strip(),
        ),
        (
            "label_window",
            f"""
            SELECT l.query_id, l.item_id, l.label_event_ts, q.query_ts
            FROM amt_frag_labels l
            INNER JOIN amt_frag_queries q ON q.query_id = l.query_id
            WHERE l.label_event_ts < q.query_ts
               OR l.label_event_ts > q.query_ts + INTERVAL {seconds} SECONDS
            LIMIT 5
            """.strip(),
        ),
    ]
    for name, group in spec.feature_groups.items():
        keys = ", ".join(quote_column(key) for key in group.keys)
        statements.append(
            (
                "contract",
                f"""
                SELECT {keys}, COUNT(*) AS n
                FROM amt_frag_{name}
                GROUP BY {keys}
                HAVING COUNT(*) > 1
                LIMIT 5
                """.strip(),
            )
        )
        alias = _alias(name)
        value_columns = [
            column
            for column in columns.get(name, [])
            if column not in set(group.keys) | {group.available_at}
        ]
        if not value_columns:
            continue
        populated = " OR ".join(
            f"{alias}.{quote_column(column)} IS NOT NULL" for column in value_columns
        )
        available = f"{alias}.{quote_column(group.available_at)}"
        statements.append(
            (
                "point_in_time",
                f"""
                SELECT q.query_id, c.item_id, {available} AS available_at, q.query_ts
                FROM amt_frag_candidates c
                INNER JOIN amt_frag_queries q ON q.query_id = c.query_id
                INNER JOIN amt_frag_{name} {alias} ON {_join_keys(group, alias)}
                WHERE ({available} > q.query_ts OR {available} IS NULL)
                  AND ({populated})
                LIMIT 5
                """.strip(),
            )
        )
    return statements


def candidate_bounds_sql(spec: DatasetSpec, select_sql: str, start: date, end: date) -> str:
    """Example queries that fall outside ``candidate_bounds`` after the split filter."""

    predicates = [
        f"to_date(query_ts) BETWEEN DATE '{start.isoformat()}' AND DATE '{end.isoformat()}'"
    ]
    minimum = spec.include_min_label()
    if minimum is not None:
        predicates.append(
            "query_id IN (SELECT query_id FROM assembled GROUP BY query_id "
            f"HAVING MAX(label) >= {minimum})"
        )
    where = " AND ".join(predicates)
    upper = ""
    if spec.candidate_bounds.max is not None:
        upper = f" OR n > {int(spec.candidate_bounds.max)}"
    return (
        "WITH assembled AS (\n"
        f"{select_sql}\n"
        ")\n"
        "SELECT query_id, n FROM (\n"
        "  SELECT query_id, COUNT(*) AS n FROM assembled\n"
        f"  WHERE {where}\n"
        "  GROUP BY query_id\n"
        f") AS counts WHERE n < {int(spec.candidate_bounds.min)}{upper}\n"
        "LIMIT 5"
    )
