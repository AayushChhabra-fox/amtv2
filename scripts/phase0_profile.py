"""Read-only Phase 0 checks against the NFY tables and cluster history.

Writes docs/phase0_profile.json. Identifiers used in SQL are taken from
information_schema and must match ^[A-Za-z_][A-Za-z0-9_]*$.
"""

from __future__ import annotations

import json
import os
import re
import time
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
WAREHOUSE_ID = "25341e4f3501d03c"
DAY = "2026-11-10"
IDENT = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
TABLES = {
    "news_interactions": "pnr_prod.silver.news_interactions",
    "news_impressions": "pnr_prod.silver.news_impressions",
    "nfy_reco_requests": "pnr_prod.silver.nfy_reco_requests",
    "nfy_user_profile_versions": "pnr_prod.silver.nfy_user_profile_versions",
}


def load_env() -> None:
    for raw in (ROOT / ".env").read_text().splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or ":" not in line:
            continue
        key, value = line.split(":", 1)
        key, value = key.strip(), value.strip()
        if key == "dbx_token":
            os.environ["DATABRICKS_TOKEN"] = value
        elif key == "dbx_host":
            os.environ["DATABRICKS_HOST"] = value.rstrip("/")


def api(method: str, path: str, body: dict | None = None) -> dict:
    host = os.environ["DATABRICKS_HOST"]
    token = os.environ["DATABRICKS_TOKEN"]
    data = None if body is None else json.dumps(body).encode()
    request = urllib.request.Request(
        host + path,
        data=data,
        method=method,
        headers={"Authorization": f"Bearer {token}", "Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(request, timeout=120) as response:
            return json.load(response)
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode()[:1500]
        raise RuntimeError(f"HTTP {exc.code} {path}: {detail}") from None


def sql(statement: str) -> list[dict]:
    payload = api(
        "POST",
        "/api/2.0/sql/statements",
        {
            "warehouse_id": WAREHOUSE_ID,
            "statement": statement,
            "wait_timeout": "50s",
            "disposition": "INLINE",
            "format": "JSON_ARRAY",
            "on_wait_timeout": "CONTINUE",
        },
    )
    deadline = time.time() + 480
    while payload.get("status", {}).get("state") in {"PENDING", "RUNNING"}:
        if time.time() > deadline:
            raise TimeoutError(statement[:120])
        time.sleep(3)
        payload = api("GET", f"/api/2.0/sql/statements/{payload['statement_id']}")
    state = payload.get("status", {}).get("state")
    if state != "SUCCEEDED":
        raise RuntimeError(f"{state}: {payload.get('status', {}).get('error')}")
    columns = [
        column.get("name")
        for column in ((payload.get("manifest") or {}).get("schema") or {}).get("columns") or []
    ]
    rows = (payload.get("result") or {}).get("data_array") or []
    return [dict(zip(columns, row)) for row in rows]


def run(summary: dict, label: str, statement: str) -> list[dict] | None:
    print(f"RUN {label}", flush=True)
    try:
        rows = sql(statement)
    except Exception as exc:
        summary["queries"].append({"label": label, "ok": False, "error": str(exc)[:1500]})
        print(f"FAIL {label}: {exc}"[:400], flush=True)
        return None
    summary["queries"].append({"label": label, "ok": True, "n": len(rows)})
    print(f"OK {label} rows={len(rows)}", flush=True)
    return rows


def qident(name: str) -> str:
    if not IDENT.match(name):
        raise ValueError(f"unsafe identifier {name}")
    return name


def names(columns: dict[str, list[dict]], table: str) -> list[str]:
    return [row["column_name"] for row in columns.get(table, [])]


def pick(columns: dict[str, list[dict]], table: str, options: list[str]) -> str | None:
    have = set(names(columns, table))
    for option in options:
        if option in have:
            return option
    return None


def partition_column(detail: dict | None, column_names: list[str]) -> str | None:
    blob = " ".join(str(value) for value in (detail or {}).values()).lower()
    for candidate in (
        "event_date",
        "date",
        "ds",
        "impression_date",
        "request_date",
        "event_day",
        "ingested_date",
    ):
        if candidate in column_names and candidate in blob:
            return candidate
    for candidate in ("event_date", "date", "ds", "impression_date", "request_date", "event_day"):
        if candidate in column_names:
            return candidate
    return None


def main() -> None:
    load_env()
    summary: dict = {"day": DAY, "queries": []}
    column_rows = run(
        summary,
        "columns",
        """
        SELECT table_name, column_name, data_type, comment
        FROM pnr_prod.information_schema.columns
        WHERE table_schema = 'silver'
          AND table_name IN (
            'news_interactions', 'news_impressions',
            'nfy_reco_requests', 'nfy_user_profile_versions'
          )
        ORDER BY table_name, ordinal_position
        """,
    )
    columns: dict[str, list[dict]] = {}
    for row in column_rows or []:
        columns.setdefault(row["table_name"], []).append(row)
    summary["column_counts"] = {table: len(rows) for table, rows in columns.items()}
    tokens = (
        "embed",
        "impression",
        "rn",
        "profile",
        "hash",
        "dwell",
        "read",
        "time",
        "date",
        "entity",
        "xid",
        "session",
        "page",
        "click",
        "quality",
        "request",
    )
    summary["interesting_columns"] = {
        table: [
            {"name": row["column_name"], "type": row["data_type"], "comment": row.get("comment")}
            for row in rows
            if any(token in row["column_name"].lower() for token in tokens) or row.get("comment")
        ]
        for table, rows in columns.items()
    }

    details: dict[str, dict] = {}
    for key, full_name in TABLES.items():
        rows = run(summary, f"detail_{key}", f"DESCRIBE DETAIL {full_name}")
        if not rows:
            continue
        details[key] = {
            field: rows[0].get(field)
            for field in rows[0]
            if "partition" in field.lower() or field in {"name", "format", "numFiles"}
        }
    summary["partitions"] = details

    summary["system_compute_tables"] = run(
        summary, "system_compute_tables", "SHOW TABLES IN system.compute"
    )
    summary["gpu_node_types"] = run(
        summary,
        "gpu_node_types",
        """
        SELECT driver_node_type, worker_node_type, policy_id, count(*) AS n
        FROM system.compute.clusters
        WHERE create_time > current_timestamp() - INTERVAL 45 DAYS
          AND (
            lower(coalesce(driver_node_type, '')) RLIKE 'g4|g5|g6|p3|p4|p5|gpu'
            OR lower(coalesce(worker_node_type, '')) RLIKE 'g4|g5|g6|p3|p4|p5|gpu'
          )
        GROUP BY 1, 2, 3
        ORDER BY n DESC
        LIMIT 40
        """,
    )
    summary["recent_node_types"] = run(
        summary,
        "recent_node_types",
        """
        SELECT driver_node_type, worker_node_type, policy_id, count(*) AS n
        FROM system.compute.clusters
        WHERE create_time > current_timestamp() - INTERVAL 21 DAYS
        GROUP BY 1, 2, 3
        ORDER BY n DESC
        LIMIT 25
        """,
    )

    part = {
        "news_interactions": partition_column(
            details.get("news_interactions"), names(columns, "news_interactions")
        ),
        "news_impressions": partition_column(
            details.get("news_impressions"), names(columns, "news_impressions")
        ),
        "nfy_reco_requests": partition_column(
            details.get("nfy_reco_requests"), names(columns, "nfy_reco_requests")
        ),
    }
    summary["chosen_partitions"] = part
    _write(summary)

    entity = pick(
        columns,
        "news_interactions",
        ["entity_id", "article_id", "content_id", "item_id"],
    )
    session = pick(
        columns,
        "news_interactions",
        ["session_id", "xid", "user_session_id", "context_id", "request_id"],
    )
    summary["embedding_keys"] = {
        "entity": entity,
        "session": session,
        "partition": part.get("news_interactions"),
    }
    if "context_embedding_3072" in names(columns, "news_interactions"):
        summary["embedding_comment"] = next(
            (
                row.get("comment")
                for row in columns.get("news_interactions", [])
                if row["column_name"] == "context_embedding_3072"
            ),
            None,
        )
    interaction_part = part.get("news_interactions")
    has_embedding = "context_embedding_3072" in names(columns, "news_interactions")
    if interaction_part and entity and has_embedding:
        part_sql = f"{qident(interaction_part)} = DATE '{DAY}'"
        session_sql = qident(session) if session else "'all'"
        summary["embedding_grain"] = run(
            summary,
            "embedding_grain",
            f"""
            WITH sample AS (
              SELECT {qident(entity)} AS entity_id,
                     {session_sql} AS session_id,
                     hash(context_embedding_3072) AS h
              FROM pnr_prod.silver.news_interactions
              WHERE {part_sql} AND context_embedding_3072 IS NOT NULL
              LIMIT 20000
            ),
            by_entity AS (
              SELECT avg(n_h) AS avg_hashes_per_entity,
                     percentile_approx(n_h, 0.5) AS median_hashes_per_entity,
                     percentile_approx(n_h, 0.9) AS p90_hashes_per_entity
              FROM (
                SELECT entity_id, count(DISTINCT h) AS n_h
                FROM sample GROUP BY entity_id
              )
            ),
            by_session AS (
              SELECT avg(n_h) AS avg_hashes_per_session,
                     percentile_approx(n_h, 0.5) AS median_hashes_per_session
              FROM (
                SELECT session_id, count(DISTINCT h) AS n_h
                FROM sample GROUP BY session_id
              )
            )
            SELECT * FROM by_entity CROSS JOIN by_session
            """,
        )

    impression_part = part.get("news_impressions")
    if impression_part and "latest_impression_rn" in names(columns, "news_impressions"):
        where_imp = f"{qident(impression_part)} = DATE '{DAY}'"
        summary["impression_rn"] = run(
            summary,
            "impression_rn",
            f"""
            SELECT latest_impression_rn, count(*) AS n
            FROM pnr_prod.silver.news_impressions
            WHERE {where_imp}
            GROUP BY 1
            ORDER BY 1
            LIMIT 15
            """,
        )
        impression_id = pick(
            columns, "news_impressions", ["impression_id", "event_id", "message_id"]
        )
        summary["impression_id_col"] = impression_id
        if impression_id:
            summary["impression_deduped"] = run(
                summary,
                "impression_deduped",
                f"""
                SELECT count(*) AS ids,
                       avg(cnt) AS avg_rows,
                       max(cnt) AS max_rows
                FROM (
                  SELECT {qident(impression_id)}, count(*) AS cnt
                  FROM pnr_prod.silver.news_impressions
                  WHERE {where_imp} AND latest_impression_rn = 1
                  GROUP BY 1
                )
                """,
            )

    request_key = pick(columns, "nfy_reco_requests", ["user_profile_key", "profile_key"])
    request_hash = pick(columns, "nfy_reco_requests", ["profile_content_hash", "profile_hash"])
    profile_key = pick(columns, "nfy_user_profile_versions", ["user_profile_key", "profile_key"])
    profile_hash = pick(
        columns, "nfy_user_profile_versions", ["profile_content_hash", "profile_hash"]
    )
    entity_type = pick(columns, "nfy_reco_requests", ["entity_type"])
    request_part = part.get("nfy_reco_requests")
    summary["profile_join_keys"] = {
        "request_key": request_key,
        "request_hash": request_hash,
        "profile_key": profile_key,
        "profile_hash": profile_hash,
        "entity_type": entity_type,
        "partition": request_part,
    }
    if request_part and request_key and request_hash and profile_key and profile_hash:
        entity_filter = ""
        if entity_type:
            entity_filter = f"AND {qident(entity_type)} = 'article'"
        summary["profile_join"] = run(
            summary,
            "profile_join",
            f"""
            WITH req AS (
              SELECT {qident(request_key)} AS user_profile_key,
                     {qident(request_hash)} AS profile_content_hash
              FROM pnr_prod.silver.nfy_reco_requests
              WHERE {qident(request_part)} = DATE '{DAY}'
              {entity_filter}
            ),
            keys AS (
              SELECT DISTINCT user_profile_key, profile_content_hash FROM req
            )
            SELECT (SELECT count(*) FROM req) AS requests,
                   (SELECT count(*) FROM keys) AS distinct_keys,
                   (SELECT count(*) FROM keys k
                     WHERE EXISTS (
                       SELECT 1
                       FROM pnr_prod.silver.nfy_user_profile_versions p
                       WHERE p.{qident(profile_key)} = k.user_profile_key
                         AND p.{qident(profile_hash)} = k.profile_content_hash
                     )) AS matched_keys,
                   (SELECT count(*) FROM req r
                     WHERE r.user_profile_key IS NULL
                        OR r.profile_content_hash IS NULL) AS missing_key
            """,
        )

    dwell_tokens = ("dwell", "read_quality", "engaged", "deep")
    summary["dwell_columns"] = [
        row["column_name"]
        for row in columns.get("news_interactions", [])
        if any(token in row["column_name"].lower() for token in dwell_tokens)
    ]
    interaction_times = [
        "event_ts",
        "event_timestamp",
        "interaction_ts",
        "client_timestamp",
        "occurred_at",
        "ts",
    ]
    impression_times = [
        "event_ts",
        "event_timestamp",
        "impression_ts",
        "client_timestamp",
        "occurred_at",
        "ts",
    ]
    summary["time_columns"] = {
        "interaction": pick(columns, "news_interactions", interaction_times),
        "impression": pick(columns, "news_impressions", impression_times),
        "request": pick(
            columns,
            "nfy_reco_requests",
            ["request_ts", "request_timestamp", "event_ts", "event_timestamp", "ts"],
        ),
    }
    interaction_ts = summary["time_columns"]["interaction"]
    impression_ts = summary["time_columns"]["impression"]
    join_entity = pick(columns, "news_interactions", ["entity_id", "article_id", "content_id"])
    join_session = pick(columns, "news_interactions", ["xid", "session_id"])
    imp_entity = pick(columns, "news_impressions", ["entity_id", "article_id", "content_id"])
    imp_session = pick(columns, "news_impressions", ["xid", "session_id"])
    if (
        interaction_ts
        and impression_ts
        and join_entity
        and join_session
        and imp_entity
        and imp_session
        and part.get("news_interactions")
        and part.get("news_impressions")
    ):
        hours = (
            "(unix_timestamp(s.interaction_ts) - "
            f"unix_timestamp(imp.{qident(impression_ts)})) / 3600.0"
        )
        summary["dwell_delay_hours"] = run(
            summary,
            "dwell_delay",
            f"""
            WITH sample AS (
              SELECT {qident(join_entity)} AS entity_id,
                     {qident(join_session)} AS session_id,
                     {qident(interaction_ts)} AS interaction_ts
              FROM pnr_prod.silver.news_interactions
              WHERE {qident(part['news_interactions'])} = DATE '{DAY}'
              LIMIT 20000
            )
            SELECT percentile_approx(hours, array(0.5, 0.9, 0.95, 0.99)) AS pct,
                   avg(CASE WHEN hours <= 1 THEN 1.0 ELSE 0.0 END) AS within_1h,
                   avg(CASE WHEN hours <= 6 THEN 1.0 ELSE 0.0 END) AS within_6h,
                   avg(CASE WHEN hours <= 24 THEN 1.0 ELSE 0.0 END) AS within_24h,
                   avg(CASE WHEN hours <= 48 THEN 1.0 ELSE 0.0 END) AS within_48h,
                   count(*) AS n
            FROM (
              SELECT {hours} AS hours
              FROM sample s
              JOIN pnr_prod.silver.news_impressions imp
                ON s.entity_id = imp.{qident(imp_entity)}
               AND s.session_id = imp.{qident(imp_session)}
               AND imp.{qident(part['news_impressions'])} = DATE '{DAY}'
              WHERE imp.{qident(impression_ts)} <= s.interaction_ts
            )
            """,
        )
    _write(summary)
    print("WROTE", ROOT / "docs" / "phase0_profile.json", flush=True)


def _write(summary: dict) -> None:
    out = ROOT / "docs" / "phase0_profile.json"
    out.write_text(json.dumps(summary, indent=2, default=str))


if __name__ == "__main__":
    main()
