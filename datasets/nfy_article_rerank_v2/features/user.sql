-- The user's activity in the 24 hours before the start of the query's hour:
-- reads on any surface, and For You article impressions and clicks. Counts are
-- per user and hour first, so a very active xid adds at most 24 rows per session.
-- user_window_end is one second before the query's hour.
WITH sessions AS (
  SELECT xid, session_id, MIN(event_created) AS query_ts
  FROM {impressions}
  WHERE pmod(xxhash64(xid), 4) = 0
    AND session_id IS NOT NULL
  GROUP BY xid, session_id
  HAVING MIN(event_created) >= TIMESTAMP '{start_ts}'
     AND MIN(event_created) < TIMESTAMP '{end_ts}'
),
read_hourly AS (
  SELECT
    xid,
    date_trunc('HOUR', event_created) AS hour_ts,
    COUNT(*) AS n_reads,
    SUM(CASE WHEN read_quality = 'deep_read' THEN 1 ELSE 0 END) AS n_deep,
    COUNT(DISTINCT category_slug) AS n_categories
  FROM {interactions}
  WHERE pmod(xxhash64(xid), 4) = 0
  GROUP BY xid, date_trunc('HOUR', event_created)
),
shown_hourly AS (
  SELECT
    xid,
    date_trunc('HOUR', event_created) AS hour_ts,
    COUNT(*) AS n_shown,
    SUM(CASE WHEN impression_clicked THEN 1 ELSE 0 END) AS n_clicked
  FROM {impressions}
  WHERE pmod(xxhash64(xid), 4) = 0
  GROUP BY xid, date_trunc('HOUR', event_created)
),
reads AS (
  SELECT
    s.xid,
    s.session_id,
    SUM(r.n_reads) AS reads_24h,
    SUM(r.n_deep) AS deep_reads_24h,
    MAX(r.n_categories) AS max_categories_per_hour
  FROM sessions AS s
  INNER JOIN read_hourly AS r
    ON r.xid = s.xid
   AND r.hour_ts >= date_trunc('HOUR', s.query_ts) - INTERVAL 24 HOURS
   AND r.hour_ts < date_trunc('HOUR', s.query_ts)
  GROUP BY s.xid, s.session_id
),
shown AS (
  SELECT
    s.xid,
    s.session_id,
    SUM(h.n_shown) AS foryou_shown_24h,
    SUM(h.n_clicked) AS foryou_clicks_24h
  FROM sessions AS s
  INNER JOIN shown_hourly AS h
    ON h.xid = s.xid
   AND h.hour_ts >= date_trunc('HOUR', s.query_ts) - INTERVAL 24 HOURS
   AND h.hour_ts < date_trunc('HOUR', s.query_ts)
  GROUP BY s.xid, s.session_id
)
SELECT
  concat(s.xid, ':', s.session_id) AS query_id,
  date_trunc('HOUR', s.query_ts) - INTERVAL 1 SECOND AS user_window_end,
  CAST(COALESCE(r.reads_24h, 0) AS DOUBLE) AS user_reads_24h,
  CAST(COALESCE(r.deep_reads_24h, 0) AS DOUBLE) AS user_deep_reads_24h,
  CAST(COALESCE(r.max_categories_per_hour, 0) AS DOUBLE) AS user_max_categories_per_hour,
  CAST(COALESCE(h.foryou_shown_24h, 0) AS DOUBLE) AS user_foryou_shown_24h,
  CAST(COALESCE(h.foryou_clicks_24h, 0) AS DOUBLE) AS user_foryou_clicks_24h,
  CASE WHEN r.reads_24h IS NULL AND h.foryou_shown_24h IS NULL THEN 1.0 ELSE 0.0 END
    AS user_new_24h
FROM sessions AS s
LEFT JOIN reads AS r
  ON r.xid = s.xid AND r.session_id = s.session_id
LEFT JOIN shown AS h
  ON h.xid = s.xid AND h.session_id = s.session_id
