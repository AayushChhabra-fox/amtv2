-- Article activity over all users in the 24 hours (and the last hour) before
-- the start of the query's hour: For You impressions and clicks, and reads on
-- any surface. Counts sit on a dense article x hour grid, so each pair joins
-- one grid row. activity_window_end is one second before the query's hour.
WITH sessions AS (
  SELECT xid, session_id, MIN(event_created) AS query_ts
  FROM {impressions}
  WHERE pmod(xxhash64(xid), 4) = 0
    AND session_id IS NOT NULL
  GROUP BY xid, session_id
  HAVING MIN(event_created) >= TIMESTAMP '{start_ts}'
     AND MIN(event_created) < TIMESTAMP '{end_ts}'
),
pairs AS (
  SELECT DISTINCT s.xid, s.session_id, s.query_ts, i.entity_id
  FROM sessions AS s
  INNER JOIN {impressions} AS i
    ON i.xid = s.xid
   AND i.session_id = s.session_id
   AND i.event_created >= s.query_ts
   AND i.event_created < s.query_ts + INTERVAL 30 MINUTES
),
items AS (
  SELECT DISTINCT entity_id FROM pairs
),
hours AS (
  SELECT explode(sequence(
    TIMESTAMP '{scan_start} 00:00:00',
    TIMESTAMP '{end} 23:00:00',
    INTERVAL 1 HOUR
  )) AS hour_ts
),
shown_hourly AS (
  SELECT
    i.entity_id,
    date_trunc('HOUR', i.event_created) AS hour_ts,
    COUNT(*) AS n_shown,
    SUM(CASE WHEN i.impression_clicked THEN 1 ELSE 0 END) AS n_clicked
  FROM {impressions} AS i
  INNER JOIN items AS it ON it.entity_id = i.entity_id
  GROUP BY i.entity_id, date_trunc('HOUR', i.event_created)
),
read_hourly AS (
  SELECT
    ix.entity_id,
    date_trunc('HOUR', ix.event_created) AS hour_ts,
    COUNT(*) AS n_reads,
    SUM(CASE WHEN ix.read_quality = 'deep_read' THEN 1 ELSE 0 END) AS n_deep
  FROM {interactions} AS ix
  INNER JOIN items AS it ON it.entity_id = ix.entity_id
  GROUP BY ix.entity_id, date_trunc('HOUR', ix.event_created)
),
grid AS (
  SELECT
    it.entity_id,
    h.hour_ts,
    COALESCE(sh.n_shown, 0) AS n_shown,
    COALESCE(sh.n_clicked, 0) AS n_clicked,
    COALESCE(rh.n_reads, 0) AS n_reads,
    COALESCE(rh.n_deep, 0) AS n_deep
  FROM items AS it
  CROSS JOIN hours AS h
  LEFT JOIN shown_hourly AS sh
    ON sh.entity_id = it.entity_id AND sh.hour_ts = h.hour_ts
  LEFT JOIN read_hourly AS rh
    ON rh.entity_id = it.entity_id AND rh.hour_ts = h.hour_ts
),
windowed AS (
  SELECT
    entity_id,
    hour_ts,
    SUM(n_shown) OVER w24 AS shown_24h,
    SUM(n_clicked) OVER w24 AS clicks_24h,
    SUM(n_reads) OVER w24 AS reads_24h,
    SUM(n_deep) OVER w24 AS deep_reads_24h,
    SUM(n_shown) OVER w1 AS shown_1h,
    SUM(n_reads) OVER w1 AS reads_1h
  FROM grid
  WINDOW
    w24 AS (PARTITION BY entity_id ORDER BY hour_ts ROWS BETWEEN 24 PRECEDING AND 1 PRECEDING),
    w1 AS (PARTITION BY entity_id ORDER BY hour_ts ROWS BETWEEN 1 PRECEDING AND 1 PRECEDING)
)
SELECT
  concat(p.xid, ':', p.session_id) AS query_id,
  p.entity_id AS item_id,
  date_trunc('HOUR', p.query_ts) - INTERVAL 1 SECOND AS activity_window_end,
  CAST(COALESCE(w.shown_24h, 0) AS DOUBLE) AS foryou_shown_24h,
  CAST(COALESCE(w.clicks_24h, 0) AS DOUBLE) AS foryou_clicks_24h,
  (COALESCE(w.clicks_24h, 0) + 1.0) / (COALESCE(w.shown_24h, 0) + 50.0) AS foryou_ctr_24h,
  CAST(COALESCE(w.reads_24h, 0) AS DOUBLE) AS reads_24h,
  CAST(COALESCE(w.deep_reads_24h, 0) AS DOUBLE) AS deep_reads_24h,
  CAST(COALESCE(w.shown_1h, 0) AS DOUBLE) AS foryou_shown_1h,
  CAST(COALESCE(w.reads_1h, 0) AS DOUBLE) AS reads_1h
FROM pairs AS p
LEFT JOIN windowed AS w
  ON w.entity_id = p.entity_id
 AND w.hour_ts = date_trunc('HOUR', p.query_ts)
