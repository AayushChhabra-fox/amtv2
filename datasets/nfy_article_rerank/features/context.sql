-- Request context. available_at is the request timestamp itself.
SELECT
  r.request_id AS query_id,
  r.request_ts AS query_ts,
  CAST(HOUR(r.request_ts) AS DOUBLE) AS hour_of_day,
  CAST(r.dma AS DOUBLE) AS dma,
  CASE WHEN r.cold_start THEN 1.0 ELSE 0.0 END AS cold_start
FROM {requests} AS r
WHERE r.entity_type = '{entity_type}'
  AND r.event_date BETWEEN DATE '{start}' AND DATE '{end}'
  AND r.request_ts >= TIMESTAMP '{start_ts}'
  AND r.request_ts < TIMESTAMP '{end_ts}'
