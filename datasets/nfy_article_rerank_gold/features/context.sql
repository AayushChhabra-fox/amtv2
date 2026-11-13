-- Request context: time of day, weekday, and how many items the request asked
-- for and retrieval returned.
SELECT
  request_id AS query_id,
  MIN(request_ts) AS feature_ts,
  CAST(MAX(hr) AS DOUBLE) AS hour_of_day,
  CAST(MAX(day_of_week) AS DOUBLE) AS day_of_week,
  CAST(MAX(max_items) AS DOUBLE) AS max_items,
  CAST(MAX(candidates_raw) AS DOUBLE) AS candidates_raw
FROM {gold}
WHERE request_ts >= TIMESTAMP '{start_ts}'
  AND request_ts < TIMESTAMP '{end_ts}'
GROUP BY request_id
