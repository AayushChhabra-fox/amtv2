-- Session context at query_ts: time of day and week, and the device of the
-- session's first For You impression.
WITH sessions AS (
  SELECT
    xid,
    session_id,
    MIN(event_created) AS query_ts,
    MIN_BY(device_type, event_created) AS device_type,
    MIN_BY(os_name, event_created) AS os_name
  FROM {impressions}
  WHERE pmod(xxhash64(xid), 4) = 0
    AND session_id IS NOT NULL
  GROUP BY xid, session_id
  HAVING MIN(event_created) >= TIMESTAMP '{start_ts}'
     AND MIN(event_created) < TIMESTAMP '{end_ts}'
)
SELECT
  concat(xid, ':', session_id) AS query_id,
  query_ts,
  CAST(HOUR(query_ts) AS DOUBLE) AS hour_of_day,
  CAST(DAYOFWEEK(query_ts) AS DOUBLE) AS day_of_week,
  device_type,
  os_name
FROM sessions
