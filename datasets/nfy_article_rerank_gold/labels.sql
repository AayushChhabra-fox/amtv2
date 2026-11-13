-- relevance_grade from the gold table. The table has no read timestamp, so the
-- label is stamped at the matched impression, at most 30 minutes after the
-- request. Impressions logged up to two minutes before request_ts (clock skew)
-- are stamped at request_ts.
SELECT
  request_id AS query_id,
  entity_id AS item_id,
  relevance_grade AS label,
  CASE WHEN clicked OR relevance_grade >= 1 THEN 1 ELSE 0 END AS click,
  GREATEST(request_ts, COALESCE(impression_ts, request_ts)) AS label_event_ts
FROM {gold}
WHERE request_ts >= TIMESTAMP '{start_ts}'
  AND request_ts < TIMESTAMP '{end_ts}'
