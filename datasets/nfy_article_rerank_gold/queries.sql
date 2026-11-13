-- One NFY article request with at least two impressed articles and a positive
-- grade among the first 50 in production rank order.
WITH ranked AS (
  SELECT
    request_id,
    request_ts,
    relevance_grade,
    ROW_NUMBER() OVER (PARTITION BY request_id ORDER BY rank, entity_id) AS slot
  FROM {gold}
  WHERE request_ts >= TIMESTAMP '{start_ts}'
    AND request_ts < TIMESTAMP '{end_ts}'
)
SELECT
  request_id AS query_id,
  MIN(request_ts) AS query_ts
FROM ranked
WHERE slot <= 50
GROUP BY request_id
HAVING COUNT(*) >= 2
   AND MAX(relevance_grade) >= 1
