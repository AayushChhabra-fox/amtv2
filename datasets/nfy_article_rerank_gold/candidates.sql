-- The articles impressed for the request, the first 50 in production rank
-- order. Requests outside queries.sql are dropped when the builder joins.
WITH ranked AS (
  SELECT
    request_id,
    entity_id,
    rank,
    item_age_seconds,
    ROW_NUMBER() OVER (PARTITION BY request_id ORDER BY rank, entity_id) AS slot
  FROM {gold}
  WHERE request_ts >= TIMESTAMP '{start_ts}'
    AND request_ts < TIMESTAMP '{end_ts}'
)
SELECT
  request_id AS query_id,
  entity_id AS item_id,
  CAST(-rank AS DOUBLE) AS neg_rank,
  CASE
    WHEN item_age_seconds IS NULL OR item_age_seconds < 0 THEN -1000000.0
    ELSE -item_age_seconds / 3600.0
  END AS neg_item_age_hours
FROM ranked
WHERE slot <= 50
