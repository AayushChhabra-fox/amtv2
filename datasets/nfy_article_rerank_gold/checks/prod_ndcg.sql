-- Independent graded NDCG@10 of the production rank on search_val, rebuilt
-- from the source without the fragments. Gains are 2^grade - 1; the ideal
-- order sorts by grade. The builder compares this scalar to the harness metric.
WITH ranked AS (
  SELECT
    request_id,
    entity_id,
    relevance_grade AS grade,
    ROW_NUMBER() OVER (PARTITION BY request_id ORDER BY rank, entity_id) AS served,
    ROW_NUMBER() OVER (PARTITION BY request_id ORDER BY relevance_grade DESC, entity_id) AS ideal
  FROM (
    SELECT *, ROW_NUMBER() OVER (PARTITION BY request_id ORDER BY rank, entity_id) AS slot
    FROM {gold}
    WHERE request_ts >= TIMESTAMP '{start_ts}'
      AND request_ts < TIMESTAMP '{end_ts}'
  ) AS capped
  WHERE slot <= 50
),
per_request AS (
  SELECT
    request_id,
    COUNT(*) AS n,
    MAX(grade) AS best,
    SUM(CASE WHEN served <= 10 THEN (POW(2, grade) - 1) / LOG(2, served + 1) ELSE 0 END) AS dcg,
    SUM(CASE WHEN ideal <= 10 THEN (POW(2, grade) - 1) / LOG(2, ideal + 1) ELSE 0 END) AS idcg
  FROM ranked
  GROUP BY request_id
)
SELECT AVG(dcg / idcg) AS metric
FROM per_request
WHERE n >= 2 AND best >= 1
