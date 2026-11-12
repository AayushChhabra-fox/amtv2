-- Independent click NDCG@10 of total_score on impressed For You pairs.
-- The builder compares this scalar to the harness metric within tol.
WITH impressed AS (
  SELECT
    c.request_id AS query_id,
    c.entity_id AS item_id,
    MAX(c.total_score) AS total_score,
    MAX(
      CASE
        WHEN ix.read_quality = 'deep_read' THEN 1
        WHEN ix.read_quality = 'engaged_read' THEN 1
        WHEN ix.read_quality = 'bounce' THEN 1
        WHEN i.impression_clicked THEN 1
        ELSE 0
      END
    ) AS click
  FROM {candidates} AS c
  INNER JOIN {requests} AS r
    ON r.request_id = c.request_id
   AND r.entity_type = '{entity_type}'
  INNER JOIN {impressions} AS i
    ON i.xid = c.xid
   AND i.entity_id = c.entity_id
   AND i.entity_type = '{entity_type}'
   AND i.page_type = 'for-you'
   AND i.latest_impression_rn = 1
   AND i.timestamp >= c.request_ts
   AND i.timestamp < c.request_ts + INTERVAL 30 MINUTES
  LEFT JOIN {interactions} AS ix
    ON ix.xid = c.xid
   AND ix.entity_id = c.entity_id
   AND ix.event_created >= c.request_ts
   AND ix.event_created < c.request_ts + INTERVAL 24 HOURS
   AND ix.read_quality IN ('bounce', 'engaged_read', 'deep_read')
  WHERE c.candidate_status = 'served'
    AND c.entity_type = '{entity_type}'
    AND c.event_date BETWEEN DATE '{start}' AND DATE '{end}'
    AND c.request_ts >= TIMESTAMP '{start_ts}'
    AND c.request_ts < TIMESTAMP '{end_ts}'
  GROUP BY c.request_id, c.entity_id
),
kept AS (
  SELECT imp.*
  FROM impressed AS imp
  INNER JOIN (
    SELECT query_id
    FROM impressed
    GROUP BY query_id
    HAVING MAX(click) >= 1
  ) AS clicked
    ON clicked.query_id = imp.query_id
),
ranked AS (
  SELECT
    query_id,
    click,
    ROW_NUMBER() OVER (
      PARTITION BY query_id
      ORDER BY total_score DESC, item_id
    ) AS rn
  FROM kept
),
ideal AS (
  SELECT
    query_id,
    click,
    ROW_NUMBER() OVER (
      PARTITION BY query_id
      ORDER BY click DESC, item_id
    ) AS rn
  FROM kept
),
dcg AS (
  SELECT
    query_id,
    SUM(CASE WHEN rn <= 10 THEN click / LOG(2, rn + 1) ELSE 0 END) AS dcg
  FROM ranked
  GROUP BY query_id
),
idcg AS (
  SELECT
    query_id,
    SUM(CASE WHEN rn <= 10 THEN click / LOG(2, rn + 1) ELSE 0 END) AS idcg
  FROM ideal
  GROUP BY query_id
)
SELECT AVG(d.dcg / i.idcg) AS metric
FROM dcg AS d
INNER JOIN idcg AS i
  ON i.query_id = d.query_id
WHERE i.idcg > 0
