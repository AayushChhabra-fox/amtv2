-- Independent click NDCG@10 of the For You display order on search_val.
-- Rebuilt from the sources without the fragments; the builder compares this
-- scalar to the harness metric within tol. Gains are 2^click - 1 = click.
WITH sessions AS (
  SELECT xid, session_id, MIN(event_created) AS query_ts
  FROM {impressions}
  WHERE pmod(xxhash64(xid), 4) = 0
    AND session_id IS NOT NULL
  GROUP BY xid, session_id
  HAVING MIN(event_created) >= TIMESTAMP '{start_ts}'
     AND MIN(event_created) < TIMESTAMP '{end_ts}'
),
shown AS (
  SELECT
    s.xid,
    s.session_id,
    s.query_ts,
    i.entity_id,
    MIN(i.event_created) AS first_shown_ts,
    MIN(COALESCE(try_cast(element_at(split(i.item_index, ','), 1) AS INT), 9999)) AS served_slot,
    MAX(CASE WHEN i.impression_clicked THEN 1 ELSE 0 END) AS clicked
  FROM sessions AS s
  INNER JOIN {impressions} AS i
    ON i.xid = s.xid
   AND i.session_id = s.session_id
   AND i.event_created >= s.query_ts
   AND i.event_created < s.query_ts + INTERVAL 30 MINUTES
  GROUP BY s.xid, s.session_id, s.query_ts, i.entity_id
),
read_any AS (
  SELECT
    sh.xid,
    sh.session_id,
    sh.entity_id,
    MAX(CASE WHEN ix.entity_id IS NULL THEN 0 ELSE 1 END) AS was_read
  FROM shown AS sh
  LEFT JOIN {interactions} AS ix
    ON ix.xid = sh.xid
   AND ix.session_id = sh.session_id
   AND ix.entity_id = sh.entity_id
   AND ix.event_created >= sh.first_shown_ts
   AND ix.event_created < sh.query_ts + INTERVAL 2 HOURS
  GROUP BY sh.xid, sh.session_id, sh.entity_id
),
ordered AS (
  SELECT
    sh.xid,
    sh.session_id,
    sh.entity_id,
    CASE WHEN sh.clicked = 1 OR r.was_read = 1 THEN 1 ELSE 0 END AS click,
    ROW_NUMBER() OVER (
      PARTITION BY sh.xid, sh.session_id
      ORDER BY sh.first_shown_ts, sh.served_slot, sh.entity_id
    ) AS shown_order,
    COUNT(*) OVER (PARTITION BY sh.xid, sh.session_id) AS n_shown
  FROM shown AS sh
  INNER JOIN read_any AS r
    ON r.xid = sh.xid AND r.session_id = sh.session_id AND r.entity_id = sh.entity_id
),
kept AS (
  SELECT o.*
  FROM ordered AS o
  WHERE o.shown_order <= 50
    AND o.n_shown >= 2
    AND EXISTS (
      SELECT 1
      FROM ordered AS c
      WHERE c.xid = o.xid
        AND c.session_id = o.session_id
        AND c.shown_order <= 50
        AND c.click = 1
    )
),
ideal AS (
  SELECT
    xid,
    session_id,
    click,
    ROW_NUMBER() OVER (
      PARTITION BY xid, session_id
      ORDER BY click DESC, entity_id
    ) AS rn
  FROM kept
),
dcg AS (
  SELECT
    xid,
    session_id,
    SUM(CASE WHEN shown_order <= 10 THEN click / LOG(2, shown_order + 1) ELSE 0 END) AS dcg
  FROM kept
  GROUP BY xid, session_id
),
idcg AS (
  SELECT
    xid,
    session_id,
    SUM(CASE WHEN rn <= 10 THEN click / LOG(2, rn + 1) ELSE 0 END) AS idcg
  FROM ideal
  GROUP BY xid, session_id
)
SELECT AVG(d.dcg / i.idcg) AS metric
FROM dcg AS d
INNER JOIN idcg AS i
  ON i.xid = d.xid AND i.session_id = d.session_id
WHERE i.idcg > 0
