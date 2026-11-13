-- Content attributes known at publish time, per query and article, so an
-- article published (or republished: publish_date is overwritten) after the
-- session started gets no row and therefore null attributes.
-- Engagement counters and the 3072-d embedding are current snapshots, so they
-- are not features.
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
articles AS (
  SELECT *
  FROM (
    SELECT
      f.*,
      ROW_NUMBER() OVER (PARTITION BY f.entity_id ORDER BY f.publish_date) AS rn
    FROM {article_features} AS f
  ) AS ranked
  WHERE rn = 1
)
SELECT
  concat(p.xid, ':', p.session_id) AS query_id,
  p.entity_id AS item_id,
  a.publish_date AS published_ts,
  (unix_timestamp(p.query_ts) - unix_timestamp(a.publish_date)) / 3600.0 AS recency_hours,
  a.category_slug AS category_slug,
  a.primary_topic AS primary_topic,
  a.tone AS tone,
  a.geo_scope AS geo_scope,
  CASE
    WHEN a.evergreen IS NULL THEN NULL
    WHEN a.evergreen THEN 1.0
    ELSE 0.0
  END AS evergreen,
  CAST(a.sentiment AS DOUBLE) AS sentiment,
  CAST(a.word_count AS DOUBLE) AS word_count
FROM pairs AS p
INNER JOIN articles AS a
  ON a.entity_id = p.entity_id
WHERE a.publish_date <= p.query_ts
