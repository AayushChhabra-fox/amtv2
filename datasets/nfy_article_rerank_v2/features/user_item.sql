-- The user's history with this article and its category in the 24 hours
-- before the start of the query's hour: reads of the article, For You
-- impressions of the article, and reads in the article's category.
-- Counts are per user, key and hour first, so each pair joins at most 24 rows.
-- user_item_window_end is one second before the query's hour.
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
categories AS (
  SELECT entity_id, MIN(category_slug) AS category_slug
  FROM {article_features}
  GROUP BY entity_id
),
item_reads AS (
  SELECT xid, entity_id, date_trunc('HOUR', event_created) AS hour_ts, COUNT(*) AS n
  FROM {interactions}
  WHERE pmod(xxhash64(xid), 4) = 0
  GROUP BY xid, entity_id, date_trunc('HOUR', event_created)
),
item_shown AS (
  SELECT xid, entity_id, date_trunc('HOUR', event_created) AS hour_ts, COUNT(*) AS n
  FROM {impressions}
  WHERE pmod(xxhash64(xid), 4) = 0
  GROUP BY xid, entity_id, date_trunc('HOUR', event_created)
),
category_reads AS (
  SELECT xid, category_slug, date_trunc('HOUR', event_created) AS hour_ts, COUNT(*) AS n
  FROM {interactions}
  WHERE pmod(xxhash64(xid), 4) = 0
    AND category_slug IS NOT NULL
  GROUP BY xid, category_slug, date_trunc('HOUR', event_created)
),
read_counts AS (
  SELECT p.xid, p.session_id, p.entity_id, SUM(r.n) AS n
  FROM pairs AS p
  INNER JOIN item_reads AS r
    ON r.xid = p.xid
   AND r.entity_id = p.entity_id
   AND r.hour_ts >= date_trunc('HOUR', p.query_ts) - INTERVAL 24 HOURS
   AND r.hour_ts < date_trunc('HOUR', p.query_ts)
  GROUP BY p.xid, p.session_id, p.entity_id
),
shown_counts AS (
  SELECT p.xid, p.session_id, p.entity_id, SUM(h.n) AS n
  FROM pairs AS p
  INNER JOIN item_shown AS h
    ON h.xid = p.xid
   AND h.entity_id = p.entity_id
   AND h.hour_ts >= date_trunc('HOUR', p.query_ts) - INTERVAL 24 HOURS
   AND h.hour_ts < date_trunc('HOUR', p.query_ts)
  GROUP BY p.xid, p.session_id, p.entity_id
),
category_counts AS (
  SELECT p.xid, p.session_id, p.entity_id, SUM(c.n) AS n
  FROM pairs AS p
  INNER JOIN categories AS cat
    ON cat.entity_id = p.entity_id
  INNER JOIN category_reads AS c
    ON c.xid = p.xid
   AND c.category_slug = cat.category_slug
   AND c.hour_ts >= date_trunc('HOUR', p.query_ts) - INTERVAL 24 HOURS
   AND c.hour_ts < date_trunc('HOUR', p.query_ts)
  GROUP BY p.xid, p.session_id, p.entity_id
)
SELECT
  concat(p.xid, ':', p.session_id) AS query_id,
  p.entity_id AS item_id,
  date_trunc('HOUR', p.query_ts) - INTERVAL 1 SECOND AS user_item_window_end,
  CAST(COALESCE(r.n, 0) AS DOUBLE) AS user_item_reads_24h,
  CAST(COALESCE(h.n, 0) AS DOUBLE) AS user_item_shown_24h,
  CAST(COALESCE(c.n, 0) AS DOUBLE) AS user_category_reads_24h
FROM pairs AS p
LEFT JOIN read_counts AS r
  ON r.xid = p.xid AND r.session_id = p.session_id AND r.entity_id = p.entity_id
LEFT JOIN shown_counts AS h
  ON h.xid = p.xid AND h.session_id = p.session_id AND h.entity_id = p.entity_id
LEFT JOIN category_counts AS c
  ON c.xid = p.xid AND c.session_id = p.session_id AND c.entity_id = p.entity_id
