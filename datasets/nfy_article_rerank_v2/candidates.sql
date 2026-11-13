-- Articles the session was shown on For You within 30 minutes of query_ts: the
-- first 50 shown, ordered by first impression and then slot (the row in
-- item_index, which reads "row,column"). Sessions shown
-- fewer than two articles are dropped.
-- neg_shown_order is the production reference (the order For You showed them);
-- neg_item_age_hours is the recency reference, with unknown ages ranked last.
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
    MIN(COALESCE(try_cast(element_at(split(i.item_index, ','), 1) AS INT), 9999)) AS served_slot
  FROM sessions AS s
  INNER JOIN {impressions} AS i
    ON i.xid = s.xid
   AND i.session_id = s.session_id
   AND i.event_created >= s.query_ts
   AND i.event_created < s.query_ts + INTERVAL 30 MINUTES
  GROUP BY s.xid, s.session_id, s.query_ts, i.entity_id
),
ordered AS (
  SELECT
    *,
    ROW_NUMBER() OVER (
      PARTITION BY xid, session_id
      ORDER BY first_shown_ts, served_slot, entity_id
    ) AS shown_order,
    COUNT(*) OVER (PARTITION BY xid, session_id) AS n_shown
  FROM shown
)
SELECT
  concat(o.xid, ':', o.session_id) AS query_id,
  o.entity_id AS item_id,
  CAST(-o.shown_order AS DOUBLE) AS neg_shown_order,
  CASE
    WHEN f.publish_date IS NULL OR f.publish_date > o.query_ts THEN -1000000.0
    ELSE -(unix_timestamp(o.query_ts) - unix_timestamp(f.publish_date)) / 3600.0
  END AS neg_item_age_hours
FROM ordered AS o
LEFT JOIN (
  SELECT entity_id, MIN(publish_date) AS publish_date
  FROM {article_features}
  GROUP BY entity_id
) AS f
  ON f.entity_id = o.entity_id
WHERE o.shown_order <= 50
  AND o.n_shown >= 2
