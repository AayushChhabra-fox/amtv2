-- Graded label: 0 shown and neither clicked nor read, 1 click or bounce,
-- 2 engaged_read, 3 deep_read. A read counts when news_interactions has the
-- same xid, session_id and article at or after it was first shown on For You,
-- and before query_ts + 2h (the label window).
-- label_event_ts is the earliest such read, or the first impression for the rest.
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
    MAX(CASE WHEN i.impression_clicked THEN 1 ELSE 0 END) AS clicked
  FROM sessions AS s
  INNER JOIN {impressions} AS i
    ON i.xid = s.xid
   AND i.session_id = s.session_id
   AND i.event_created >= s.query_ts
   AND i.event_created < s.query_ts + INTERVAL 30 MINUTES
  GROUP BY s.xid, s.session_id, s.query_ts, i.entity_id
),
graded AS (
  SELECT
    sh.xid,
    sh.session_id,
    sh.entity_id,
    sh.first_shown_ts,
    sh.clicked,
    MAX(
      CASE ix.read_quality
        WHEN 'deep_read' THEN 3
        WHEN 'engaged_read' THEN 2
        WHEN 'bounce' THEN 1
        ELSE 0
      END
    ) AS read_grade,
    MIN(ix.event_created) AS read_ts
  FROM shown AS sh
  LEFT JOIN {interactions} AS ix
    ON ix.xid = sh.xid
   AND ix.session_id = sh.session_id
   AND ix.entity_id = sh.entity_id
   AND ix.event_created >= sh.first_shown_ts
   AND ix.event_created < sh.query_ts + INTERVAL 2 HOURS
  GROUP BY sh.xid, sh.session_id, sh.entity_id, sh.first_shown_ts, sh.clicked
)
SELECT
  concat(xid, ':', session_id) AS query_id,
  entity_id AS item_id,
  CASE
    WHEN read_grade >= 1 THEN read_grade
    WHEN clicked = 1 THEN 1
    ELSE 0
  END AS label,
  CASE WHEN read_grade >= 1 OR clicked = 1 THEN 1 ELSE 0 END AS click,
  CASE WHEN read_grade >= 1 THEN read_ts ELSE first_shown_ts END AS label_event_ts
FROM graded
