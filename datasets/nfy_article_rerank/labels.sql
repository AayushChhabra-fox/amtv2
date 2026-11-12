-- Graded label: 0 impressed, 1 click or bounce, 2 engaged_read, 3 deep_read.
-- read_quality counts only with a click outcome. The event must fall inside 24h.
-- label_event_ts is what the builder checks against label_window.
-- news_interactions records no surface (page_type is always 'dynamic'), so a read of the
-- item by the same xid within 24h of the For You impression is credited to it.
WITH impressed AS (
  SELECT
    c.request_id AS query_id,
    c.entity_id AS item_id,
    c.request_ts AS query_ts,
    c.xid AS xid,
    MAX(CASE WHEN i.impression_clicked THEN 1 ELSE 0 END) AS clicked,
    MIN(i.timestamp) AS impression_ts
  FROM {candidates} AS c
  INNER JOIN {impressions} AS i
    ON i.xid = c.xid
   AND i.entity_id = c.entity_id
   AND i.entity_type = '{entity_type}'
   AND i.page_type = 'for-you'
   AND i.latest_impression_rn = 1
   AND i.timestamp >= c.request_ts
   AND i.timestamp < c.request_ts + INTERVAL 30 MINUTES
  WHERE c.candidate_status = 'served'
    AND c.entity_type = '{entity_type}'
    AND c.event_date BETWEEN DATE '{start}' AND DATE '{end}'
    AND c.request_ts >= TIMESTAMP '{start_ts}'
    AND c.request_ts < TIMESTAMP '{end_ts}'
  GROUP BY c.request_id, c.entity_id, c.request_ts, c.xid
),
interacted AS (
  SELECT
    imp.query_id AS query_id,
    imp.item_id AS item_id,
    MAX(
      CASE ix.read_quality
        WHEN 'deep_read' THEN 3
        WHEN 'engaged_read' THEN 2
        WHEN 'bounce' THEN 1
        ELSE 0
      END
    ) AS interaction_grade,
    MIN(ix.event_created) AS interaction_ts
  FROM impressed AS imp
  LEFT JOIN {interactions} AS ix
    ON ix.xid = imp.xid
   AND ix.entity_id = imp.item_id
   AND ix.event_created >= imp.query_ts
   AND ix.event_created < imp.query_ts + INTERVAL 24 HOURS
   AND ix.read_quality IN ('bounce', 'engaged_read', 'deep_read')
  GROUP BY imp.query_id, imp.item_id
)
SELECT
  imp.query_id AS query_id,
  imp.item_id AS item_id,
  CASE
    WHEN inter.interaction_grade >= 1 THEN inter.interaction_grade
    WHEN imp.clicked = 1 THEN 1
    ELSE 0
  END AS label,
  CASE
    WHEN inter.interaction_grade >= 1 OR imp.clicked = 1 THEN 1
    ELSE 0
  END AS click,
  CASE
    WHEN inter.interaction_grade >= 1 THEN inter.interaction_ts
    ELSE imp.impression_ts
  END AS label_event_ts
FROM impressed AS imp
INNER JOIN interacted AS inter
  ON inter.query_id = imp.query_id
 AND inter.item_id = imp.item_id
