-- For You impressions of the item strictly before the request, looking back 7 days.
-- popularity_window_end is one second before query_ts.
WITH served AS (
  SELECT request_id, entity_id, request_ts
  FROM (
    SELECT
      c.request_id AS request_id,
      c.entity_id AS entity_id,
      c.request_ts AS request_ts,
      ROW_NUMBER() OVER (
        PARTITION BY c.request_id, c.entity_id
        ORDER BY c.occurrence_seq
      ) AS rn
    FROM {candidates} AS c
    WHERE c.candidate_status = 'served'
      AND c.entity_type = '{entity_type}'
      AND c.event_date BETWEEN DATE '{start}' AND DATE '{end}'
  ) AS ranked
  WHERE rn = 1
)
SELECT
  s.request_id AS query_id,
  s.entity_id AS item_id,
  s.request_ts - INTERVAL 1 SECOND AS popularity_window_end,
  CAST(COUNT(i.entity_id) AS DOUBLE) AS prior_impressions
FROM served AS s
LEFT JOIN {impressions} AS i
  ON i.entity_id = s.entity_id
 AND i.entity_type = '{entity_type}'
 AND i.page_type = 'for-you'
 AND i.latest_impression_rn = 1
 AND i.timestamp < s.request_ts
 AND i.timestamp >= s.request_ts - INTERVAL 7 DAYS
GROUP BY s.request_id, s.entity_id, s.request_ts
