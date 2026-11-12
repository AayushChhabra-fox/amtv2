-- Served candidates that were impressed on For You within 30 minutes.
-- latest_impression_rn = 1 drops duplicate impression rows.
SELECT DISTINCT
  c.request_id AS query_id,
  c.entity_id AS item_id
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
WHERE c.candidate_status = 'served'
  AND c.entity_type = '{entity_type}'
  AND c.event_date BETWEEN DATE '{start}' AND DATE '{end}'
