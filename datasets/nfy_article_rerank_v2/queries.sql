-- One For You session is one query. query_ts is the session's first For You
-- article impression; sessions that began before the window are left out.
-- 1 in 4 users is kept by a hash of xid.
SELECT
  concat(xid, ':', session_id) AS query_id,
  MIN(event_created) AS query_ts
FROM {impressions}
WHERE pmod(xxhash64(xid), 4) = 0
  AND session_id IS NOT NULL
GROUP BY xid, session_id
HAVING MIN(event_created) >= TIMESTAMP '{start_ts}'
   AND MIN(event_created) < TIMESTAMP '{end_ts}'
