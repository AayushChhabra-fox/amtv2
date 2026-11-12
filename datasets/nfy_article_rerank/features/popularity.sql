-- For You impressions of the item in the 7 days before the start of the request's hour.
-- Impressions are counted per item and hour on a dense grid, so each served pair joins
-- one grid row instead of every impression of a popular item.
-- popularity_window_end is one second before the request's hour.
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
      AND c.request_ts >= TIMESTAMP '{start_ts}'
      AND c.request_ts < TIMESTAMP '{end_ts}'
  ) AS ranked
  WHERE rn = 1
),
items AS (
  SELECT DISTINCT entity_id FROM served
),
hours AS (
  SELECT explode(sequence(
    TIMESTAMP '{scan_start} 00:00:00',
    TIMESTAMP '{end} 23:00:00',
    INTERVAL 1 HOUR
  )) AS hour_ts
),
hourly AS (
  SELECT
    i.entity_id AS entity_id,
    date_trunc('HOUR', i.timestamp) AS hour_ts,
    COUNT(*) AS n
  FROM {impressions} AS i
  INNER JOIN items AS it ON it.entity_id = i.entity_id
  WHERE i.entity_type = '{entity_type}'
    AND i.page_type = 'for-you'
    AND i.latest_impression_rn = 1
  GROUP BY i.entity_id, date_trunc('HOUR', i.timestamp)
),
windowed AS (
  SELECT
    it.entity_id AS entity_id,
    h.hour_ts AS hour_ts,
    SUM(COALESCE(hr.n, 0)) OVER (
      PARTITION BY it.entity_id
      ORDER BY h.hour_ts
      ROWS BETWEEN 168 PRECEDING AND 1 PRECEDING
    ) AS prior_impressions
  FROM items AS it
  CROSS JOIN hours AS h
  LEFT JOIN hourly AS hr
    ON hr.entity_id = it.entity_id
   AND hr.hour_ts = h.hour_ts
)
SELECT
  s.request_id AS query_id,
  s.entity_id AS item_id,
  date_trunc('HOUR', s.request_ts) - INTERVAL 1 SECOND AS popularity_window_end,
  CAST(COALESCE(w.prior_impressions, 0) AS DOUBLE) AS prior_impressions
FROM served AS s
LEFT JOIN windowed AS w
  ON w.entity_id = s.entity_id
 AND w.hour_ts = date_trunc('HOUR', s.request_ts)
