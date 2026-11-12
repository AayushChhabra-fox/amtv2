-- One request is one query. entity_type comes from the spec (article or clip).
SELECT
  request_id AS query_id,
  request_ts AS query_ts
FROM {requests}
WHERE entity_type = '{entity_type}'
  AND event_date BETWEEN DATE '{start}' AND DATE '{end}'
