-- The per-signal values the production scorer weighs into its score.
-- The linear-reweight ladder fits its own weights on exactly these columns.
SELECT
  request_id AS query_id,
  entity_id AS item_id,
  request_ts AS feature_ts,
  sig_cosine AS cosine,
  sig_category AS category,
  sig_iptc_category AS iptc_category,
  sig_entity_people AS entity_people,
  sig_entity_orgs AS entity_orgs,
  sig_entity_locations AS entity_locations,
  sig_recency AS recency,
  sig_velocity AS velocity,
  sig_breaking AS breaking,
  sig_continuity AS continuity,
  sig_local AS local,
  sig_visual_clickability AS visual_clickability
FROM {gold}
WHERE request_ts >= TIMESTAMP '{start_ts}'
  AND request_ts < TIMESTAMP '{end_ts}'
