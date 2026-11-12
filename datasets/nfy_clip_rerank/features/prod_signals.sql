-- Twelve production signals plus the scores the reference columns read.
-- Occurrence duplicates collapse to one row per request and item.
SELECT
  ranked.request_id AS query_id,
  ranked.entity_id AS item_id,
  ranked.request_ts AS query_ts,
  CAST(ranked.signals['breaking'] AS DOUBLE) AS signal_breaking,
  CAST(ranked.signals['category'] AS DOUBLE) AS signal_category,
  CAST(ranked.signals['continuity'] AS DOUBLE) AS signal_continuity,
  CAST(ranked.signals['cosine'] AS DOUBLE) AS signal_cosine,
  CAST(ranked.signals['entity_people'] AS DOUBLE) AS signal_entity_people,
  CAST(ranked.signals['entity_orgs'] AS DOUBLE) AS signal_entity_orgs,
  CAST(ranked.signals['entity_locations'] AS DOUBLE) AS signal_entity_locations,
  CAST(ranked.signals['iptc_category'] AS DOUBLE) AS signal_iptc_category,
  CAST(ranked.signals['local'] AS DOUBLE) AS signal_local,
  CAST(ranked.signals['recency'] AS DOUBLE) AS signal_recency,
  CAST(ranked.signals['velocity'] AS DOUBLE) AS signal_velocity,
  CAST(ranked.signals['visual_clickability'] AS DOUBLE) AS signal_visual_clickability,
  ranked.total_score AS total_score,
  ranked.trending_score AS trending_score,
  CAST(ranked.rank AS DOUBLE) AS prod_rank,
  CAST(ranked.n_buckets AS DOUBLE) AS n_buckets,
  CAST(ranked.item_age_seconds AS DOUBLE) AS item_age_seconds,
  CASE WHEN ranked.is_breaking THEN 1.0 ELSE 0.0 END AS is_breaking
FROM (
  SELECT
    c.*,
    ROW_NUMBER() OVER (
      PARTITION BY c.request_id, c.entity_id
      ORDER BY c.occurrence_seq
    ) AS rn
  FROM {candidates} AS c
  WHERE c.candidate_status = 'served'
    AND c.entity_type = '{entity_type}'
    AND c.event_date BETWEEN DATE '{start}' AND DATE '{end}'
) AS ranked
WHERE ranked.rn = 1
