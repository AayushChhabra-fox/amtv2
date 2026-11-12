-- Twelve production signals. Each map entry is a struct; the feature is its value.
-- Occurrence duplicates collapse to one row per request and item.
SELECT
  ranked.request_id AS query_id,
  ranked.entity_id AS item_id,
  ranked.request_ts AS query_ts,
  ranked.signals['breaking'].value AS signal_breaking,
  ranked.signals['category'].value AS signal_category,
  ranked.signals['continuity'].value AS signal_continuity,
  ranked.signals['cosine'].value AS signal_cosine,
  ranked.signals['entity_people'].value AS signal_entity_people,
  ranked.signals['entity_orgs'].value AS signal_entity_orgs,
  ranked.signals['entity_locations'].value AS signal_entity_locations,
  ranked.signals['iptc_category'].value AS signal_iptc_category,
  ranked.signals['local'].value AS signal_local,
  ranked.signals['recency'].value AS signal_recency,
  ranked.signals['velocity'].value AS signal_velocity,
  ranked.signals['visual_clickability'].value AS signal_visual_clickability,
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
