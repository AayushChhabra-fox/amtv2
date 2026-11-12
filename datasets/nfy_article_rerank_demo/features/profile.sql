-- Profile version the request named, joined on user_profile_key + profile_content_hash.
-- first_seen is when that version became available. Arrays stay out of the frame.
SELECT
  r.request_id AS query_id,
  p.first_seen AS profile_version_ts,
  CASE
    WHEN p.user_profile_key IS NULL THEN NULL
    WHEN p.cold_start THEN 1.0
    ELSE 0.0
  END AS cold_start,
  CAST(p.n_story_ids_last_10 AS DOUBLE) AS n_story_ids_last_10,
  CAST(p.n_story_ids_last_30 AS DOUBLE) AS n_story_ids_last_30,
  CAST(p.n_recent_category_slugs AS DOUBLE) AS n_recent_category_slugs,
  CAST(p.n_articles_category_slugs_top_10 AS DOUBLE) AS n_articles_category_slugs_top_10,
  CAST(p.n_people_top_20 AS DOUBLE) AS n_people_top_20,
  CAST(p.n_organisations_top_20 AS DOUBLE) AS n_organisations_top_20,
  CAST(p.n_locations_top_20 AS DOUBLE) AS n_locations_top_20,
  CAST(p.n_iptc_categories_top_20 AS DOUBLE) AS n_iptc_categories_top_20,
  CAST(p.recent_embedding_dim AS DOUBLE) AS recent_embedding_dim,
  CAST(p.overall_embedding_dim AS DOUBLE) AS overall_embedding_dim
FROM {requests} AS r
LEFT JOIN {profiles} AS p
  ON p.user_profile_key = r.user_profile_key
 AND p.profile_content_hash = r.profile_content_hash
WHERE r.entity_type = '{entity_type}'
  AND r.event_date BETWEEN DATE '{start}' AND DATE '{end}'
