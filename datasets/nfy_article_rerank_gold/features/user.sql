-- The user's profile as the serving request saw it: history sizes, how stale
-- the profile was, and whether the user was cold-start.
SELECT
  request_id AS query_id,
  MIN(request_ts) AS feature_ts,
  MAX(CASE WHEN cold_start THEN 1.0 ELSE 0.0 END) AS cold_start,
  MAX(profile_staleness_seconds) / 3600.0 AS profile_staleness_hours,
  CAST(MAX(n_story_ids_last_10) AS DOUBLE) AS stories_last_10,
  CAST(MAX(n_story_ids_last_30) AS DOUBLE) AS stories_last_30,
  CAST(MAX(n_articles_category_slugs_top_10) AS DOUBLE) AS top_categories,
  CAST(MAX(n_recent_category_slugs) AS DOUBLE) AS recent_categories,
  CAST(MAX(n_people_top_20) AS DOUBLE) AS people_top_20,
  CAST(MAX(n_organisations_top_20) AS DOUBLE) AS orgs_top_20,
  CAST(MAX(n_locations_top_20) AS DOUBLE) AS locations_top_20,
  CAST(MAX(n_iptc_categories_top_20) AS DOUBLE) AS iptc_top_20,
  CAST(MAX(n_people_last_10) AS DOUBLE) AS people_last_10,
  CAST(MAX(n_organisations_last_10) AS DOUBLE) AS orgs_last_10,
  CAST(MAX(n_locations_last_10) AS DOUBLE) AS locations_last_10,
  CAST(MAX(n_iptc_categories_last_10) AS DOUBLE) AS iptc_last_10
FROM {gold}
WHERE request_ts >= TIMESTAMP '{start_ts}'
  AND request_ts < TIMESTAMP '{end_ts}'
GROUP BY request_id
