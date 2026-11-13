-- How the article overlaps the user's profile: category rank, recent stories,
-- and shared people, organisations, locations, and IPTC categories.
SELECT
  request_id AS query_id,
  entity_id AS item_id,
  request_ts AS feature_ts,
  CASE WHEN f_cat_in_top_10 THEN 1.0 WHEN NOT f_cat_in_top_10 THEN 0.0 END
    AS category_in_top_10,
  CAST(f_cat_rank_in_top_10 AS DOUBLE) AS category_rank_in_top_10,
  CASE WHEN f_story_in_last_10 THEN 1.0 WHEN NOT f_story_in_last_10 THEN 0.0 END
    AS story_in_last_10,
  CASE WHEN f_story_in_last_30 THEN 1.0 WHEN NOT f_story_in_last_30 THEN 0.0 END
    AS story_in_last_30,
  CASE WHEN f_is_most_read_story THEN 1.0 WHEN NOT f_is_most_read_story THEN 0.0 END
    AS is_most_read_story,
  CAST(f_iptc_overlap_top_20 AS DOUBLE) AS iptc_overlap_top_20,
  CAST(f_people_overlap_top_20 AS DOUBLE) AS people_overlap_top_20,
  CAST(f_orgs_overlap_top_20 AS DOUBLE) AS orgs_overlap_top_20,
  CAST(f_locations_overlap_top_20 AS DOUBLE) AS locations_overlap_top_20,
  CAST(f_people_overlap_last_10 AS DOUBLE) AS people_overlap_last_10,
  CAST(f_orgs_overlap_last_10 AS DOUBLE) AS orgs_overlap_last_10,
  CAST(f_locations_overlap_last_10 AS DOUBLE) AS locations_overlap_last_10,
  CAST(f_iptc_overlap_last_10 AS DOUBLE) AS iptc_overlap_last_10,
  CAST(f_min_entity_rank_top_20 AS DOUBLE) AS min_entity_rank_top_20,
  CAST(f_entity_overlap_any AS DOUBLE) AS entity_overlap_any,
  CASE
    WHEN f_primary_city_in_locations_top_20 THEN 1.0
    WHEN NOT f_primary_city_in_locations_top_20 THEN 0.0
  END AS city_in_locations_top_20,
  CASE
    WHEN f_primary_state_in_locations_top_20 THEN 1.0
    WHEN NOT f_primary_state_in_locations_top_20 THEN 0.0
  END AS state_in_locations_top_20
FROM {gold}
WHERE request_ts >= TIMESTAMP '{start_ts}'
  AND request_ts < TIMESTAMP '{end_ts}'
