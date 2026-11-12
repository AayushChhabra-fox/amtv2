-- Content attributes known at publish time. Engagement counters and the
-- 3072-d embedding are current snapshots, so they are not features.
SELECT
  f.entity_id AS item_id,
  f.publish_date AS published_ts,
  f.category_slug AS category_slug,
  f.primary_topic AS primary_topic,
  f.tone AS tone,
  f.geo_scope AS geo_scope,
  CASE
    WHEN f.evergreen IS NULL THEN NULL
    WHEN f.evergreen THEN 1.0
    ELSE 0.0
  END AS evergreen,
  CAST(f.sentiment AS DOUBLE) AS sentiment,
  CAST(f.word_count AS DOUBLE) AS word_count,
  a.title AS title
FROM {article_features} AS f
LEFT JOIN {articles} AS a
  ON a.id = f.entity_id
