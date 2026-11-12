-- Content attributes known at publish time. Engagement counters and the
-- 3072-d embedding are current snapshots, so they are not features.
-- news_article_feature keeps a row per headline revision, so the title is the
-- first one recorded (earliest etl_created_date), not a later edit.
-- publish_date is overwritten when an article is republished. An article whose
-- publish_date is after its first serve in [start, end] has attributes from
-- after some requests, so it gets no row (null article features) instead.
WITH first_served AS (
  SELECT entity_id, MIN(request_ts) AS first_served_ts
  FROM {candidates}
  WHERE candidate_status = 'served'
    AND entity_type = '{entity_type}'
    AND event_date BETWEEN DATE '{start}' AND DATE '{end}'
  GROUP BY entity_id
),
first_title AS (
  SELECT id, title
  FROM (
    SELECT
      id,
      title,
      ROW_NUMBER() OVER (PARTITION BY id ORDER BY etl_created_date, title) AS rn
    FROM {articles}
  ) AS ranked
  WHERE rn = 1
)
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
LEFT JOIN first_title AS a
  ON a.id = f.entity_id
LEFT JOIN first_served AS s
  ON s.entity_id = f.entity_id
WHERE s.first_served_ts IS NULL
   OR f.publish_date <= s.first_served_ts
