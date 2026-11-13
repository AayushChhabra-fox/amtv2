-- Article content, age, community activity, and the previous day's popularity
-- and CTR. publish_date is overwritten on republish, so an article whose
-- publish_date or enrichment is after the request gets no row.
SELECT
  request_id AS query_id,
  entity_id AS item_id,
  GREATEST(publish_date, enrichment_version_ts) AS article_ts,
  item_age_seconds / 3600.0 AS age_hours,
  CAST(word_count AS DOUBLE) AS word_count,
  CAST(approx_read_time_min AS DOUBLE) AS read_time_min,
  CASE WHEN is_breaking THEN 1.0 WHEN NOT is_breaking THEN 0.0 END AS is_breaking,
  CASE WHEN evergreen THEN 1.0 WHEN NOT evergreen THEN 0.0 END AS evergreen,
  sentiment,
  category_slug,
  primary_topic,
  tone,
  bias,
  geo_scope,
  complexity,
  CASE WHEN is_backfill THEN 1.0 WHEN NOT is_backfill THEN 0.0 END AS is_backfill,
  CAST(n_buckets AS DOUBLE) AS n_retrieval_buckets,
  CAST(n_react_article_72h AS DOUBLE) AS react_article_72h,
  CAST(n_react_comment_72h AS DOUBLE) AS react_comment_72h,
  CAST(n_comment_72h AS DOUBLE) AS comment_72h,
  CAST(n_reply_72h AS DOUBLE) AS reply_72h,
  CAST(n_users_72h AS DOUBLE) AS users_72h,
  CAST(n_commenters_72h AS DOUBLE) AS commenters_72h,
  CAST(n_react_article_15m AS DOUBLE) AS react_article_15m,
  CAST(n_react_comment_15m AS DOUBLE) AS react_comment_15m,
  CAST(n_comment_15m AS DOUBLE) AS comment_15m,
  CAST(n_reply_15m AS DOUBLE) AS reply_15m,
  story_unique_users_72h,
  community_acceleration_pct,
  reactions_per_comment_pct,
  community_weighted_events_pct,
  community_score,
  item_popularity_pct AS popularity_pct_prev_day,
  item_prior_ctr AS ctr_prev_day
FROM {gold}
WHERE request_ts >= TIMESTAMP '{start_ts}'
  AND request_ts < TIMESTAMP '{end_ts}'
  AND GREATEST(publish_date, enrichment_version_ts) <= request_ts
