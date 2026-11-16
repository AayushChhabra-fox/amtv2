# Ready report: nfy_demo_v3

Status: **pass**

Dataset `nfy_article_rerank_gold`. Primary metric `ndcg@10_graded (impressed=0, bounce_or_click=1, engaged_read=2, deep_read=3)`.
Budget 60 DBU.
min_effect auto. Proposed 0.010000. Resolved 0.019578.
Noise pass. Ladder pass.

Viable ladder families: lgbm, neural, linear_reweight

Modality: 72 numeric, 6 categorical, embeddings [], 1866977 rows, 99702 queries.

## Unsupported families

- embedding: embedding ladder needs feat__profile__emb* and feat__item__emb*

amt sign ready locks this packet: the problem and task, the spec, the manifest, accepted repair diffs, the resolved min_effect, the budget, and the viable families.
