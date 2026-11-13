# Noise report: nikhil_neural_v1

Status: **pass**

Dataset `nfy_article_rerank_gold`. Primary metric `ndcg@10_graded (impressed=0, bounce_or_click=1, engaged_read=2, deep_read=3)`.
Budget 40 DBU, 5 DBU per candidate.
Brief min_effect 0.020000. Proposed starting value 0.010.

Half-widths are the binding paired-delta half-width at each rung: the larger of the day-blocked bootstrap half-width and the three-seed t interval. min_effect must be at least 2× the R3 value.

R3 half-width 0.009789. Required min_effect 0.019578.

| Rung | Half-width |
| --- | --- |
| R0 | 0.046981 |
| R1 | 0.024474 |
| R2 | 0.013288 |
| R3 | 0.009789 |

LightGBM reference beats production at R3 (paired delta 0.033558, LightGBM 0.518310, production 0.484752).

Gate G1 signs this report together with the brief, the dataset spec, the manifest, the evaluation definition, min_effect, and the budget.
