# Noise report: nfy_demo_v3

Status: **pass**

Dataset `nfy_article_rerank_gold`. Primary metric `ndcg@10_graded (impressed=0, bounce_or_click=1, engaged_read=2, deep_read=3)`.
Budget 60 DBU, 5 DBU per candidate.
Brief min_effect auto (proposed 0.010). Resolved min_effect 0.019578.

Half-widths are the binding paired-delta half-width at each rung: the larger of the day-blocked bootstrap half-width and the three-seed t interval. min_effect must be at least 2× the R3 value.

R3 half-width 0.009789. Required min_effect 0.019578.

| Rung | Half-width |
| --- | --- |
| R0 | 0.046981 |
| R1 | 0.024474 |
| R2 | 0.013288 |
| R3 | 0.009789 |

LightGBM reference beats production at R3 (paired delta 0.033558, LightGBM 0.518310, production 0.484752).

## Remediation

- min_effect auto resolves to 0.019578, the larger of the proposed starting value 0.010 and 2× the R3 half-width.

The ready gate signs this report together with the brief, the dataset spec, the manifest, the evaluation definition, the resolved min_effect, and the budget.
