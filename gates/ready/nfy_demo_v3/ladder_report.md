# Ladder report: nfy_article_rerank_gold

Status: **pass**

Family set `lgbm, neural, linear_reweight`.
Kendall tau between R1 and R3: 0.8667 (minimum 0.6, n=6).

Not supported by this dataset, so not checked:

- embedding: embedding ladder needs feat__profile__emb* and feat__item__emb*

R1 train fraction 0.10, suggested next fraction 0.40.

| Family | Rung | Good | Bad | Holds |
| --- | --- | --- | --- | --- |
| lgbm | R1 | 0.5165 | 0.4684 | yes |
| lgbm | R2 | 0.5208 | 0.4610 | yes |
| lgbm | R3 | 0.5178 | 0.4548 | yes |
| neural | R1 | 0.4754 | 0.4494 | yes |
| neural | R2 | 0.4742 | 0.4409 | yes |
| neural | R3 | 0.4692 | 0.4500 | yes |
| linear_reweight | R1 | 0.4845 | 0.4593 | yes |
| linear_reweight | R2 | 0.4858 | 0.4684 | yes |
| linear_reweight | R3 | 0.4812 | 0.4560 | yes |

Gate G2 signs this report once per dataset and family set. Later runs on the same pair reuse the signature.
