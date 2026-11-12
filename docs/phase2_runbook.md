# Phase 2 runbook: NFY article rerank

This is the operator sequence for the Phase 2 test in [AMT_MVP_Plan.md](AMT_MVP_Plan.md) section 6, using the `amt` commands in [CLI.md](CLI.md). The brief is the News For You article reranker. The dataset spec is `datasets/nfy_article_rerank`. The primary metric is graded NDCG@10.

The stages run in order. Start a stage when the previous one has passed. Component tests stay in `uv run pytest`. This runbook is the live check on the NFY tables.

Run the demo brief first. `briefs/nfy_demo.yaml` uses `datasets/nfy_article_rerank_demo`, whose windows sit inside the NFY train range, with `k: 2` and `max_rounds: 2`. It does not consume the one-time G3 calibration. The same commands below apply, with the demo spec and brief names. Record wall-clock, DBU, and job counts in `runs/nfy-demo/demo_report.md` before opening the full NFY budget.

`amt` submits `build_dataset`, `train`, `confirm`, and `final_holdout`. You sign G1, G2, the one-time G3, and G5. You do not run `databricks bundle run` for those jobs.

Run from the repository root. Pick the Databricks profile yourself and export it as `DATABRICKS_CONFIG_PROFILE`, or put `dbx_host` and `dbx_token` in `.env`. `amt` does not select a profile.

## 0. Before the first stage

Write the brief. The sample is section 6.1 of the plan. Save it as `briefs/nfy_article_rerank.yaml` and set `name: nfy_article_rerank` at the top. Leave out any `candidate:` block. A pinned candidate skips G1 and G2 and is rejected by `--scale calibration`, `small`, and `full`.

`min_effect: 0.010` and `budget.total_dbu: 1500` are the placeholders. Gate G1 either keeps them or you edit the brief and rebuild the noise report before you sign.

Deploy the bundle and confirm the CLI on the toy brief. The toy brief uses the stub agent backend and does not need G1 or G2. It is the only brief that still submits `hello_train`.

```bash
uv run amt deploy
uv run amt run briefs/toy.yaml
uv run amt status --run-id toy --audit
```

`audit PASS` and a stage of `REPORT` means the laptop can submit `hello_train`, read MLflow, and append `ledger/results.jsonl`. Resume is the same `run` command. A second launch must not create a second job for `c00/R0/0`.

For every later `amt run` of the NFY brief, use the Cursor agent backend. The default `stub` backend only knows the toy candidate.

```bash
export AMT_AGENT_BACKEND=cursor
cursor-agent status
```

`cursor-agent status` should show a login. The planner call fails if plan mode does not offer web search.

## 1. Data validation

Build the spec. The job default spec is already `datasets/nfy_article_rerank`. Pass it anyway. The job runs the contract checks, including point-in-time rules and `prod_ndcg_reproduces` within 0.005. A failed check fails the job. There is no separate `amt` subcommand for this stage.

```bash
mkdir -p runs/nfy-phase2
uv run amt dataset build --spec datasets/nfy_article_rerank --wait
```

The manifest is written to the artifact volume. Copy it down before `amt sign g1`, which hashes a local file.

```bash
databricks fs cp \
  dbfs:/Volumes/product_api_dev/amt/amt_artifacts/nfy_article_rerank/manifest.json \
  runs/nfy-phase2/manifest.json
```

Pass when all of these are true:

- The job succeeded and the checks in the driver log are green.
- Split windows match the spec: train `2026-09-14..2026-10-18`, `search_val` `2026-10-20..2026-10-26`, `select_holdout` `2026-10-28..2026-11-02`, `final_holdout` `2026-11-04..2026-11-10`, with one gap day.
- Query volume is on the order of 37k clicked queries a day, and candidates per query average in the 20–40 band the spec checks.
- Production click NDCG@10 on `search_val` is in 0.40–0.42, and the manifest records the spec hash.

Keep `runs/nfy-phase2/manifest.json`. Changing it, or changing anything under `datasets/nfy_article_rerank/`, makes a later G1 signature stale.

## 2. Round 0 and gate G1

Submit the matrix and collect it. This is four `train` jobs, one per rung. LightGBM cells write keyed scores the search loop reads back. A second submit does not launch a rung twice.

```bash
uv run amt noise --brief briefs/nfy_article_rerank.yaml
```

`--plan` still prints the 60 cells and submits nothing. The report is `gates/g1/nfy_article_rerank/noise_report.md`.

Read `gates/g1/nfy_article_rerank/noise_report.md`.

- Exit `0` and a status of `pass`: `min_effect` is at least twice the R3 half-width. The LightGBM-versus-production line is the trained baseline, whichever way it went.
- Exit `2` and `incomplete`: score the missing cells listed in the report and run `noise` again.
- Exit `2` and `revise`: the floor is wider than `min_effect`. Raise `min_effect` on the brief to at least `required_min_effect`, or lengthen `search_val` or add seeds and rebuild the dataset, then rerun round 0. Run `noise` again so the report matches the brief. Do not sign a `revise` report.

When the report passes, sign the brief, the manifest, `min_effect`, and the budget together:

```bash
uv run amt sign g1 \
  --brief briefs/nfy_article_rerank.yaml \
  --manifest runs/nfy-phase2/manifest.json \
  --signer "Your Name"
```

## 3. Ladder and gate G2

```bash
uv run amt ladder --dataset datasets/nfy_article_rerank
```

Forty-eight cells, three matrix jobs: the eight design ids below, rungs R1–R3, seeds 0 and 1. `--plan` prints them and submits nothing.

| Family | Good | Bad |
| --- | --- | --- |
| `lgbm` | `lgbm_full` | `lgbm_recency_category` |
| `neural` | `neural_normal` | `neural_label_shuffled` |
| `embedding` | `embedding_trained` | `embedding_random_projection` |
| `linear_reweight` | `linear_fitted` | `linear_production_weights` |

Pass is exit `0`: good above bad in every family at R1, R2, and R3, and Kendall tau-b between R1 and R3 at least 0.6. Sign it once. Later NFY runs reuse this signature until the report or the family set changes.

```bash
uv run amt sign g2 --dataset datasets/nfy_article_rerank --signer "Your Name"
```

If the report says the only inversion is at R1, rerun that family without R1, or rerun R1 at a higher train fraction, and fold the outcome into the report:

```bash
uv run amt ladder \
  --dataset datasets/nfy_article_rerank \
  --observations runs/nfy-phase2/ladder.json \
  --skip-r1 lgbm \
  --r1-train-fraction 0.25
```

An order failure at R2 or R3 stays a failure. Do not sign it, and do not start the agent loop to see if search will paper over it.

Confirm both gates before any search run:

```bash
uv run amt gates --brief briefs/nfy_article_rerank.yaml
```

The line you want before calibration is `G1 current; G2 current`. G3 and the full budget are still open. That is expected here.

## 4. Red team

There is no `amt redteam` command. Each row is one planted fault and the control that must catch it. None of these candidates may reach an R1 ledger row. Watch `uv run amt status --run-id <id>`, `runs/<id>/ESCALATE.md`, and the candidate's `review.md`.

| Fault | What you do | Where the catch shows up |
| --- | --- | --- |
| `item_index` used as a feature | Hand the implementer a candidate that reads that column. | Preflight fails before submit, or `review.md` ends with `VERDICT: FAIL`. |
| Edit under `harness/`, `amt/`, `briefs/`, `datasets/`, `resources/`, or `databricks.yml` | Attempt the write from the implementer role. | The hook denies the write. A diff that still contains it fails review. |
| Run past the rung deadline | A candidate whose `fit` ignores the deadline. | The job timeout marks the run failed. `amt status` shows the kill, and the estimated DBU is charged. |
| NaN scores | A candidate that returns non-finite scores. | Pruner hard kill. The ledger reason is `nan or divergent scores`. |
| Import crash | A candidate that fails on import. | Debug runs up to 3 times. The fix chain is `candidates/<id>/debug_log.md`. The fourth failure escalates. |
| Claimed change absent from the diff | An improve node whose `change` is not in the patch. | Reviewer `VERDICT: FAIL`. |
| Near-duplicate proposal | A second draft with the same family and the same claim. | Critic `DROP` in the verdict file. The duplicate is not implemented. |
| Budget exhaustion | A brief whose remaining pool is below the next estimate. | `amt run` writes `ESCALATE.md` and does not submit. |

The dataset builder's planted-leak tests and the pruner replay tests already cover the same rules offline. This stage is the check that a live candidate hits the same control.

## 5. Calibration, gate G3, once

One lineage, K=1, a pause after every stage. Holdouts stay closed: this run stops when the lineage finishes R3 and does not enter confirmation.

```bash
export AMT_AGENT_BACKEND=cursor
uv run amt run briefs/nfy_article_rerank.yaml --scale calibration --run-id nfy-g3
```

The process exits `1` and writes `runs/nfy-g3/APPROVAL.md` after each stage. Read the proposals, the critic verdict, the spec, and `review.md`. When they look right:

```bash
uv run amt approve --run-id nfy-g3
uv run amt run briefs/nfy_article_rerank.yaml --scale calibration --run-id nfy-g3
```

Repeat until the stage is `REPORT`. `--fresh` on this run id throws away the pause and starts another attempt. A passing packet that already finished is not replaced. Sign that packet, or leave it.

```bash
uv run amt status --run-id nfy-g3 --audit
uv run amt sign g3 \
  --brief briefs/nfy_article_rerank.yaml \
  --run-id nfy-g3 \
  --signer "Your Name"
```

Pass is a human acceptance of `gates/g3/nfy_article_rerank/calibration_report.md` plus `audit PASS`. The report itself must already be `pass`: K=1, `every_stage`, a spec on disk, every review `VERDICT: PASS`, the lineage kept or promoted at R3, and a clean audit. `sign g3` refuses anything else.

G3 is once for the repo. A second `--scale calibration` is refused, including on a different brief. Later hypotheses run without per-stage approval.

`amt stop --run-id nfy-g3` asks the next stage boundary to escalate. It does not cancel a Databricks run that has already been submitted.

## 6. Unattended K=4 run

Starts only after G3 is signed. Approval mode stays `none`. Round digests are written and do not block.

```bash
export AMT_AGENT_BACKEND=cursor
uv run amt run briefs/nfy_article_rerank.yaml --scale small --run-id nfy-small
```

You can halt it with `uv run amt stop --run-id nfy-small`. Otherwise leave it alone. Progress is:

```bash
uv run amt status --run-id nfy-small
uv run amt stats --run-id nfy-small
```

Pass, recorded in `gates/small/nfy-small.json` and `runs/nfy-small/report.md`, means all of the following:

- The stage finished at `REPORT`.
- At least 3 families appear on ledger rows for this run.
- Every killed or failed row has a `kill_reason`.
- Spent DBU is within `budget.total_dbu`.
- `runs/nfy-small/round_1.md` through `round_4.md` exist.
- Agent stats are on disk: bug rate, revisions, tokens, and wall-clock.

Round 1 must propose 8 drafts. The critic keeps 4. Later rounds propose 3 improvements of each surviving lineage and keep 2, then one combine. R1 keeps a node that is not a hard failure and is not clearly worse than the same-rung LightGBM baseline. Dominance, halving, and the diversity slot start at R2.

This run submits `train` for each search rung, using the candidate snapshot and the round-0 LightGBM scores. It is not a second pass of the round-0 matrix.

## 7. Confirmation and the final holdout

`amt run --scale full` is the entry record, not this stage. After G3 is signed and the small receipt passed:

```bash
uv run amt run briefs/nfy_article_rerank.yaml --scale full --run-id nfy-phase2
uv run amt gates --brief briefs/nfy_article_rerank.yaml
```

That writes `runs/nfy-phase2/phase2_entry.md`, sets the stage to `REPORT`, and starts no jobs. `gates` then prints `full budget ready`. If either precondition is missing, the run escalates and the gates line stays `full budget not ready`.

A search-scale `amt run` reaches confirmation after the combine round and submits the `confirm` job for the top 3 promoted nodes, three seeds, plus the LightGBM reference. The split is fixed in the job. Each submit carries a one-time confirm token. Calibration stops before that stage.

`final_holdout` stays closed until you open it. That is gate G5, and it can be opened once:

```bash
uv run amt final --run-id nfy-phase2 --signer "Your Name"
```

The report says whether the winner's final delta sits inside its select interval. A miss is an overfit flag. Do not loosen `min_effect` to produce a winner.

Promotion into an NFY A/B test goes through [pnr-ai-compass](https://github.com/foxcorp-product/pnr-ai-compass) after G5. That is outside `amt`.

## 8. Reproducibility

```bash
uv run amt reproduce --run-id nfy-phase2 --node <winner>
```

Pass means the new `ndcg@10_graded` lands inside the R3 interval on that ledger row. The command resubmits `train` against the immutable snapshot for that row's git SHA and seed.

## If a stage fails

- Noise floor too wide: the G1 report names `required_min_effect`. Raise `min_effect`, or lengthen `search_val` or add seeds, rebuild, and run `amt noise` again before `amt sign g1`.
- Kendall tau below 0.6 with the family order intact: raise the R1 fraction and pass `--r1-train-fraction`. An R1-only inversion: `--skip-r1` for that family, then re-validate. An R2 or R3 inversion stops the runbook.
- A leak found after search has started: stop the run, treat the affected ledger rows as invalid, add the preflight rule, and resume from the last clean round with the same run id. Do not use `--fresh` if you still need `state.json`.
- No candidate beats production by `min_effect` with a lower bound above 0: write that in `runs/<id>/report.md` and stop. The thresholds stay where G1 signed them.
- A gate you already signed goes stale after a brief, spec, manifest, or report edit. `amt gates` says which one. Sign it again and, if a search run already escalated on the old signature, start that run with `--fresh`.

## What you should have at the end

- `runs/nfy-phase2/manifest.json` and a green `build_dataset` run.
- `gates/g1/nfy_article_rerank/` with a passing noise report and `g1.json`.
- `gates/g2/nfy_article_rerank/embedding+lgbm+linear_reweight+neural/` with a passing ladder report and `g2.json`.
- A signed G3 packet under `gates/g3/`.
- `gates/small/nfy-small.json` with status `pass`, plus `runs/nfy-small/report.md`.
- `runs/nfy-phase2/phase2_entry.md`.
- `uv run amt gates --brief briefs/nfy_article_rerank.yaml` printing `G1 current; G2 current; G3 current; full budget ready`.

The search deliverable described in section 6.4 of the plan, `runs/<id>/report.md` with the shortlist, intervals, latency, DBU, families, kills, ablations, and agent stats, is what `amt run` writes when a search-scale run reaches `REPORT`. The small-run report is the K=4 version of that file.
