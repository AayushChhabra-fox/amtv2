# Phase 2 runbook: NFY article rerank

This is the operator sequence for the Phase 2 test in section 6 of [AMT_MVP_Plan.md](AMT_MVP_Plan.md), using the `amt` commands in [CLI.md](CLI.md). The brief is the News For You article reranker, the dataset spec is `datasets/nfy_article_rerank`, and the primary metric is graded NDCG@10.

Sections 0 to 4 are the sequence that ran live on 2026-11-12/13 with the demo brief `briefs/nfy_demo.yaml` on `datasets/nfy_article_rerank_demo`. The numbers quoted are from that run, and its full record is `runs/nfy-demo/demo_report.md`. Sections 5 to 9 (red team, G3, the K=4 run, confirmation and the final holdout, reproducibility) have not been run live yet.

The split of work:

- **`amt` submits:** `build_dataset`, `train`, `confirm`, and `final_holdout`.
- **You sign:** G1, G2, the one-time G3, and G5.
- **You do not** run `databricks bundle run` for any of these jobs.

Run every command from the repository root. Each stage starts only after the previous one has passed.

## 0. Setup

### `.env`

Copy `.env.example` to `.env` and fill in these values:

- **Auth.** Use one of: `DATABRICKS_HOST` with `DATABRICKS_TOKEN`, an OAuth client, or `DATABRICKS_CONFIG_PROFILE`. The legacy keys `dbx_host` and `dbx_token` are still read. `amt` never picks a profile for you.
- **Token scopes.** A scoped personal access token needs `files` as well as `jobs`, `workspace`, `clusters`, `mlflow`, `sql`, and `unity-catalog`.
  - Matrices, candidate snapshots, and confirm tokens are written to the Volume.
  - Without `files`, any Volume write fails with `required scopes: files`. Scopes can be edited on an existing token, and the change takes about a minute to take effect.
- **`AMT_CLUSTER_ID`.** Set this to run every job on one all-purpose cluster.
  - The demo used `1112-131423-rwjrucy8`, i3.xlarge with 1–6 workers.
  - Fixed-cluster runs execute from the deployed bundle files and install the job YAML's PyPI libraries (LightGBM, MLflow).
- **Agents.** Set `AMT_AGENT_BACKEND=cursor` and `CURSOR_API_KEY`. The `stub` backend only knows the toy candidate.

Never commit `.env` or print its values.

### Deploy

```bash
uv run amt doctor
uv run amt deploy
```

When to deploy again:

- **After any change under `harness/`, `datasets/`, or `resources/`.** Submits are refused with `redeploy: harness changed since deploy` until you do.
- **Not for changes under `amt/`.** Those run on the laptop.
- **Expect reruns after each deploy.** Every deploy changes the tree hash. That hash is part of the round-0 and ladder idempotency tokens, so the next submit reruns those matrices.

Leave `spark.sql.ansi.enabled` at its default (on) on the cluster. The dataset content hash now sums as DECIMAL(38,0), so it no longer overflows with ANSI on.

### `databricks` CLI outside `amt`

A bare `databricks fs …` command does not read `.env`. Either export a profile you choose, or run the command through `amt`'s env loader:

```bash
uv run python -c "import subprocess, sys; from amt.envutil import load_env; load_env(); \
sys.exit(subprocess.call(sys.argv[1:]))" databricks fs ls dbfs:/Volumes/product_api_dev/amt/amt_artifacts
```

### The brief

Write `briefs/nfy_article_rerank.yaml` from the sample in section 6.1 of the plan, using `briefs/nfy_demo.yaml` as the template. Leave out any `candidate:` block. A pinned candidate skips G1 and G2, and it is refused by `--scale calibration`, `small`, and `full`.

`min_effect` and `budget.total_dbu` are set at G1 (section 2), not before.

The demo's search could only draw from two families: "learned reweighting of production signals" and "gradient-boosted rankers". In the demo's one-hour window, the popularity baseline was the strongest ranker (R3 graded NDCG@10 of 0.5805). Before spending the full budget, put popularity- and freshness-led models in `design_space.families`.

### The demo brief first

`briefs/nfy_demo.yaml` sets `k: 2`, `max_rounds: 2`, and a 40-DBU budget, on six hours of 2026-10-07:

| Split | Window (UTC) |
| --- | --- |
| train | 12:00–15:00 |
| search_val | 15:00–16:00 |
| select_holdout | 16:00–17:00 |
| final_holdout | 17:00–18:00 |

Split windows can be whole days or `YYYY-MM-DDTHH:MM..YYYY-MM-DDTHH:MM`, with the end exclusive. The demo does not consume the one-time G3 calibration. Its G1 and G2 signatures do not carry over: G1 is per brief, and G2 is per dataset.

## 1. Data validation

```bash
uv run amt dataset build --spec datasets/nfy_article_rerank --wait
```

What the build does:

- It scans the sources once over the full span and materializes each SQL fragment as `<dataset>_stage_<fragment>`.
- It runs every contract check on those copies, including the point-in-time rules and `prod_ndcg_reproduces` within 0.005.
- It writes the splits from one assembled table. A failed check fails the job and leaves the staging tables in place for querying. A successful build drops them.

The demo build (run `189792139594058`) took about 14 minutes for six hours of data. The full NFY spans five weeks, so expect a much longer build.

Article features are one row per article, using the earliest title. An article whose `publish_date` is later than its first serve in the build window gets null article features, because the publish date is overwritten when an article is republished.

Copy the manifest down. G1 hashes a local file:

```bash
mkdir -p runs/nfy-phase2
uv run python -c "import subprocess, sys; from amt.envutil import load_env; load_env(); \
sys.exit(subprocess.call(sys.argv[1:]))" databricks fs cp --overwrite \
  dbfs:/Volumes/product_api_dev/amt/amt_artifacts/nfy_article_rerank/manifest.json \
  runs/nfy-phase2/manifest.json
```

Pass when all of these are true:

- The job succeeded, and every check in the manifest's `checks` passed.
- The split windows match the spec:
  - train `2026-09-14..2026-10-18`
  - `search_val` `2026-10-20..2026-10-26`
  - `select_holdout` `2026-10-28..2026-11-02`
  - `final_holdout` `2026-11-04..2026-11-10`
  - one gap day between splits
- Candidates per query average in the spec's band. The demo saw 18.8–21.1.
- The harness's production NDCG equals the expected value. The demo got 0.538548 on both.
- The manifest's `spec_hash` equals `load_spec(...).spec_hash()` for the spec on disk.

Keep `runs/nfy-phase2/manifest.json`. Changing it, or anything under `datasets/nfy_article_rerank/`, makes a later G1 signature stale.

## 2. Round 0 and gate G1

```bash
uv run amt noise --brief briefs/nfy_article_rerank.yaml --submit
uv run amt noise --brief briefs/nfy_article_rerank.yaml --collect
```

How round 0 runs:

- **Jobs.** `--submit` launches four `train` matrix jobs, one per rung (R0–R3). Together they cover 5 baselines × 4 rungs × 3 seeds, which is 60 cells. On the demo, each job finished in minutes.
- **Collection.** `--collect` waits for the jobs, then reads MLflow across every experiment under `AMT_MLFLOW_EXPERIMENT_ROOT`.
  - It keeps the newest scored run per cell.
  - It writes `runs/round0-<brief>/round0.json` and `gates/g1/<brief>/noise_report.{json,md}`.
- **Bare `amt noise --brief …`** does both steps.
- **Resubmitting** skips rungs that succeeded and reruns failed ones.

Read `gates/g1/<brief>/noise_report.md`:

- **`pass` (exit 0).** `min_effect` is at least twice the R3 paired-delta half-width.
- **`incomplete` (exit 2).** Cells are missing. Check the failed jobs, fix them, and submit again.
- **`revise` (exit 2).** The noise floor is wider than `min_effect`. Either raise `min_effect` to at least `required_min_effect`, or lengthen `search_val` or add seeds and rebuild.
  - Raising `min_effect` needs no retraining. Run `--collect` again so the report matches the brief.
  - Never sign a `revise` report.

What the demo saw:

- **Half-widths:** R0 0.0427, R1 0.0191, R2 0.0140, R3 0.0127.
- **`min_effect`:** the report said `revise` at 0.01. Raising it to 0.026 (required: 0.0255) made it `pass`.
- **R3 baselines:**

| Baseline | NDCG@10 |
| --- | --- |
| popularity | 0.5805 |
| lgbm_reference | 0.5470 |
| production | 0.5217 |
| recency | 0.4860 |
| random | 0.4689 |

NFY's seven-day `search_val` has far more queries than the demo's one hour, so measure its floor rather than reusing 0.026. With only one day of data, the day-blocked bootstrap reduces to a query bootstrap, which makes the floor optimistic.

Sign the brief, the manifest, `min_effect`, and the budget together:

```bash
uv run amt sign g1 \
  --brief briefs/nfy_article_rerank.yaml \
  --manifest runs/nfy-phase2/manifest.json \
  --signer "Your Name"
```

Editing the brief, the spec, the manifest, or the report after signing makes G1 stale. Sign again after any of those edits.

## 3. Ladder and gate G2

```bash
uv run amt ladder --dataset nfy_article_rerank --submit
uv run amt ladder --dataset nfy_article_rerank --collect
```

`--dataset` accepts the spec name or `datasets/<name>`. The ladder is 48 cells in three matrix jobs: eight designs × R1–R3 × seeds 0 and 1. It took about 14 minutes on the demo. `--plan` prints the cells and submits nothing.

| Family | Good | Bad |
| --- | --- | --- |
| `lgbm` | `lgbm_full` | `lgbm_recency_category` |
| `neural` | `neural_normal` | `neural_label_shuffled` |
| `embedding` | `embedding_trained` | `embedding_random_projection` |
| `linear_reweight` | `linear_fitted` | `linear_label_shuffled` |

How unsupported families are handled:

- A design that needs columns the dataset lacks logs `status=unsupported` instead of failing.
- When every cell of a family is unsupported, the report lists the family with the reason and checks the rest.
- The NFY specs have no point-in-time embedding columns, so `embedding` is listed there. The article vector is a current snapshot, so using it would leak.

Pass is exit 0, and it requires all of the following:

- Good ranks above bad in every checked family at R1, R2, and R3.
- Kendall tau-b between the R1 and R3 scores is at least 0.6.
- The checked families plus the unsupported ones cover the whole catalog.

What the demo saw:

- `lgbm`, `neural`, and `linear_reweight` held their order at every rung. Tau was 0.733.
- `linear_reweight`'s R2 margin (0.0085) was below the R2 half-width, so that pair separates weakly.

```bash
uv run amt sign g2 --dataset nfy_article_rerank --signer "Your Name"
```

If the only inversion is at R1, rerun with `--skip-r1 <family>` or a higher `--r1-train-fraction`. An inversion at R2 or R3 is a failure: do not sign it.

Check both gates before searching:

```bash
uv run amt gates --brief briefs/nfy_article_rerank.yaml
```

You want `G1 current; G2 current`. G3 and the full budget are still open at this point, and that is expected.

## 4. Search run

Run it in the background and keep a log. A run takes tens of minutes, and most of that time is agent calls.

```bash
uv run amt run briefs/nfy_article_rerank.yaml 2>&1 | tee -a runs/<run-id>/run.log
```

Each round goes through these stages:

| Stage | What happens | Output |
| --- | --- | --- |
| Planner | Read-only, in ask mode, with web search | `runs/<id>/round_<n>_proposals.yaml` |
| Critic | Read-only, in ask mode | `round_<n>_verdicts.json` |
| SPEC | The orchestrator writes each candidate's definition | `candidate.yaml` and `spec.md` |
| Implementer | Writes the model | `candidates/<id>/model.py` |
| Reviewer | Checks the implementation | `candidates/<id>/review.md` |
| Preflight and submit | One `train` job per candidate per rung, using an immutable snapshot on the Volume | Ledger rows |

The pruner runs after each rung. Every agent call is logged in `runs/<id>/agent_calls.jsonl`.

Watch the run from another terminal:

```bash
uv run amt status --run-id <run-id>
```

While the implementer or reviewer is active, the repository hook blocks shell commands it cannot scope to `candidates/<id>/`. That includes `amt status`. Read `runs/<id>/state.json` directly until the stage moves on.

The stage ends at `REPORT` or `ESCALATE`, and `runs/<id>/ESCALATE.md` gives the reason:

| Reason | Meaning | What to do |
| --- | --- | --- |
| `outer retry cap` | Every draft in round 1 was pruned, three times. | Treat this as a result. Record it, and consider widening `design_space`. |
| `refusing submit: … exceeds cap` | The next job would pass the per-candidate cap or the pool. | Check the estimate. The planner's `est_cost` prices only a candidate's first job; later rungs are priced from its measured job. |
| `planner output had no candidates` | The planner returned no JSON. | Check `cursor-agent` auth and the call log. |
| a stale gate | The brief, spec, or manifest changed. | Sign again, then rerun with `--fresh`. |

How resuming works:

- **Resume after a crash.** Rerun the same command. A job key already in `submitted_jobs` is never submitted twice.
- **`ESCALATE` is terminal.** `--fresh` deletes `state.json` and plans again from scratch.
  - Node ids restart at `c01`. Idempotency tokens are `<run_id>:<node>/<rung>/<seed>`, so within 48 hours a fresh run on the same run id can be handed the old Databricks runs.
  - Use a new `--run-id` for a fresh search.
- **Status polling** rides out transient network errors such as DNS lookups and 5xx responses.

Demo result (`nfy-demo`):

- **Calls:** three rounds of drafts, six candidates, and 20 agent calls. All agent calls succeeded, and every review ended `VERDICT: PASS`.
- **Outcome:** every candidate was pruned at R2 below the LightGBM reference. The best was c02, a shallow LambdaRank model over every feature group: 0.5347 against the reference's 0.5472.
- **Spend:** 0.448 of 40 DBU. A candidate-rung cost about 0.014 DBU at R1 and up to 0.14 DBU at R2.
- **Audit:** passed.

```bash
uv run amt status --run-id <run-id> --audit
```

## 5. Red team (not yet run live)

There is no `amt redteam` command. Each row plants one fault, and its control must catch it before an R1 ledger row exists.

| Fault | What you do | Where the catch shows up |
| --- | --- | --- |
| `item_index` used as a feature | Hand the implementer a candidate that reads it. | Preflight fails, or `review.md` ends `VERDICT: FAIL`. |
| Edit under `harness/`, `amt/`, `briefs/`, `datasets/`, `resources/`, `databricks.yml` | Attempt the write from the implementer role. | The hook denies it. A diff that still contains it fails review. |
| Run past the rung deadline | A `fit` that ignores the deadline. | The job timeout fails the run, and DBU is charged. |
| NaN scores | A candidate that returns non-finite scores. | Pruner hard kill: `nan or divergent scores`. |
| Import crash | A candidate that fails on import. | Debug runs up to 3 times (`debug_log.md`). The fourth failure escalates. |
| Claimed change absent | An improve node whose `change` isn't in the diff. | Reviewer `VERDICT: FAIL`. |
| Near-duplicate proposal | Same family and claim twice. | Critic `DROP`. |
| Budget exhaustion | Remaining pool below the next estimate. | `ESCALATE.md`, and nothing is submitted. |

## 6. Calibration, gate G3, once (not yet run live)

This is one lineage, K=1, with a pause after every stage. The holdouts stay closed: the run stops when the lineage finishes R3.

```bash
uv run amt run briefs/nfy_article_rerank.yaml --scale calibration --run-id nfy-g3
uv run amt approve --run-id nfy-g3    # after reading runs/nfy-g3/APPROVAL.md
uv run amt run briefs/nfy_article_rerank.yaml --scale calibration --run-id nfy-g3
```

Repeat the approve-and-run pair until the stage is `REPORT`, then:

```bash
uv run amt status --run-id nfy-g3 --audit
uv run amt sign g3 --brief briefs/nfy_article_rerank.yaml --run-id nfy-g3 --signer "Your Name"
```

`sign g3` refuses anything other than a `pass` calibration packet with a clean audit. G3 happens once for the repo, and a second calibration is refused. `amt stop --run-id nfy-g3` escalates at the next stage boundary. It does not cancel a Databricks run that is already in flight.

## 7. Unattended K=4 run (not yet run live)

This starts after G3, with `approval_mode: none`.

```bash
uv run amt run briefs/nfy_article_rerank.yaml --scale small --run-id nfy-small
uv run amt status --run-id nfy-small
uv run amt stats --run-id nfy-small
```

How the rounds go:

- Round 1 proposes 8 drafts, and the critic keeps 4.
- Later rounds propose 3 improvements of each surviving lineage and keep 2, then one combine.

The run passes, recorded in `gates/small/nfy-small.json`, when all of the following hold:

- The stage is `REPORT`.
- Ledger rows cover at least 3 families.
- Every kill has a reason.
- Spend is within the budget.
- `round_1.md` through `round_4.md` exist.
- Agent stats are on disk.

## 8. Confirmation and the final holdout (not yet run live)

```bash
uv run amt run briefs/nfy_article_rerank.yaml --scale full --run-id nfy-phase2
uv run amt gates --brief briefs/nfy_article_rerank.yaml
```

The first command writes `runs/nfy-phase2/phase2_entry.md` and starts no jobs. `gates` then prints `full budget ready`.

How confirmation works:

- A search-scale run reaches confirmation after the combine round.
- It submits `confirm` for the top 3 promoted nodes, with three seeds, plus the LightGBM reference.
- Each holdout submit uses a one-time confirm token.

Opening the final holdout is gate G5, and it can be done once:

```bash
uv run amt final --run-id nfy-phase2 --signer "Your Name"
```

`amt final` only scores R3 survivors. With none, it refuses (`no R3 survivor to score on the final holdout`) and the holdout stays closed. The demo ended that way. A final delta outside the winner's select interval is an overfit flag. Do not loosen `min_effect` to produce a winner.

Promotion into an NFY A/B test goes through [pnr-ai-compass](https://github.com/foxcorp-product/pnr-ai-compass) after G5.

## 9. Reproducibility (not yet run live)

```bash
uv run amt reproduce --run-id nfy-phase2 --node <winner>
```

This resubmits `train` against the immutable snapshot for that row's git SHA and seed. It passes when the new `ndcg@10_graded` lands inside the row's R3 interval. It needs an R3 ledger row.

## If a stage fails

- **`required scopes: files`:** add the `files` scope to the token and wait about a minute.
- **`redeploy: harness changed since deploy`:** run `amt deploy`. Expect the round-0 and ladder matrices to rerun on their next submit.
- **Noise report `revise`:** raise `min_effect`, run `amt noise --collect` again, then sign. Or lengthen `search_val` or add seeds, rebuild, and rerun round 0.
- **Ladder `incomplete` with a family failing everywhere:** check `failure_reason` on its MLflow runs.
  - Unsupported families are listed automatically.
  - Any other failure needs fixing before G2.
- **Kendall tau below 0.6 with the order intact:** pass `--r1-train-fraction`.
  - An R1-only inversion: use `--skip-r1` for that family.
  - An R2 or R3 inversion stops the runbook.
- **A leak found after search started:** stop the run, treat the affected ledger rows as invalid, add the preflight rule, and resume from the last clean round with the same run id.
- **No candidate beats the LightGBM reference:** write that up and stop. The thresholds stay where G1 signed them.
- **A signed gate went stale:** `amt gates` says which one. Sign it again. If a search already escalated on the old signature, start a new run id.

## What you should have at the end

- `runs/nfy-phase2/manifest.json` and a green `build_dataset` run.
- `gates/g1/nfy_article_rerank/` with a passing noise report and `g1.json`.
- `gates/g2/nfy_article_rerank/embedding+lgbm+linear_reweight+neural/` with a passing ladder report and `g2.json`.
- A signed G3 packet under `gates/g3/`.
- `gates/small/nfy-small.json` with status `pass`, plus `runs/nfy-small/report.md`.
- `runs/nfy-phase2/phase2_entry.md`.
- `uv run amt gates --brief briefs/nfy_article_rerank.yaml` printing `G1 current; G2 current; G3 current; full budget ready`.
