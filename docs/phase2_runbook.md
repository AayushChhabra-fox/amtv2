# Operator runbook

This is the sequence for a News For You search. The commands are documented in [CLI.md](CLI.md). Two briefs are ready to run:

- `briefs/nfy_demo.yaml` on `datasets/nfy_article_rerank_demo` (six hours of 2026-10-07, 40 DBU, K=2).
- `briefs/nfy_demo_v2.yaml` on `datasets/nfy_article_rerank_gold` (2026-10-01 through 2026-10-18, 40 DBU, K=2).

Both briefs are problem statements. `min_effect` is `auto`. GPU serving and external weights are off, the license policy is commercial-only, and the serving budget is 20 ms p95 per 100 candidates. Neither brief lists model families. Prep measures the dataset and locks the viable set; the planner drafts from that profile.

The split of work:

- **`amt prep` submits:** `build_dataset`, then round-0 `train` and the ladder, together.
- **You sign once:** `amt sign ready`.
- **`amt run` submits:** search `train` jobs, then `confirm` for whoever the promotion rule selects.
- **You sign once more:** `amt final`, after a promoted outcome.
- **You do not** run `databricks bundle run` for any of these jobs, and you do not edit the brief to raise `min_effect` or to add families.

Run every command from the repository root. Each stage starts only after the previous one has passed.

## 0. Setup

### `.env`

Copy `.env.example` to `.env` and fill in these values:

- **Auth.** Use one of: `DATABRICKS_HOST` with `DATABRICKS_TOKEN`, an OAuth client, or `DATABRICKS_CONFIG_PROFILE`. The legacy keys `dbx_host` and `dbx_token` are still read. `amt` never picks a profile for you.
- **Token scopes.** A scoped personal access token needs `files` as well as `jobs`, `workspace`, `clusters`, `mlflow`, `sql`, and `unity-catalog`. Matrices, candidate snapshots, and confirm tokens are written to the volume. Without `files`, any volume write fails with `required scopes: files`.
- **`AMT_CLUSTER_ID`.** Set this to run every job on one all-purpose cluster. Leave it blank to use the job clusters in `resources/*.job.yml`.
- **`AMT_HF_OFFLINE_CACHE`.** Set this only when a candidate may read a HuggingFace snapshot that is already on a Unity Catalog volume. Hub downloads are rejected when it is unset.
- **`AMT_DATASET_REPAIR_MAX_ATTEMPTS` and `AMT_DATASET_REPAIR_MAX_DBU`.** Caps for `amt prep --repair`. Defaults are 3 attempts and 10 DBU. The fixer edits `datasets/<spec>/features/*.sql` only.
- **Agents.** Set `AMT_AGENT_BACKEND=cursor` and `CURSOR_API_KEY`. The `stub` backend only knows the toy candidate.

Never commit `.env` or print its values.

### Deploy

```bash
uv run amt doctor
uv run amt deploy
```

`amt prep` deploys again when `harness/` or `resources/` is newer than `.amt/deployed.json`. Editing one dataset spec does not invalidate the harness hash. A Databricks submit of training code is refused with `redeploy: harness changed since deploy` until the hashes match.

Leave `spark.sql.ansi.enabled` at its default (on) on the cluster.

### The demo window

`briefs/nfy_demo.yaml` uses six hours of 2026-10-07:

| Split | Window (UTC) |
| --- | --- |
| train | 12:00–15:00 |
| search_val | 15:00–16:00 |
| select_holdout | 16:00–17:00 |
| final_holdout | 17:00–18:00 |

`briefs/nfy_demo_v2.yaml` uses whole-day splits from 2026-10-01 through 2026-10-18. The demo does not consume the optional calibration. Its ready signature does not carry over: the signature is per brief, and it hashes that brief's spec.

## 1. Prep

```bash
uv run amt prep --brief briefs/nfy_demo.yaml --repair
```

What this does, in order:

- Runs `amt doctor` and deploys if the harness hash is stale.
- Runs the local checks (the demo spec allows a short `search_val`; a full spec needs five distinct days) and the 1-hour slice.
- With `--repair`, a cast or analysis error inside `features/*.sql` is repaired and the slice runs again. The brief and `dataset.yaml` are not rewritten. A diff is stored at `runs/dataset-repair/<spec>/<n>/diff.patch`. Anything else (labels, splits, leakage, harness bugs) stops with the diagnosis and does not start the full build.
- Builds the dataset, downloads `manifest.json` to `gates/ready/nfy_demo/manifest.json`, and submits noise and the ladder together.

Read `gates/ready/nfy_demo/ready_report.md` when the command exits 0:

- **Status `pass`.** The noise report is complete, the ladder order holds, and at least one family is viable under the latency, CPU, and license constraints.
- **Resolved `min_effect`.** This is `max(0.010, 2× the R3 half-width)`. It is not written back into the brief.
- **Viable families.** The intersection of the manifest modalities and the brief constraints. Embedding is omitted when the spec has no point-in-time embedding columns.
- **Repairs.** Any feature-SQL diff the fixer applied. `amt dataset revert --spec datasets/nfy_article_rerank_demo` restores the first snapshot.

Exit `2` means the packet is `fail` or `incomplete`. The remediation is in the report. Exit `1` means prep stopped before a packet existed (auth, a non-repairable preflight failure, or a failed build).

The v2 brief is the same command with `briefs/nfy_demo_v2.yaml`. Its packet is `gates/ready/nfy_demo_v2/`.

## 2. Sign the ready gate

```bash
uv run amt sign ready --brief briefs/nfy_demo.yaml --signer "Your Name"
uv run amt gates --brief briefs/nfy_demo.yaml
```

`sign ready` writes `gates/ready/nfy_demo/ready.json`. Commit it. The signature locks the problem and task, the spec hash, the manifest hash, the repair hashes, the resolved `min_effect`, the 40 DBU budget, and the viable families.

`gates` should print `ready current`. A later edit to the brief's budget, constraints, or task, or to the spec or the manifest, makes it stale. Prep and sign again. Do not hand-edit `ready.json`.

## 3. Search

```bash
uv run amt run briefs/nfy_demo.yaml 2>&1 | tee -a runs/nfy-demo/run.log
```

The run id comes from the brief (`nfy-demo` or `nfy-demo-v2`) unless you pass `--run-id`. Stderr shows `[HH:MM:SS] [ROUND r | STAGE]` for every stage. Watch another terminal with:

```bash
uv run amt status --run-id nfy-demo
```

What the loop does:

- Materializes `lgbm_reference` as `runs/<id>/candidates/c00/` and copies its R1–R3 round-0 intervals onto that node.
- Round 1 drafts families the modality profile allows. Rounds 2 and 3 improve. Round 4 combines. `max_rounds` on the demo briefs stops the schedule early.
- A critic `REVISE` on part of a batch keeps the passing candidates.
- An R1 exception in candidate code is appended to that node's `debug_log.md` and the node goes back through `DEBUG`, up to three times.
- A cluster `INTERNAL_ERROR` is retried once. A second kill escalates. The candidate stays ready to submit.
- Promotion is symmetric. A candidate must beat production by the resolved `min_effect` with `ci_lo > 0`, stay inside the latency budget, and beat `c00` (`delta_vs_lgbm.ci_lo > 0`). If no candidate beats `c00` and `c00` clears the production bar, `c00` is confirmed.

The process prints one of these and exits 0:

```text
OUTCOME: winner_promoted
OUTCOME: baseline_c00_promoted
OUTCOME: no_winner
```

`runs/<id>/report.json` has the same `outcome`. `no_winner` is a finished result, not a failure. Exit `1` means `ESCALATE`. Read `runs/<id>/ESCALATE.md`.

| Reason | What to do |
| --- | --- |
| `refusing submit` | The next job would pass the signed DBU pool. The pool is inside `ready.json`. Raise it only by editing the brief, running `amt prep` again, and signing again. Then start a new run id. |
| `infrastructure failure: infra_crash` | The cluster killed the job twice. `amt resume --run-id <id>` submits it again. |
| `debug cap` | One node used three debug attempts and nothing else was left to run. Read that node's `debug_log.md`. |
| `critic revision cap` | The critic rejected every candidate twice. |
| `outer retry cap` | Round 1 produced no survivor three times. Treat that as a result. |
| ready gate is not signed, or it is stale | `amt prep`, `amt sign ready`, then `amt run --fresh` with a new run id. |

`amt resume --run-id <id>` is the way back into a saved run. It does not allocate a new id. A job key that already has a run id is not submitted again. `amt stop --run-id <id>` escalates at the next stage boundary and does not cancel a Databricks run that is already in flight.

```bash
uv run amt status --run-id nfy-demo --audit
```

## 4. Final holdout

Run this after `winner_promoted` or `baseline_c00_promoted`:

```bash
uv run amt final --run-id nfy-demo --signer "Your Name"
```

`amt final` scores the top R3 survivors once, plus production and the LightGBM reference, and writes `runs/<id>/final_holdout.md`. A second call is refused. If the search outcome is `no_winner` and no R3 row was kept, the command exits `1` and the holdout stays closed. Do not loosen `min_effect` to produce a winner. The floor was locked at sign-off.

```bash
uv run amt reproduce --run-id nfy-demo --node <winner>
```

This resubmits `train` against the immutable snapshot for that row's git SHA and seed. It passes when the new metric lands inside the row's interval.

## Optional calibration

Calibration is a diagnostic, not a gate in front of search.

```bash
uv run amt run briefs/nfy_demo.yaml --scale calibration --run-id nfy-calibration
uv run amt approve --run-id nfy-calibration
uv run amt resume --run-id nfy-calibration
```

Repeat approve and resume until the stage is `REPORT`. Then:

```bash
uv run amt status --run-id nfy-calibration --audit
uv run amt sign g3 --brief briefs/nfy_demo.yaml --run-id nfy-calibration --signer "Your Name"
```

G3 is once for the repo. A second calibration is refused.

## If a stage fails

- **`required scopes: files`:** add the `files` scope to the token and wait about a minute.
- **`redeploy: harness changed since deploy`:** run `amt deploy`, or let the next `amt prep` deploy. Expect round 0 to rerun when the harness hash changes.
- **Preflight cast or analysis error in `features/*.sql`:** rerun `amt prep --repair`. Inspect `runs/dataset-repair/<spec>/<n>/diff.patch` before you sign.
- **Preflight error in labels, splits, or leakage:** do not repair it by hand inside `features/`. Fix the spec or the source data and prep again.
- **Repair attempt cap or DBU cap:** the message names the cap. `amt dataset revert --spec <spec>` restores the original feature SQL.
- **Ready report `revise` or `fail`:** follow the remediation in `ready_report.md`. An explicit `min_effect` below the measured floor is the usual noise failure; these briefs leave it on `auto` so prep can raise the locked value without editing the file.
- **A signed gate went stale:** `amt gates` says so. Prep and sign again.

## What you should have at the end

- `gates/ready/nfy_demo/ready_report.md` with status `pass`, and `gates/ready/nfy_demo/ready.json` committed.
- The same pair for `nfy_demo_v2` when that brief has been prepped and signed.
- `runs/<id>/report.json` whose `outcome` is `winner_promoted`, `baseline_c00_promoted`, or `no_winner`.
- `runs/<id>/final_holdout.md` when a promoted outcome was scored.
- `uv run amt gates --brief briefs/nfy_demo.yaml` printing `ready current`.

`gates/ready/` is the only gate directory to commit. `gates/g1/` and `gates/g2/` are not part of this flow.
