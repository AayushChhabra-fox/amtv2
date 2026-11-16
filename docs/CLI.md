# AMT command line

`amt` runs two pipelines with a human signature between them, then one final signature after search.

1. `amt prep` builds the dataset and writes the readiness packet.
2. `amt sign ready` locks that packet.
3. `amt run` searches until it prints an outcome. `amt resume` continues a saved run.
4. `amt final` opens the final holdout once.

The [operator runbook](phase2_runbook.md) is that sequence for `briefs/nfy_demo.yaml` and `briefs/nfy_demo_v3.yaml`. Run every command from the repository root. `run` uses the current directory. The other commands take `--root` and default to the current directory.

```bash
uv run amt --help
uv run amt <command> --help
```

`uv run python -m amt` is the same entry point.

## Configuration

Copy `.env.example` to `.env`. Do not commit `.env`. `amt` loads it before parsing arguments. A real environment variable wins over `.env`, and `.env` wins over the defaults in `amt/config.py`. A limit in `.env` can only tighten a brief: concurrency is the smaller of `AMT_MAX_CONCURRENT_JOBS` and `budget.max_concurrent_jobs`. The DBU budget stays on the brief and is locked again at `amt sign ready`.

Auth is one of three methods: `DATABRICKS_TOKEN`, a service principal (`DATABRICKS_CLIENT_ID` and `DATABRICKS_CLIENT_SECRET`), or `DATABRICKS_CONFIG_PROFILE`. `dbx_host` and `dbx_token` are still copied onto the Databricks names when those are unset.

`AMT_CATALOG`, `AMT_SCHEMA`, `AMT_ARTIFACT_VOLUME`, and `AMT_MLFLOW_EXPERIMENT_ROOT` select the workspace target. The experiment for a brief is `<root>/<brief name>`. `amt deploy` passes those values into the bundle as `--var`.

`AMT_CLUSTER_ID` runs every job on that cluster. Leave it blank to use the job clusters in `resources/*.job.yml`. On a shared cluster, search latency is `latency_suspect`. The hard latency kill applies when that run is the only AMT job in flight. Spend is wall-clock times `AMT_DBU_PER_HOUR`.

`AMT_GPU_CLUSTER_ID` or `AMT_GPU_NODE_TYPE` is required before a proposal with `device: gpu` can run. The brief also has to set `constraints.allow_gpu_serving: true`. With the default `false`, GPU families are out of the viable set.

`AMT_HF_OFFLINE_CACHE` is the Unity Catalog volume that already holds any HuggingFace snapshot a candidate is allowed to read. `from_pretrained` and other hub downloads are rejected unless this is set. Example: `/Volumes/product_api_dev/amt/amt_artifacts/hf`.

`AMT_DATASET_REPAIR_MAX_ATTEMPTS` (default 3) and `AMT_DATASET_REPAIR_MAX_DBU` (default 10) cap `amt prep --repair`. The fixer may edit `datasets/<spec>/features/*.sql` only. Each attempt re-runs the 90-second slice. Diffs are kept under `runs/dataset-repair/<spec>/<n>/`.

`amt doctor` prints the auth method and checks the workspace, the clusters, and `cursor-agent status`.

Agent calls use `AMT_AGENT_BACKEND`:

| Value | Behavior |
| --- | --- |
| `stub` | Default. Writes the pinned toy candidate. A search brief needs a real planner. |
| `cursor` | Calls `cursor-agent -p`. The planner and critic run read-only in ask mode, and the planner requires web search. |
| `auto` | Uses `cursor` when `cursor-agent status` shows a login, otherwise records the miss and uses the stub. |
| `fake` | Test double. |

Model and limit variables live in `.env.example`: `AMT_MODEL_PLANNER`, `AMT_MODEL_CRITIC`, `AMT_MODEL_IMPLEMENTER`, `AMT_MODEL_REVIEWER`, `AMT_AGENT_TIMEOUT_S` (default 600), `AMT_TOKEN_CAP` (default 32000), `CURSOR_API_KEY`. A transient agent error (`resource_exhausted`, `[internal]`, `Connection lost`, timeout) is retried three times. Job submission uses `AMT_JOBS_BACKEND` (`databricks` or `fake`). `AMT_<JOB>_JOB_ID` overrides the id from the bundle summary for `train`, `confirm`, `final_holdout`, and `build_dataset`.

## Commands

| Command | Purpose |
| --- | --- |
| `prep` | Doctor, deploy if stale, preflight, build, and parallel round-0 noise and ladder. |
| `sign ready` | Lock the readiness packet. |
| `run` | Search until the outcome is printed. |
| `resume` | Continue a saved run, including one that escalated. |
| `final` | Open the final holdout once. |
| `deploy` | Build the wheel and deploy the bundle. |
| `doctor` | Check auth, the workspace, clusters, and the agent CLI. |
| `dataset build` | Submit the dataset build. `amt prep` does this for you. |
| `dataset preflight` | Run the 90-second slice. `--repair` fixes feature SQL first. |
| `dataset revert` | Restore feature SQL from the first repair snapshot. |
| `noise` | Write the round-0 noise report. `amt prep` does this for you. |
| `ladder` | Write the ladder report. `amt prep` does this for you. |
| `gates` | Show whether the ready signature is current. |
| `status` | Print stage, spend, jobs, live nodes, and kills. |
| `stop` | Ask a run to escalate at the next stage. |
| `approve` | Unblock one approval pause. |
| `stats` | Print agent statistics for one run. |
| `reproduce` | Rerun one ledger row and compare it with its interval. |
| `sign g3` | Sign the optional calibration packet. |

Exit status is `0` when a report finishes, including `no_winner`. `run` and `resume` return `1` on `ESCALATE` or an approval pause. `prep` returns `2` when the packet is not a pass, and `1` when it cannot be built. `sign` returns `1` when the packet is not signable. `final` returns `1` when there is no R3 survivor or the holdout was already opened. `status --audit` returns `1` when a ledger row fails the trace check. `gates` returns `1` when a search brief is blocked.

### `prep`

```bash
uv run amt prep --brief <brief> [--repair]
```

Pipeline A. The order is fixed:

1. `doctor`.
2. `deploy` when `harness/` or `resources/` is newer than `.amt/deployed.json`. A spec edit does not by itself force a harness redeploy.
3. The 90-second dataset preflight: local split and metric checks, then a 1-hour slice (`EXPLAIN`, key and point-in-time checks, label support).
4. With `--repair`, a repairable error in `datasets/<spec>/features/*.sql` is handed to `dataset_fixer` and the slice runs again, up to the attempt and DBU caps. Errors in `queries.sql`, `candidates.sql`, `labels.sql`, splits, leakage, or the harness stop here with a diagnosis.
5. The full Spark build. It writes Delta splits, Volume Parquet caches, and `manifest.json`.
6. The manifest is copied to `gates/ready/<brief>/manifest.json`.
7. Round-0 noise and the ladder are submitted together, then collected.

The brief file is not edited. `min_effect: auto` is resolved to `max(0.010, 2× the R3 paired-delta half-width)` and stored on the packet.

Outputs, all under `gates/ready/<brief>/`:

- `ready_report.json` and `ready_report.md`
- `manifest.json`
- `noise_report.json` and `noise_report.md`
- `ladder_report.json` and `ladder_report.md`

Accepted feature-SQL diffs are listed on the packet and stored under `runs/dataset-repair/<spec>/<n>/`.

### `sign ready`

```bash
uv run amt sign ready --brief <brief> --signer "Your Name"
```

The signer is a person's name. `auto`, `amt`, `system`, `orchestrator`, and `agent` are rejected. A packet whose status is not `pass` is rejected.

This writes `gates/ready/<brief>/ready.json`. Commit that file. It locks the problem and task hash, the spec hash, the manifest hash, accepted repair hashes, the resolved `min_effect`, the DBU budget, and the viable ladder families. Optional `design_space.families` hints are not part of the hash. Editing the brief's problem, task, constraints, budget, the spec, or the manifest makes the signature stale.

`amt sign g3` remains the optional calibration signature. It is not required before `amt run`.

### `gates`

```bash
uv run amt gates --brief <brief>
```

A walking-skeleton brief (one with `candidate:`) prints `walking skeleton; the ready gate is not required` and exits `0`. A search brief prints:

```text
ready current; calibration not signed
```

`current` means `ready.json` still matches the brief, the spec, the manifest, and the repair diffs. Exit `1` means `amt run` would escalate instead of starting.

### `run`

```bash
uv run amt run <brief> [--run-id ID] [--fresh]
  [--approval-mode none|every_stage|round_shortlist]
  [--scale search|calibration|small|full]
```

The run id is `--run-id`, then `run_id` on the brief, then `name`, then the filename stem. State is `runs/<id>/state.json`. `--fresh` deletes that file before starting. Use it after a stale ready signature you intend to replace. Leave it off when you mean to resume; prefer `amt resume` once the run has escalated.

A search brief with no pinned candidate stops in `ESCALATE` until `ready.json` is current. The message says to prep, sign, and re-run with `--fresh`.

On a signed brief the loop materializes `lgbm_reference` as node `c00` under `runs/<id>/candidates/c00/`, then drafts families from the modality profile and the brief constraints. A `design_space.families` list is an optional hint, not a menu the planner must follow.

Each round resets its revision counters. A critic batch that passes at least one candidate drops the unrevised stragglers and continues. A candidate that raises at R1 is sent back through `DEBUG` (up to 3 attempts) while its siblings keep running. A cluster `INTERNAL_ERROR` is retried once, then the run escalates with the node still ready to submit.

Progress lines go to stderr as `[HH:MM:SS] [ROUND r | STAGE]`.

A finished search prints one line and exits `0`:

```text
OUTCOME: winner_promoted
OUTCOME: baseline_c00_promoted
OUTCOME: no_winner
```

`winner_promoted` means a candidate cleared the production bar and beat `c00`. `baseline_c00_promoted` means `c00` cleared that bar and no candidate beat it. `no_winner` means nothing cleared it. The same value is in `runs/<id>/report.json`. `ESCALATE` exits `1` and writes `runs/<id>/ESCALATE.md` plus a `report.json` with no outcome.

`--approval-mode` overrides the brief. `none` writes `runs/<id>/round_<n>.md` and continues. `round_shortlist` pauses after the critic. `every_stage` pauses after each stage. A pause exits `1` and writes `runs/<id>/APPROVAL.md`. Continue with `amt approve --run-id ID`, then `amt resume --run-id ID`.

`--scale calibration` is the optional one-time diagnostic. It forces K=1 and `approval_mode: every_stage`, and it stops before confirmation. `small` and `full` are still accepted; they do not replace the ready gate.

On the Databricks backend, search submits `train` and confirmation submits `confirm`. Each candidate is uploaded to `<volume>/candidates/<run_id>/<node_id>/<git_sha>/`. A submit is refused when `harness/` or `resources/` have changed since `amt deploy`. The message is `redeploy: harness changed since deploy`.

### `resume`

```bash
uv run amt resume --run-id ID
```

Loads `runs/<id>/state.json` and continues. A run that stopped in `ESCALATE` restarts at the stage that stopped it. A budget refusal restarts at submit after you raise `budget.total_dbu` on a walking-skeleton brief. On a search brief the budget is inside `ready.json`, so a higher pool needs `amt prep` and `amt sign ready` again before a fresh run id. A cluster escalation leaves the candidate ready; `amt resume` submits that job again. A job key that already has a run id is not submitted twice.

### `final`

```bash
uv run amt final --run-id ID --signer "Your Name"
```

Opens `final_holdout` once for the top R3 survivors (kept or promoted), plus production and the LightGBM reference. A second call is refused. With no R3 survivor the command exits `1` and the holdout stays closed. A final delta outside the winner's select interval is an overfit flag in `runs/<id>/final_holdout.md`.

### `status`, `stop`, `approve`

```bash
uv run amt status [--run-id ID] [--audit]
uv run amt stop [--run-id ID]
uv run amt approve --run-id ID
```

`--run-id` defaults to `toy` for `status` and `stop`. The status report is the stage, the round, the outcome when one is set, spent DBU against the pool, each submitted job, live nodes, and every kill.

`--audit` checks that every ledger row for the run has a job run id, an MLflow run id, and a git SHA other than `unknown`. For a non-fake job id it also calls the Jobs API and the MLflow get-run API.

`stop` writes `runs/<id>/STOP`. The orchestrator escalates at the next stage boundary. It does not cancel a Databricks run already in flight.

`approve` writes `runs/<id>/APPROVE`. The next `amt resume` of that id consumes the file.

### `dataset`

```bash
uv run amt dataset preflight --spec <spec> [--brief <brief>] [--repair] [--local-only]
uv run amt dataset build --spec <spec> [--brief <brief>] [--preflight] [--repair] [--wait]
uv run amt dataset revert --spec <spec>
```

`preflight` is the 90-second check. `--local-only` skips the slice. `--repair` is the same fixer `amt prep --repair` uses. `revert` copies feature SQL back from the first snapshot under `runs/dataset-repair/<spec>/`.

### `noise` and `ladder`

`amt prep` submits and collects both. The standalone commands are there when you need to rebuild one report from observations you already have:

```bash
uv run amt noise --brief <brief> [--plan] [--submit] [--collect] [--observations <file>]
uv run amt ladder --dataset <spec-or-name> [--plan] [--submit] [--collect]
  [--observations <file>] [--skip-r1 FAMILY] [--r1-train-fraction FLOAT]
```

Reports land in `gates/ready/<brief>/`, not in a separate gate directory. Noise passes when every baseline cell is present. With `min_effect: auto` the packet resolves the floor; an explicit `min_effect` below twice the R3 half-width is `revise`. The ladder passes when the good design ranks above the bad one in each checked family and Kendall tau-b between R1 and R3 is at least 0.6. A family the dataset cannot support is listed and the others are checked.

### `reproduce` and `stats`

```bash
uv run amt reproduce --run-id ID --node <id>
uv run amt stats --run-id ID
```

`reproduce` resubmits `train` for that ledger row and passes when the new metric lands inside the recorded interval. `stats` reads `runs/<id>/state.json`, writes `runs/<id>/agent_stats.json`, and prints it.

## Files

| Path | When |
| --- | --- |
| `gates/ready/<brief>/ready_report.json` | After `amt prep`. |
| `gates/ready/<brief>/ready.json` | After `amt sign ready`. Commit this. |
| `gates/ready/<brief>/manifest.json` | The manifest prep downloaded. Part of the signature. |
| `runs/<id>/state.json` | After every stage. |
| `runs/<id>/candidates/<node>/` | Implementer output. `c00` is the materialized LightGBM reference. |
| `runs/<id>/round_<n>.md` | After each round. |
| `runs/<id>/report.json` | Finished search. `outcome` is the scriptable field. |
| `runs/<id>/report.md` | Human summary, including `Outcome:`. |
| `runs/<id>/ESCALATE.md` | Budget, cluster, caps, a stop file, or a stale ready gate. |
| `runs/<id>/APPROVAL.md` | While an approval mode is waiting. |
| `runs/<id>/agent_calls.jsonl` | One record per agent call, including retry counts. |
| `runs/dataset-repair/<spec>/<n>/` | Feature SQL before, after, and the unified diff. |
| `ledger/results.jsonl` | Append-only. Written by the orchestrator and the pruner, not by agents. |

`gates/ready/` is the git-tracked sign-off directory. Do not recreate `gates/g1/` or `gates/g2/`.

## Bundle jobs

`amt deploy` is the deploy step. `amt` submits every job except `pruner`, which remains a direct bundle run for replaying a saved rung file.

| Job key | Who submits it |
| --- | --- |
| `build_dataset` | `amt prep`, `amt dataset build`, `amt dataset preflight` |
| `train` | `amt prep` (noise and ladder), `amt run`, `amt reproduce` |
| `confirm` | `amt run`, after promotion. Holdouts need a one-time confirm token. |
| `final_holdout` | `amt final` |
| `pruner` | `databricks bundle run pruner` when you want to replay a file by hand |
