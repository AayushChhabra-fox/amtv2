# AMT command line

`amt` is the control-loop command. It runs a brief, deploys the bundle, builds a dataset, submits the round-0 and ladder matrices, and scores the final holdout. The only steps that stay manual are the gate signatures. The [Phase 2 runbook](phase2_runbook.md) is the sequence for `briefs/nfy_article_rerank.yaml`. Run `briefs/nfy_demo.yaml` first.

Run every command from the repository root. `run`, `status`, `stop`, and `approve` use the current directory. The other commands take `--root` and default to the current directory.

```bash
uv run amt --help
uv run amt <command> --help
```

`uv run python -m amt` is the same entry point.

## Configuration

Copy `.env.example` to `.env`. Do not commit `.env`. `amt` loads it before parsing arguments. A real environment variable wins over `.env`, and `.env` wins over the defaults in `amt/config.py`. A limit in `.env` can only tighten a brief: concurrency is the smaller of `AMT_MAX_CONCURRENT_JOBS` and `budget.max_concurrent_jobs`. Budgets stay on the brief.

Auth is one of three methods: `DATABRICKS_TOKEN`, a service principal (`DATABRICKS_CLIENT_ID` and `DATABRICKS_CLIENT_SECRET`), or `DATABRICKS_CONFIG_PROFILE`. `dbx_host` and `dbx_token` are still copied onto the Databricks names when those are unset.

`AMT_CATALOG`, `AMT_SCHEMA`, `AMT_ARTIFACT_VOLUME`, and `AMT_MLFLOW_EXPERIMENT_ROOT` select the workspace target. The experiment for a brief is `<root>/<brief name>`. `amt deploy` passes those values into the bundle as `--var`.

`AMT_CLUSTER_ID` runs every job on that cluster. Leave it blank to use the job clusters in `resources/*.job.yml`. Confirmation uses a one-time token on the volume, because an existing cluster cannot take per-run environment variables. Latency on a shared cluster during search is `latency_suspect`; the hard latency kill applies at confirmation, which runs with no other AMT job in flight. Spend is wall-clock times `AMT_DBU_PER_HOUR`.

`AMT_GPU_CLUSTER_ID` or `AMT_GPU_NODE_TYPE` is required for `device: gpu`. With neither set, preflight drops GPU proposals.

`amt doctor` prints the auth method and checks the workspace, the clusters, and `cursor-agent status`.

Agent calls use `AMT_AGENT_BACKEND`:

| Value | Behavior |
| --- | --- |
| `stub` | Default. Writes the pinned toy candidate. A search brief needs a real planner. |
| `cursor` | Calls `cursor-agent -p`. The planner also requires plan mode with web search. |
| `auto` | Uses `cursor` when `cursor-agent status` shows a login, otherwise records the miss and uses the stub. |
| `fake` | Test double. |

Model and limit variables live in `.env.example`: `AMT_MODEL_PLANNER`, `AMT_MODEL_CRITIC`, `AMT_MODEL_IMPLEMENTER`, `AMT_MODEL_REVIEWER`, `AMT_AGENT_TIMEOUT_S` (default 600), `AMT_TOKEN_CAP` (default 32000), `CURSOR_API_KEY`. Job submission uses `AMT_JOBS_BACKEND` (`databricks` or `fake`). `AMT_<JOB>_JOB_ID` overrides the id from the bundle summary for `TRAIN`, `CONFIRM`, `FINAL_HOLDOUT`, `BUILD_DATASET`, and `HELLO_TRAIN`.

## Commands

| Command | Purpose |
| --- | --- |
| `deploy` | Build the wheel, deploy the bundle, and record the harness hash. |
| `doctor` | Check auth, the workspace, clusters, and the agent CLI. |
| `dataset build` | Submit the dataset build. `--wait` blocks until the job finishes. |
| `run` | Advance one brief until it reports, escalates, or waits for approval. |
| `final` | Open the final holdout once for a run. |
| `reproduce` | Rerun one ledger row and compare it with its interval. |
| `status` | Print stage, spend, jobs, live nodes, and kills. |
| `stop` | Ask a run to escalate at the next stage. |
| `approve` | Unblock one approval pause. |
| `noise` | Write the round-0 noise report for gate G1. |
| `ladder` | Write the ladder report for gate G2. |
| `sign` | Record a human signature for G1, G2, or G3. |
| `gates` | Show whether a search brief is allowed to start. |
| `stats` | Print agent statistics for one run. |

Exit status is `0` for a finished report or a clean audit. `run` returns `1` when the stage is anything else, including an approval pause and `ESCALATE`. `noise` and `ladder` return `2` when the report is `revise`, `fail`, or `incomplete`, and `1` when the input cannot be read. `status --audit` returns `1` when any ledger row fails the trace check. `gates` returns `1` when a search brief is blocked.

### `run`

```bash
uv run amt run <brief> [--run-id ID] [--fresh]
  [--approval-mode none|every_stage|round_shortlist]
  [--scale search|calibration|small|full]
```

The run id is `--run-id`, then `run_id` on the brief, then `name`, then the filename stem. State is `runs/<id>/state.json`. The same command resumes that file. A job key already recorded in `submitted_jobs` is not submitted again. `--fresh` deletes `state.json` before starting. Use it after a gate rejection or an escalation you intend to replace. Leave it off when you mean to resume.

`--approval-mode` overrides the brief, except on a calibration run. `none` writes `runs/<id>/round_<n>.md` and continues. `round_shortlist` pauses after the critic. `every_stage` pauses after each stage. A pause exits `1` and writes `runs/<id>/APPROVAL.md`. `amt status` shows the next stage. Continue with `approve`, then the same `run` command.

`--scale` overrides the brief:

| Scale | What it does |
| --- | --- |
| `search` | Default. Draft, improve, combine, then confirmation of promoted nodes. Requires a current G1 and G2 on a brief with no pinned candidate. |
| `calibration` | One-time G3. Forces K=1 and `approval_mode: every_stage`. Stops at the end of that lineage, before confirmation, so the holdouts stay closed. A second calibration is refused. |
| `small` | K=4 unattended run. Requires `approval_mode: none` and a signed G3. Round 1 proposes 8 drafts and keeps 4 across at least 3 families. |
| `full` | Writes `runs/<id>/phase2_entry.md` when G3 is signed and a small run has passed. Starts no training. |

A brief with a `candidate:` block is the walking skeleton (`briefs/toy.yaml`). Scale values other than `search` are refused for that brief. A search brief with no pinned candidate stops in `ESCALATE` until G1 and G2 are current. The message tells you to sign the gate and re-run with `--fresh`.

On the Databricks backend, `run` submits `train` for a search brief and `confirm` for confirmation. A walking-skeleton brief (one with `candidate:`) still submits `hello_train`. Job ids come from `amt deploy`'s bundle summary, unless `AMT_<JOB>_JOB_ID` is set. Each candidate is uploaded to `<volume>/candidates/<run_id>/<node_id>/<git_sha>/` and that `model.py` is the `--model` path. The snapshot is not overwritten. Search submits include `--baseline-scores` for the same-rung LightGBM reference from round 0. If those scores are missing, the run escalates with `run amt noise --submit first`.

A submit is refused when `harness/`, `datasets/`, or `resources/` have changed since `amt deploy`. The message is `redeploy: harness changed since deploy`. The idempotency token is `<run_id>:<candidate_id>/<rung>/<seed>`.

Stopping the process and starting the same command again resumes. `amt stop` is the halt that escalates on the next stage boundary.

### `status`

```bash
uv run amt status [--run-id ID] [--audit]
```

`--run-id` defaults to `toy`. The report is the stage, the round, spent DBU against the pool, each submitted job, a table of live nodes, and every kill with its reason.

`--audit` checks that every ledger row for the run has a job run id, an MLflow run id, and a git SHA other than `unknown`. For a non-fake job id it also calls the Jobs API and the MLflow get-run API. `audit PASS` is the success line. Any problem prints `audit FAIL` and the process exits `1`.

### `stop`

```bash
uv run amt stop [--run-id ID]
```

Writes `runs/<id>/STOP`. The default id is `toy`. The orchestrator reads the file at the next stage boundary, sets the stage to `ESCALATE`, and writes `runs/<id>/ESCALATE.md`. It does not cancel a Databricks run already in flight.

### `approve`

```bash
uv run amt approve --run-id ID
```

Writes `runs/<id>/APPROVE`. The next `amt run` of that id consumes the file and clears the pause. This command does not resume the loop by itself.

### `noise`

```bash
uv run amt noise --brief <brief> [--plan] [--observations <file>] [--root DIR]
```

Gate G1. `--plan` prints the round-0 `train` parameters and does not submit them. `--submit` launches one matrix job per rung (the LightGBM cells also write keyed scores). `--collect` waits, writes `round0.json`, and builds the noise report. With neither flag, `amt noise` submits and then collects. A second `--submit` does not launch a rung that already has a run id.

`--observations` reads a JSON list, or an object with an `observations` list, and writes:

- `gates/g1/<brief-name>/noise_report.json`
- `gates/g1/<brief-name>/noise_report.md`

The brief name is `name` on the brief, or the filename stem. Each cell is one baseline, rung, and seed on `search_val`:

```json
{
  "baseline_id": "lgbm_reference",
  "rung": "R3",
  "seed": 0,
  "split": "search_val",
  "metric": 0.41,
  "delta_point": 0.004,
  "delta_ci_lo": 0.001,
  "delta_ci_hi": 0.007
}
```

`metric` is graded NDCG@10 (`ndcg@10_graded` on the train task output). A nested `delta` object with `point`, `ci_lo`, and `ci_hi` is accepted. `production` cells do not need a delta. Every other cell needs a finite paired delta against the same-rung production scores, with `ci_hi` at least `ci_lo`. A holdout split is rejected.

The report passes when every cell is present and `min_effect` on the brief is at least twice the R3 paired-delta half-width. The half-width at a rung is the larger of the day-blocked bootstrap half-width and the three-seed t interval. A missing cell is `incomplete`. A `min_effect` under the floor is `revise`. Either result exits `2` and lists the remediation in the report. Whether LightGBM beats production is recorded either way.

Changing `min_effect` or the budget after the report means running `noise` again before `sign`. The signature hashes the report.

### `ladder`

```bash
uv run amt ladder --dataset <spec-or-name> [--plan] [--observations <file>]
  [--skip-r1 FAMILY] [--r1-train-fraction FLOAT] [--root DIR]
```

Gate G2, once per dataset and family set. `--dataset` may be `datasets/nfy_article_rerank` or the spec name `nfy_article_rerank`. `--plan` prints the 48 jobs. `--submit` and `--collect` work the same way as `amt noise`: three matrix runs, one per rung.

| Family | Good design | Bad design |
| --- | --- | --- |
| `lgbm` | `lgbm_full` | `lgbm_recency_category` |
| `neural` | `neural_normal` | `neural_label_shuffled` |
| `embedding` | `embedding_trained` | `embedding_random_projection` |
| `linear_reweight` | `linear_fitted` | `linear_production_weights` |

The `baseline` parameter of each job is the design id. `harness/train.py` treats that id as a ladder ranker.

A cell is `design_id`, `rung`, `seed`, `split`, and `metric`. The report passes when the good design ranks above the bad one in each family at every checked rung, and Kendall tau-b between the R1 and R3 scores is at least 0.6. Reports land in `gates/g2/<dataset>/<family-key>/`, where the family key is the sorted family names joined with `+` (`embedding+lgbm+linear_reweight+neural`).

`--skip-r1 FAMILY` drops R1 from the order check for that family. Repeat the flag for more than one family. `--r1-train-fraction` records the R1 sample fraction used for a re-run. Both values are stored on the report and on the G2 signature. A failure only at R1 is the case for skipping R1. A low tau with the within-family order intact is the case for a higher R1 fraction. A failure at R2 or R3 stays a failure.

### `sign`

```bash
uv run amt sign g1 --brief <brief> --manifest <file> --signer "Your Name"
uv run amt sign g2 --dataset <spec-or-name> --signer "Your Name"
uv run amt sign g3 --brief <brief> --run-id ID --signer "Your Name"
```

The signer is a person's name. `auto`, `amt`, `system`, `orchestrator`, and `agent` are rejected. A report whose status is not `pass` is rejected, and the remediation is printed.

G1 writes `gates/g1/<brief-name>/g1.json`. It hashes the noise report, the brief, the dataset spec tree, and the manifest file. The brief's metric name, `min_effect`, and `budget.total_dbu` must match the report, and `min_effect` must still clear twice the R3 half-width. The manifest path has to be a local file. A signature goes stale when any of those inputs change.

G2 writes `gates/g2/<dataset>/<family-key>/g2.json`. Later runs on the same dataset and family set reuse it.

G3 writes `gates/g3/g3.json`. It signs one passing calibration packet at `gates/g3/<brief-name>/calibration_report.json`, and only when `amt status --audit` is clean for that run id. G3 is once for the whole repo. It is not repeated for the next brief.

### `gates`

```bash
uv run amt gates --brief <brief>
```

A walking-skeleton brief prints `walking skeleton; G1 and G2 are not required` and exits `0`. A search brief prints one line:

```text
G1 current; G2 current; G3 missing; full budget not ready
```

`current` means the stored signature still matches its report and inputs. `full budget ready` means G3 is signed and `gates/small/<run-id>.json` contains a passing K=4 receipt. Exit `1` means `amt run` would escalate instead of starting the search.

### `stats`

```bash
uv run amt stats --run-id ID
```

Reads `runs/<id>/state.json`, writes `runs/<id>/agent_stats.json`, and prints it. Bug rate is debug attempts divided by implementer calls. Revisions count schema failures and critic `REVISE` cycles. Tokens and wall-clock are summed from `runs/<id>/agent_calls.jsonl`. Exit `1` means there is no state file.

## Files a run leaves behind

| Path | When |
| --- | --- |
| `runs/<id>/state.json` | After every stage. |
| `runs/<id>/round_<n>.md` | After each round. Does not block the run. |
| `runs/<id>/APPROVAL.md` | While `every_stage` or `round_shortlist` is waiting. |
| `runs/<id>/ESCALATE.md` | Caps, budget refusal, a stop file, or a stale gate. |
| `runs/<id>/report.md` | Stage `REPORT`, including calibration and the small run. |
| `runs/<id>/agent_calls.jsonl` | One record per agent call, including `references_count` on planner calls. |
| `runs/<id>/agent_stats.json` | On report, and again from `amt stats`. |
| `runs/<id>/phase2_entry.md` | `amt run --scale full` after G3 and a passing small run. |
| `ledger/results.jsonl` | Append-only. Written by the orchestrator and the pruner, not by agents. |
| `gates/g1/`, `gates/g2/`, `gates/g3/`, `gates/small/` | Noise report, ladder report, calibration packet, signatures, small-run receipt. |

## Bundle jobs

`amt deploy` is the deploy step. `amt` submits every job except `pruner`, which remains a direct bundle run for replaying a saved rung file.

| Job key | Who submits it |
| --- | --- |
| `build_dataset` | `amt dataset build` |
| `train` | `amt noise`, `amt ladder`, `amt run` (search rungs), `amt reproduce` |
| `confirm` | `amt run`, after promotion. Holdouts need a one-time confirm token. |
| `final_holdout` | `amt final` |
| `hello_train` | `amt run` on a walking-skeleton brief only |
| `pruner` | `databricks bundle run pruner` when you want to replay a file by hand |
