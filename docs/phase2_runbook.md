# Operator runbook

This is the sequence for a News For You search. Commands are documented in [CLI.md](CLI.md). Two briefs are ready:

- `briefs/nfy_demo.yaml` on `datasets/nfy_article_rerank_demo` (six hours of 2026-10-07, 40 DBU, K=2). Use this for a short smoke.
- `briefs/nfy_demo_v3.yaml` on `datasets/nfy_article_rerank_gold` (2026-10-01 through 2026-10-18, 60 DBU, K=2, `max_rounds: 2`). This is the gold-scale brief that completed the two-gate flow.

Both briefs are problem statements. `min_effect` is `auto`. GPU serving and external weights are off, the license policy is commercial-only, and the serving budget is 20 ms p95 per 100 candidates. Neither brief lists model families. Prep measures the dataset and locks the viable set; the planner drafts from that profile.

The split of work:

- **`amt prep` submits:** preflight (optional `--repair`), `build_dataset`, then round-0 noise and the ladder together.
- **You sign once:** `amt sign ready`.
- **`amt run` submits:** search `train` jobs, then `confirm` only if something promotes.
- **You sign once more:** `amt final` when there is an R3 survivor to score (promoted or kept).
- **You do not** run `databricks bundle run` for any of these jobs, and you do not edit the brief to raise `min_effect` or to add families.

Run every command from the repository root. Log long stages:

```bash
mkdir -p runs/nfy-demo-v3
PYTHONUNBUFFERED=1 uv run amt prep --brief briefs/nfy_demo_v3.yaml --repair 2>&1 | tee runs/nfy-demo-v3/prep.log
```

## 0. Setup

### `.env`

Copy `.env.example` to `.env` and fill in:

- **Auth.** One of: `DATABRICKS_HOST` + `DATABRICKS_TOKEN`, OAuth client, or `DATABRICKS_CONFIG_PROFILE`. Legacy `dbx_host` / `dbx_token` still work. `amt` never picks a profile for you.
- **Token scopes.** A scoped PAT needs `files` as well as `jobs`, `workspace`, `clusters`, `mlflow`, `sql`, and `unity-catalog`.
- **`AMT_CLUSTER_ID`.** One shared all-purpose cluster for every job. **Leave blank for gold.** Concurrent round-0 matrix jobs on a shared driver OOMed during the v3 prep (`driver killed / out of memory`). Job clusters from `resources/*.job.yml` completed noise cleanly. Override per command without editing `.env`:

  ```bash
  AMT_CLUSTER_ID= uv run amt prep --brief briefs/nfy_demo_v3.yaml --repair
  AMT_CLUSTER_ID= uv run amt run briefs/nfy_demo_v3.yaml
  ```

- **`AMT_HF_OFFLINE_CACHE`.** Only when a candidate may read a HuggingFace snapshot already on a UC volume.
- **`AMT_DATASET_REPAIR_MAX_ATTEMPTS` / `AMT_DATASET_REPAIR_MAX_DBU`.** Caps for `amt prep --repair` (defaults 3 and 10).
- **Agents.** `AMT_AGENT_BACKEND=cursor` and `CURSOR_API_KEY`. Search needs a real planner; `stub` only knows the toy candidate.

Never commit `.env` or print its values.

### Deploy

```bash
uv run amt doctor
uv run amt deploy
```

`amt prep` deploys again when `harness/` or `resources/` is newer than `.amt/deployed.json`. A submit is refused with `redeploy: harness changed since deploy` until the hashes match.

On a fixed cluster, entry scripts put the bundle root on `sys.path` before importing `amt`. Job-cluster runs install the wheel from the bundle.

Leave `spark.sql.ansi.enabled` at its default (on).

### Windows

| Brief | Spec | Splits (UTC) |
| --- | --- | --- |
| `nfy_demo` | demo | 2026-10-07 12:00–18:00 in four one/three-hour blocks |
| `nfy_demo_v3` | gold | whole-day train / search_val / select_holdout / final_holdout from 2026-10-01 through 2026-10-18 |

Signatures do not carry across briefs. Each ready packet hashes that brief’s problem, task, constraints, budget, and spec.

## 1. Prep

```bash
PYTHONUNBUFFERED=1 AMT_CLUSTER_ID= uv run amt prep --brief briefs/nfy_demo_v3.yaml --repair \
  2>&1 | tee runs/nfy-demo-v3/prep.log
```

Order:

1. `doctor`, then `deploy` if the harness hash is stale.
2. Local checks (gold needs five distinct `search_val` days; demo allows one) and the 1-hour preflight slice.
3. With `--repair`, a cast or analysis error in `features/*.sql` is fixed and the slice reruns. Labels, splits, leakage, and harness bugs escalate with a diagnosis.
4. Full Spark build. Manifest lands at `gates/ready/nfy_demo_v3/manifest.json`.
5. Round-0 noise and the ladder submit together, then collect.

On the completed gold prep:

- Manifest: ~1.87M rows, ~100k queries, 72 numeric + 6 categorical features, no embeddings.
- Ladder **pass** (lgbm, neural, linear_reweight; embedding skipped).
- Noise **pass** after a job-cluster resubmit; resolved `min_effect` **0.019578** (`max(0.010, 2× R3 half-width)`).
- Packet: `gates/ready/nfy_demo_v3/ready_report.md` with status `pass`.

Exit `0` and status `pass` before you sign. Exit `2` means the packet is `fail` or `incomplete` (read the remediation). Exit `1` means prep stopped earlier (auth, non-repairable preflight, failed build).

### If noise is incomplete after prep

Prep can finish the build and ladder while round-0 matrix jobs die on a shared cluster. The report then says something like “39 round-0 cells are missing.” Resubmit noise alone on job clusters, then rebuild the ready packet:

```bash
AMT_CLUSTER_ID= uv run amt noise --brief briefs/nfy_demo_v3.yaml --submit --collect
# when status is pass, either re-run prep (build is reused if the tree hash matches)
# or rebuild the ready report from the existing noise + ladder + manifest
```

Failed run ids in `runs/round0-nfy_demo_v3/state.json` are not reused; submit writes new ones with a retry token.

## 2. Sign the ready gate

```bash
uv run amt sign ready --brief briefs/nfy_demo_v3.yaml --signer "Your Name"
uv run amt gates --brief briefs/nfy_demo_v3.yaml
```

`sign ready` writes `gates/ready/nfy_demo_v3/ready.json`. Commit the whole `gates/ready/nfy_demo_v3/` packet. The signature locks the problem and task, the spec hash, the manifest hash, repair hashes, the resolved `min_effect`, the 60 DBU budget, and the viable families.

`gates` should print `ready current`. Editing the brief’s budget, constraints, or task, or the spec or manifest, makes it stale. Prep and sign again. Do not hand-edit `ready.json`.

## 3. Search

```bash
PYTHONUNBUFFERED=1 AMT_CLUSTER_ID= uv run amt run briefs/nfy_demo_v3.yaml \
  2>&1 | tee -a runs/nfy-demo-v3/run.log
```

Run id is `nfy-demo-v3` from the brief unless you pass `--run-id`. Progress lines go to stderr as `[HH:MM:SS] [ROUND r | STAGE]`. In another terminal:

```bash
uv run amt status --run-id nfy-demo-v3
```

What the loop does:

- Materializes `lgbm_reference` as `runs/<id>/candidates/c00/` from round-0.
- Round 1 drafts from the modality profile and brief constraints. Round 2 improves (`max_rounds: 2` stops before combine).
- A partial critic `REVISE` keeps the PASS candidates.
- An R1 exception goes through `DEBUG` (up to three times).
- A cluster `INTERNAL_ERROR` is retried once, then escalates with the node still ready for `amt resume`.
- Promotion is symmetric: beat production by the resolved `min_effect` with `ci_lo > 0`, stay inside latency, and beat `c00` (`delta_vs_lgbm.ci_lo > 0`). If nothing beats `c00` but `c00` clears the production bar, `c00` is confirmed.

Printed outcomes (exit 0):

```text
OUTCOME: winner_promoted
OUTCOME: baseline_c00_promoted
OUTCOME: no_winner
```

The gold v3 search ended `OUTCOME: no_winner`: best kept R3 was an lgbm improve slightly below production (Δ≈−0.001). That is a finished result, not a crash. `runs/nfy-demo-v3/report.json` carries the same `outcome`.

Exit `1` means `ESCALATE`. Read `runs/<id>/ESCALATE.md`.

| Reason | What to do |
| --- | --- |
| `refusing submit` | Next job would pass the signed DBU pool. Edit the brief budget, prep, sign, new run id. |
| `infrastructure failure: infra_crash` | `amt resume --run-id <id>` submits again. Prefer job clusters on gold. |
| `debug cap` / `critic revision cap` / `outer retry cap` | Read the node logs or treat as a finished poor search. |
| ready gate missing or stale | `amt prep`, `amt sign ready`, then `amt run --fresh` with a new run id. |

```bash
uv run amt status --run-id nfy-demo-v3 --audit
```

## 4. Final holdout

```bash
AMT_CLUSTER_ID= uv run amt final --run-id nfy-demo-v3 --signer "Your Name"
```

Scores the top R3 survivors (kept or promoted), plus production and the LightGBM reference, once. Writes `runs/<id>/final_holdout.md`. A second call is refused.

On the v3 `no_winner` run, `c03` was still a kept R3 row, so final opened and scored `c03`, production, and `lgbm_reference`. If there is no R3 survivor at all, final exits `1` and the holdout stays closed. Do not loosen `min_effect` after sign-off.

```bash
uv run amt reproduce --run-id nfy-demo-v3 --node <winner>
```

## Optional calibration

Calibration is a diagnostic, not a gate in front of search. Prefer the short demo brief:

```bash
uv run amt run briefs/nfy_demo.yaml --scale calibration --run-id nfy-calibration
uv run amt approve --run-id nfy-calibration
uv run amt resume --run-id nfy-calibration
```

Repeat approve/resume until `REPORT`, then `amt sign g3`. G3 is once for the repo.

## If a stage fails

- **`required scopes: files`:** add the scope; wait about a minute.
- **`redeploy: harness changed since deploy`:** `amt deploy`, or let the next prep deploy.
- **`ModuleNotFoundError: amt` on a fixed-cluster job:** redeploy after a harness change; entry scripts must set `sys.path` before importing `amt`.
- **`Retrieving the output of runs with multiple tasks is not supported`:** fixed in `amt/batch.py` (get-output uses the task run id). Redeploy is not required for that client-side fix.
- **Round-0 / train driver OOM on `AMT_CLUSTER_ID`:** clear the env for that command and resubmit (`amt noise --submit --collect`, or `amt resume`).
- **Preflight cast/analysis in `features/*.sql`:** `amt prep --repair`. Inspect `runs/dataset-repair/<spec>/<n>/diff.patch` before signing.
- **Preflight labels / splits / leakage:** fix the spec or source data; do not patch around it inside features.
- **Ready `incomplete` / `revise` / `fail`:** follow `ready_report.md`. Incomplete noise is the usual gold failure on a shared cluster.
- **Stale ready signature:** `amt gates` says so; prep and sign again.

## What you should have at the end

For the gold brief after a full pass:

- `gates/ready/nfy_demo_v3/ready_report.md` with status `pass`, and `gates/ready/nfy_demo_v3/ready.json` committed (with the rest of that directory).
- `runs/nfy-demo-v3/report.json` with `outcome` one of `winner_promoted`, `baseline_c00_promoted`, or `no_winner`.
- `runs/nfy-demo-v3/final_holdout.md` when an R3 survivor was scored.
- `uv run amt gates --brief briefs/nfy_demo_v3.yaml` printing `ready current`.

`gates/ready/` is the only gate directory to commit.
