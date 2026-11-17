# AMT Architecture (Zero Knowledge)

This document explains **amtv2** (Automated Model Trainer) from first principles: what the pieces are, how they fit together, what runs on Databricks, and what stays on your laptop.

If you only remember one sentence: **AMT is a local control plane that uses Cursor agents to invent ranking models, while Databricks jobs are the only place that train, score, and produce trusted metrics.**

For how to operate a run, see [CLI.md](CLI.md) and [phase2_runbook.md](phase2_runbook.md). For the original design rationale, see [AMT_MVP_Plan.md](AMT_MVP_Plan.md).

---

## 1. What problem does AMT solve?

Offline search for a better **ranking model**—today, the News For You (NFY) article reranker—scored primarily by **graded NDCG@10**.

Without AMT, an LLM agent can:

- rewrite metrics or peek at holdouts
- invent its own workflow and quit early
- “win” on noisy validation that does not generalize

AMT’s answer is a **code-enforced workflow**:

| Pillar | Meaning |
| --- | --- |
| Workflow as code | A Python state machine (`amt/orchestrator.py`) decides the next stage. Agents are stateless: they read files and write files. |
| Three-tier data | Agents only ever see aggregate scores from `search_val`. A private `select_holdout` confirms finalists. A `final_holdout` opens once, with a human signature. |
| Locked harness | Agents never edit `harness/` or metrics. They only write candidate code under `runs/<run_id>/candidates/<node_id>/`. |
| Noise-aware promotion | Promotion uses day-blocked paired bootstrap CIs against production and against a LightGBM reference (`c00`), not a raw score bump. |
| Human signatures | `amt sign ready` locks the problem before search. `amt final` opens the final holdout once. |

The first production surface is NFY article rerank. Genericity of the data layer is proven with a second surface (`datasets/nfy_clip_rerank/`) on the same builder—no harness forks.

---

## 2. Glossary (read this first)

| Term | Definition |
| --- | --- |
| **Brief** | A YAML problem statement under `briefs/`. Names the dataset spec, budget (DBUs), constraints (GPU, latency, license), and search knobs. It is a *problem*, not a model. |
| **Dataset / dataset spec** | A folder under `datasets/<name>/` with `dataset.yaml` plus SQL fragments. It describes *how* to build ranking frames from the warehouse—not a raw dump of tables. |
| **Materialization** | Running the Spark builder so the spec becomes concrete Delta tables, Parquet caches on a Unity Catalog volume, and a `manifest.json`. |
| **Manifest** | Fingerprint of a built dataset: spec hash, split sizes, content hashes, checks, modality profile. Prep copies it into the ready packet. |
| **Harness** | Locked train/eval/data code under `harness/`. Ships in the Databricks wheel. Agents may *read* the interface; they may not *edit* it. |
| **Candidate / node** | One searchable design (id like `c01`). Lives at `runs/<run_id>/candidates/<node_id>/` with `model.py` implementing `Ranker`. |
| **`c00`** | Reserved id for the materialized LightGBM reference baseline (`lgbm_reference`). Not proposed by the planner. |
| **Ranker** | The protocol every candidate must implement: `fit`, `score`, `save`, `load` (`harness/interface.py`). |
| **Family** | Model paradigm label (`lgbm`, `neural`, `linear_reweight`, …). Constrained by modality + brief. |
| **Operator** | How a node was created: `draft`, `improve`, `debug`, `ablate`, `combine`. |
| **Rung** | Fidelity level: `R0` (smoke) → `R1` → `R2` → `R3` (fuller data / longer deadline). |
| **Ledger** | Append-only archive `ledger/results.jsonl`. Written only by `amt/` code (pruner/orchestrator), never by agents. |
| **Pruner** | Deterministic kill/keep/promote rules. Never reads holdout metrics during search. |
| **Ready gate** | Human-signed packet under `gates/ready/<brief>/` locking hashes, `min_effect`, DBU budget, viable families. |
| **Outcome** | End of search: `winner_promoted`, `baseline_c00_promoted`, or `no_winner`. |
| **DBU** | Databricks billing unit. Search spend is capped from the signed ready packet. |

---

## 3. Mental model

```text
                    ┌─────────────────────────────────────┐
                    │  Brief (YAML) + Dataset spec (SQL)  │
                    └─────────────────┬───────────────────┘
                                      │
                         amt prep (local orchestration)
                                      │
              ┌───────────────────────┼───────────────────────┐
              ▼                       ▼                       ▼
     dataset preflight        Databricks                round-0 noise
     (+ optional fixer)       build_dataset             + ladder jobs
                                      │
                                      ▼
                         gates/ready/<brief>/ packet
                                      │
                         human: amt sign ready
                                      │
                         ┌────────────┴────────────┐
                         ▼                         │
              amt run / orchestrator (local)       │
                         │                         │
         Cursor agents invent code on disk         │
                         │                         │
              upload candidate → UC volume         │
                         │                         │
                         ▼                         │
              Databricks train / confirm jobs      │
                         │                         │
                         ▼                         │
              metrics → MLflow + pruner → ledger ──┘
                                      │
                         human: amt final (once)
                                      │
                         Databricks final_holdout
```

**Agents invent. Databricks measures. The orchestrator decides. Humans unlock the gates.**

---

## 4. What is a dataset here?

A dataset is a **spec**, not “whatever is in the warehouse.”

### 4.1 Folder layout

```text
datasets/<name>/
  dataset.yaml          # sources, features, refs, splits, sampling, checks
  queries.sql           # one row per query (query_id, query_ts, …)
  candidates.sql        # (query_id, item_id) pairs
  labels.sql            # graded label + label_window; optional aux labels
  features/*.sql        # feature groups (join keys + available_at for PIT)
  checks/*.sql          # optional expected-metric SQL (e.g. prod NDCG)
```

Examples: `nfy_article_rerank_gold`, `nfy_article_rerank_demo`, `nfy_clip_rerank`, `toy` (local fixture only).

### 4.2 What the spec declares

From `dataset.yaml` / `harness/data/contract.py`:

| Concept | Role |
| --- | --- |
| **`task_type`** | MVP: `ranking` |
| **`sources`** | Unity Catalog tables + partition filters (`{start}` / `{end}`). Optional `materialize: true` + column projection so shared sources are scanned once into staging. |
| **`feature_groups`** | SQL fragments keyed by `query_id` and/or `item_id`, with `available_at ≤ query_ts` (point-in-time / no future leakage). |
| **`reference_scores`** | e.g. production rank → physical column `ref__production`. Used by baselines; often forbidden as model features. |
| **`forbidden_columns`** | Leakage / post-request / position fields agents must not use. |
| **`split`** | Time windows on `query_ts` with gaps for `train`, `search_val`, `select_holdout`, `final_holdout`. |
| **`sampling`** | Smoke sizes and per-rung `train_fraction` / `eval_queries`. |
| **`checks`** | e.g. production NDCG reproduces within tolerance. |

### 4.3 Output contract (what every built split looks like)

Every surface produces the **same physical schema**. Harness, eval, baselines, and pruner speak only this contract:

| Logical | Physical (Delta-safe) |
| --- | --- |
| core ids / time / label | `query_id`, `item_id`, `query_ts`, `event_day`, `label` |
| aux labels | `aux_label__*` |
| features | `feat__<group>__*` |
| reference scores | `ref__*` |

Dots become `__` because Delta treats dots specially.

### 4.4 Three data tiers (why splits matter)

| Split | Who may use it | Purpose |
| --- | --- | --- |
| `train` + `search_val` | Train jobs during search; agents see **aggregate** ledger metrics only | Iterate and prune |
| `select_holdout` | Confirmation job only (orchestrator-triggered) | Private selection of finalists |
| `final_holdout` | `amt final` once | Human-gated last score |

Holdout frames are never passed to candidate code or agents. Static preflight rejects Spark I/O, warehouse table names, network calls, and out-of-directory `open()`.

---

## 5. What does materialization look like?

**Materialization** = turn a dataset spec into durable, hashed training artifacts on Databricks.

### 5.1 End-to-end

```text
brief.dataset_spec → datasets/<name>/
        │
        ▼
[local] amt dataset preflight / amt prep
  · contract / split / metric sanity
  · ~90s 1-hour Spark slice (EXPLAIN, PIT, labels)
  · optional dataset_fixer on features/*.sql only
        │
        ▼
[Databricks] harness.data.build_dataset
  1. Partition-filter sources; optional source materialize
  2. Render SQL fragments for the full time span
  3. Stage fragment tables; scan for forbidden columns
  4. Enforce PIT / label_window / contract
  5. Assemble → filter per split window → Delta tables
  6. Write Volume Parquet per split (fast train I/O)
  7. Run checks (e.g. prod NDCG)
  8. Draw rung sample query sets; write manifest (+ modality)
        │
        ▼
[local] prep downloads manifest → gates/ready/<brief>/manifest.json
```

### 5.2 Where the bytes live

| Artifact | Location (defaults) |
| --- | --- |
| Delta split tables | `{catalog}.{schema}.{dataset}_{split}` (e.g. `product_api_dev.amt.*`) |
| Staging during build | `…_stage_*` |
| Parquet caches | `/Volumes/<catalog>/<schema>/amt_artifacts/<spec>/<spec_hash>/splits/<split>.parquet` |
| Manifest | Volume + table `…_manifest`; copied into ready packet |

Train jobs prefer volume Parquet; Delta is the warehouse source of truth.

### 5.3 Source `materialize: true` vs full build

Two different meanings of “materialize”:

1. **Source materialize** (inside the builder): copy a filtered, projected warehouse table into a staging table once so many SQL fragments do not rescan a huge gold table.
2. **Dataset materialization** (the build job): assemble the full ranking frames, splits, samples, and manifest.

When people say “materialize the dataset,” they usually mean (2).

### 5.4 Repair

`amt prep --repair` (capped by `AMT_DATASET_REPAIR_MAX_ATTEMPTS` / `…_MAX_DBU`) can hand feature SQL failures to **dataset_fixer**, which may edit only `datasets/<spec>/features/*.sql`. Failures in labels, splits, leakage, or the harness do **not** auto-repair. Diffs live under `runs/dataset-repair/<spec>/<n>/`.

---

## 6. What is a model here?

In AMT, “model” almost always means a **candidate Ranker**, not an MLflow registry entry in the abstract.

### 6.1 Candidate directory

```text
runs/<run_id>/candidates/<node_id>/
  candidate.yaml / fields from the planner claim
  spec.md              # acceptance criteria / claimed change
  model.py             # must define class Ranker
  requirements.txt     # optional exact PyPI pins; harness installs
  debug_log.md         # fix chain for debug operator
  review.md            # VERDICT: PASS | FAIL
```

Write scope is enforced by `.cursor/hooks` → implementer may only touch that directory.

### 6.2 Ranker protocol

```python
class Ranker(Protocol):
    def fit(self, train: RankingFrame, valid: RankingFrame, ctx: FitContext) -> None: ...
    def score(self, frame: RankingFrame) -> np.ndarray: ...
    def save(self, path: str) -> None: ...
    @classmethod
    def load(cls, path: str) -> Ranker: ...
```

Rules that matter architecturally:

- `fit` / `score` may use **only** the frames passed in.
- `score` must not read labels (labels are stripped before score).
- No Spark, warehouse tables, subprocess, sockets, or hub downloads unless explicitly allowed via offline HF volume.
- Evaluation metric names cannot be logged from `fit` (banned tokens include `ndcg`, `holdout`, …).

### 6.3 Training and evaluation (harness)

| Piece | Role |
| --- | --- |
| `harness/train.py` | Databricks job entry: load rung frames, install deps, `fit` under wall deadlines, score allowed split, log MLflow |
| `harness/evaluate.py` | Primary metric `ndcg@10_graded` (gains \(2^r-1\)); also click NDCG, MRR, coverage; **paired day-blocked bootstrap** deltas |
| `harness/baselines/` | `reference_score`, `popularity`, `random`, `lgbm_reference` |
| `harness/mlflow_model.py` | At R3, log pyfunc (+ optional native) model for promotion/serve paths |
| `harness/latency.py` | Fixed p95 latency benchmark (score 100 candidates, single-threaded CPU) |

Search rungs score **`search_val`**. Confirm uses **`select_holdout`**. Final uses **`final_holdout`**. Tokens gate which split a job may open.

### 6.4 Rungs and search operators

| Rung | Intent |
| --- | --- |
| R0 | Smoke / round-0 noise & ladder |
| R1–R2 | Cascade: cheaper then fuller fidelity |
| R3 | Full promotion-grade eval |

| Operator | Intent |
| --- | --- |
| `draft` | New design from a viable family |
| `improve` | Exactly one atomic change to a parent |
| `debug` | Fix a runtime failure (depth capped) |
| `ablate` | Remove one component to locate gain |
| `combine` | Blend / stack top lineages (late round) |

Round schedule (high level): drafts → improve survivors → optional fork-on-plateau → combine. See `amt/search_policy.py`.

---

## 7. What does AMT (the control plane) do?

`amt/` is the **control plane**. It never trains models itself. It:

1. Prepares and signs readiness
2. Calls agents to propose and implement candidates
3. Submits / waits on Databricks jobs
4. Prunes with deterministic rules
5. Writes the ledger and reports outcomes
6. Opens confirmation and final holdout under policy

### 7.1 Operator lifecycle (commands)

```text
amt doctor / amt deploy
        │
        ▼
amt prep --brief … [--repair]     # build data + noise + ladder → ready packet
        │
        ▼
amt sign ready --signer "…"       # human locks the packet
        │
        ▼
amt run <brief> / amt resume      # search until OUTCOME:*
        │
        ▼
amt final --run-id … --signer     # open final_holdout once
```

Supporting commands: `dataset {preflight,build,revert}`, `noise`, `ladder`, `gates`, `status`, `stop`, `approve`, `reproduce`, `deploy`.

### 7.2 Orchestrator stages

Persisted in `runs/<run_id>/state.json`. Rough loop per round:

```text
PLAN → CRITIQUE → SPEC → IMPLEMENT → PREFLIGHT
     → SUBMIT → WAIT → (DEBUG?) → PRUNE
     → CONFIRM (if promoted) → REPORT
```

Caps exist for debug depth, outer retries, critic revisions, and the DBU pool from the signed ready packet. Escalation writes `ESCALATE.md` and pauses for a human.

### 7.3 Agents (roles)

Defined in `.cursor/agents/*.md`, invoked via `amt/agents.py` (`cursor-agent -p` when `AMT_AGENT_BACKEND=cursor`).

| Role | Mode | Writes |
| --- | --- | --- |
| **planner** | read-only (+ web search) | proposals JSON only |
| **critic** | read-only | PASS / REVISE / DROP verdicts |
| **implementer** | write | `runs/.../candidates/<node>/` only |
| **reviewer** | write | `review.md` only |
| **dataset_fixer** | write | `datasets/<spec>/features/*.sql` only |

Context is scoped (`amt/context.py`): e.g. 12 newest ledger lines + best per family—not the whole archive. Agents never write the ledger.

### 7.4 Pruner and promotion

`amt/pruner.py` is deterministic:

- Hard kills: missing/NaN metric, latency (or `latency_suspect` on shared clusters), DBU overspend, preflight/review fail, infra vs runtime classification
- Floor vs **`c00`** (tightens as rungs rise); dominance and halving at higher rungs; diversity / family slots
- **R3 promotion (symmetric):** beat production by locked `min_effect` with `ci_lo > 0`, latency OK, **and** `delta_vs_lgbm.ci_lo > 0`; else promote `c00` if it clears production; else empty → `no_winner`

The pruner does **not** read holdout metrics during search.

### 7.5 Prep: noise floor and ladder

Before search is allowed:

| Check | Purpose |
| --- | --- |
| **Noise** (`amt/noise.py`) | Round-0 baseline matrix → CI half-widths. `min_effect: auto` resolves to `max(0.010, 2× R3 half-width)`. |
| **Ladder** (`amt/ladder.py`) | Known-good vs known-bad designs per family; Kendall τ(R1, R3) threshold; unsupported families listed. |
| **Modality** | Feature profile → **viable_families** locked into the ready packet. |

### 7.6 Gates

| Gate | Command | Locks |
| --- | --- | --- |
| Ready | `amt sign ready` | problem/task hash, spec hash, manifest hash, repairs, `min_effect`, DBU budget, viable families |
| Final | `amt final` | one-shot final holdout |

Walking-skeleton / toy briefs with a pinned `candidate:` can skip ready. A search brief without a current `ready.json` escalates.

---

## 8. What happens on Databricks vs what does not

### 8.1 Runs on your laptop / Cursor

| Concern | Where |
| --- | --- |
| CLI and state machine | `uv run amt …`, `amt/cli.py`, `amt/orchestrator.py` |
| Agent calls | `cursor-agent` via `amt/agents.py` |
| Briefs, specs, candidate source | git tree; candidates under `runs/` |
| Ready signatures and reports | `gates/ready/` |
| Ledger, run state, digests | `ledger/`, `runs/<run_id>/` |
| Deploy fingerprint | `amt/deploy.py` → `.amt/deployed.json` |
| Volume put/get, job submit/wait | Databricks CLI / Jobs API from the laptop (`amt/volume.py`, `amt/jobs_backend.py`) |
| Fake/test mode | `AMT_JOBS_BACKEND=fake`, stub agents |

### 8.2 Runs on Databricks

| Job | Entry | Purpose |
| --- | --- | --- |
| `build_dataset` | `harness.data.build_dataset` | Spark assemble → Delta + Parquet + manifest |
| `train` | `harness.train` | Fit/score search rungs; round-0 noise/ladder cells |
| `confirm` | train path + confirm token | `select_holdout`, multi-seed |
| `final_holdout` | train path + final token | One-shot final scoring |
| `pruner` | bundle job (manual replay) | Usually orchestrator calls `decide` **in-process** locally |

Compute comes from job clusters in `resources/*.job.yml`, or a fixed `AMT_CLUSTER_ID` (gold-scale prep prefers job clusters—shared drivers can OOM). The wheel is built with `uv build` and deployed via the bundle (`databricks.yml`). UC catalog/schema/volume and MLflow experiment root are configured via env (`AMT_CATALOG`, `AMT_SCHEMA`, `AMT_ARTIFACT_VOLUME`, `AMT_MLFLOW_EXPERIMENT_ROOT`).

### 8.3 The hard boundary

| Allowed | Forbidden |
| --- | --- |
| Agents write candidate `model.py` locally | Agents edit `harness/`, metrics, or briefs during search |
| Orchestrator uploads a git-SHA snapshot of the candidate to the volume | Candidate code opens holdout tables or warehouse paths |
| Databricks jobs produce metrics and model weights | Agents invent “metrics” from logs |
| Ledger written by pruner/orchestrator | Agents append ledger rows |
| Humans sign ready / final | Search without a current ready packet (escalates) |

**All real metrics and weights come from Databricks jobs** reading the locked harness + snapped candidate code. Holdout frames never go to agents.

---

## 9. Repository map

| Path | Role |
| --- | --- |
| **`amt/`** | Control plane: CLI, orchestrator, agents wrapper, gates, pruner, budget, prep, deploy, jobs, ledger I/O |
| **`harness/`** | Locked train/eval/data/baselines; shipped in the wheel |
| **`datasets/`** | Specs + SQL; one folder per surface |
| **`briefs/`** | Human problem YAML |
| **`runs/`** | Per-run state, candidates, digests, escalate/approval, repair, round-0/ladder job state |
| **`ledger/`** | Append-only `results.jsonl` |
| **`gates/ready/`** | Signed readiness packets (intended to be committed) |
| **`docs/`** | CLI, runbook, MVP plan, this architecture doc |
| **`.cursor/agents/`** | Role prompts |
| **`.cursor/hooks*`** | Path sandbox for implementer / fixer |
| **`resources/`** | Databricks job YAML |
| **`databricks.yml`** | Asset bundle |
| **`mlruns/`** | Local MLflow residue if any; workspace MLflow is authoritative for jobs |

---

## 10. Key artifacts (audit trail)

| Artifact | Location | Producer |
| --- | --- | --- |
| Ready packet | `gates/ready/<brief>/ready_report.*`, `ready.json`, `manifest.json`, noise/ladder reports | prep + human sign |
| Run state | `runs/<id>/state.json` | orchestrator |
| Candidates | `runs/<id>/candidates/<node>/` | implementer; `c00` materialized |
| Round digest | `runs/<id>/round_<n>.md` | orchestrator |
| Outcome report | `runs/<id>/report.json` / `.md` | REPORT stage |
| Escalation / approval | `ESCALATE.md`, `APPROVAL.md`, `STOP`, `APPROVE` | human + orchestrator |
| Ledger | `ledger/results.jsonl` | pruner / orchestrator |
| Candidate snapshot | Volume `…/candidates/<run>/<node>/<git_sha>/` | `amt/snapshot.py` |
| Split caches | Volume `…/<spec>/<hash>/splits/*.parquet` | build_dataset |
| Delta splits | UC tables `<dataset>_<split>` | build_dataset |
| MLflow run | experiment `<root>/<brief>` | train / confirm / final |
| Logged model | `runs:/<mlflow_run_id>/model` (+ `native_model`) | R3 / lgbm_reference |
| Deploy fingerprint | `.amt/deployed.json` | `amt deploy` |

`amt status --audit` expects each ledger row to carry a job run id, MLflow run id, and real `git_sha`.

---

## 11. Worked example (NFY gold)

A concrete path you will see in the runbook:

1. Brief: `briefs/nfy_demo_v3.yaml` → dataset `datasets/nfy_article_rerank_gold`
2. Spec reads `product_api_dev.news_for_you.gold_nfy_reranker_training` with `materialize: true` and a projected column list
3. `amt prep` builds ~multi-day splits (train / search_val / select_holdout / final_holdout), writes volume Parquet + manifest, runs noise + ladder
4. You sign ready → locked `min_effect`, DBU pool, viable families (e.g. lgbm / neural / linear_reweight)
5. `amt run` drafts diverse families, implements under `runs/<id>/candidates/`, trains on Databricks, prunes, may confirm on `select_holdout`
6. Outcome printed; optional `amt final` scores `final_holdout` once

Demo brief (`nfy_demo`) is the short smoke window on `nfy_article_rerank_demo`. Same architecture, smaller data and budget.

---

## 12. Design invariants (for readers and contributors)

1. **Agents are not the workflow.** The orchestrator is. If an agent ignores a prompt, the state machine still holds.
2. **Evaluation hygiene beats search cleverness.** Private holdouts, locked metrics, paired noise-aware promotion.
3. **Nodes are immutable lineages.** Improve/debug create children; do not silently rewrite a scored parent in place.
4. **One source of truth for search decisions:** the ledger + pruner. MLflow stores runs; it does not decide promotion.
5. **Budget and latency live outside candidate code.** Caps are env/brief/ready-packet; jobs enforce wall-clock.
6. **Databricks is the measurement plane.** Local code proposes and coordinates; trusted numbers come from jobs.

---

## 13. Where to go next

| Goal | Doc / entry |
| --- | --- |
| Run the system | [phase2_runbook.md](phase2_runbook.md) |
| Command reference | [CLI.md](CLI.md) |
| Why it was built this way | [AMT_MVP_Plan.md](AMT_MVP_Plan.md) |
| Change a feature | `datasets/<spec>/features/*.sql` (+ prep/repair) |
| Change search policy | `amt/search_policy.py`, `amt/pruner.py` (not agent prompts alone) |
| Change metrics | `harness/evaluate.py` only—and treat it as a harness change requiring redeploy + new ready signature |
