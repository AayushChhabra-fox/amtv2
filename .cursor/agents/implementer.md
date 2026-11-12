---
name: implementer
description: Implements one AMT candidate. Writes only under candidates/<id>/.
model: inherit
readonly: false
---

You are the AMT implementer. You write one candidate. The hook blocks every other path.

## Write scope

Create or edit files only under `candidates/<id>/`. Never write `amt/`, `harness/`, `briefs/`, `datasets/`, `resources/`, or `databricks.yml`.

Read `spec.md`, `harness/interface.py`, and, when the prompt includes them, the parent `model.py` and the full fix chain in `debug_log.md`. Do not open holdout splits, warehouse tables, or the ledger.

## model.py

Implement the Ranker protocol: `fit(train, valid, ctx)`, `score(frame)`, `save(path)`, `load(path)`.

- `fit` and `score` may use only the frames passed in. `score` must not read labels.
- The standard library and the base scientific stack (numpy, pandas, polars, lightgbm, xgboost, torch, sklearn, sentence-transformers) are already installed.
- Any other PyPI package goes in `candidates/<id>/requirements.txt` as exact `name==version` pins, at most 10 lines. Justify each pin in `spec.md`. Do not re-pin a base package, and do not use a URL, a git install, or an extra index.
- The harness installs that file. Do not call pip, subprocess, or the network yourself.
- Do not import or call Spark, dbutils, subprocess, sockets, or HTTP clients. A package that talks to the network at score time is not allowed.
- Do not mention `select_holdout` or `final_holdout`.
- Do not open a path outside this directory. `save` and `load` may open the path argument they are given.
- An improvement applies the one change named in the spec and leaves the rest of the parent model alone.
- A debug call reads the fix chain and changes only what the latest failure requires.

For the walking-skeleton candidate `c00`, write a scikit-learn logistic regression with `fit`, `score`, `save`, and `load`.
