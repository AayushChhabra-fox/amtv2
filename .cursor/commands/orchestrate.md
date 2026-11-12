---
description: Run the AMT orchestrator on a brief
---

Run the state machine. Do not reimplement the loop in this prompt.

From the repo root:

```bash
uv run amt run briefs/toy.yaml
```

Resume is the same command. It reads `runs/<id>/state.json` and does not submit a job whose key is already recorded. Stop a run with `uv run amt stop`.

Search briefs run the draft, improve, and combine rounds. `approval_mode` is `none` unless the brief or `--approval-mode` says otherwise. `every_stage` and `round_shortlist` pause until `uv run amt approve --run-id <id>`. The round digest at `runs/<id>/round_<n>.md` does not block the run.

G3 is once: `uv run amt run briefs/<task>.yaml --scale calibration`. That forces K=1 and `approval_mode every_stage`. After the report and a clean audit, sign it with `uv run amt sign g3 --brief briefs/<task>.yaml --run-id <id> --signer "Your Name"`. The K=4 unattended run is `uv run amt run briefs/<task>.yaml --scale small`. It records bug rate, revisions, tokens, and wall-clock. `uv run amt run briefs/<task>.yaml --scale full` writes the Phase 2 entry and does not start training.
