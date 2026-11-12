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
