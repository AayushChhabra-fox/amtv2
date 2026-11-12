"""Agent calls.

`cursor-agent` (installed as `agent` / `cursor-agent`, CLI 2026.11.11) is the
live backend. Flags confirmed from `cursor-agent --help`:

- `-p` / `--print` non-interactive, with write and shell tools
- `--output-format json|text|stream-json` (only with `--print`)
- `--model <model>`
- `--mode plan|ask` for read-only roles
- `--force` to allow commands without prompts
- `--trust` to accept the workspace
- `--workspace <path>`
- auth: `cursor-agent login`, or `CURSOR_API_KEY` / `--api-key`
- `cursor-agent status` reports whether the CLI is logged in

The toy brief defaults to the stub backend so `amt run` finishes unattended
when the CLI has no login. Set `AMT_AGENT_BACKEND=cursor` to require the CLI.
`AMT_AGENT_BACKEND=auto` tries the CLI and records a failure if it is logged out.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import time
from pathlib import Path

import yaml

from amt.context import build_context
from amt.schemas import Brief, BriefCandidate, Candidate, Verdict

MODEL_PY = '''\
"""Logistic regression ranker for the Phase 0 toy candidate."""

from __future__ import annotations

import pickle

import numpy as np
from sklearn.linear_model import LogisticRegression


class LogisticRanker:
    def __init__(self) -> None:
        self._model = LogisticRegression(max_iter=300, random_state=0)

    def fit(self, train: dict, valid: dict, ctx: dict | None = None) -> None:
        del valid, ctx
        self._model.fit(np.asarray(train["X"]), np.asarray(train["y"]))

    def score(self, frame: dict) -> np.ndarray:
        return self._model.predict_proba(np.asarray(frame["X"]))[:, 1]

    def save(self, path: str) -> None:
        with open(path, "wb") as handle:
            pickle.dump(self._model, handle)

    @classmethod
    def load(cls, path: str) -> LogisticRanker:
        ranker = cls()
        with open(path, "rb") as handle:
            ranker._model = pickle.load(handle)
        return ranker
'''


def _stats_path(root: Path, run_id: str) -> Path:
    path = root / "runs" / run_id / "agent_calls.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def _log_call(root: Path, run_id: str, record: dict) -> None:
    with _stats_path(root, run_id).open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record) + "\n")


def cursor_agent_bin() -> str | None:
    return shutil.which("cursor-agent") or shutil.which("agent")


def cursor_agent_logged_in() -> bool:
    binary = cursor_agent_bin()
    if binary is None:
        return False
    completed = subprocess.run(
        [binary, "status"],
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
    )
    text = f"{completed.stdout}\n{completed.stderr}".lower()
    if "not logged in" in text or "authentication required" in text:
        return False
    return completed.returncode == 0 and "logged in" in text


def _backend() -> str:
    return os.environ.get("AMT_AGENT_BACKEND", "stub").strip().lower()


def run_role(role: str, brief: Brief, root: Path, run_id: str) -> dict:
    started = time.perf_counter()
    mode = _backend()
    if mode == "auto":
        mode = "cursor" if cursor_agent_logged_in() else "stub"
        if mode == "stub":
            _log_call(
                root,
                run_id,
                {
                    "role": role,
                    "backend": "cursor",
                    "ok": False,
                    "latency_s": round(time.perf_counter() - started, 4),
                    "tokens": None,
                    "error": "cursor-agent is not logged in; toy loop used the stub",
                },
            )
            started = time.perf_counter()
    if mode == "cursor":
        try:
            output = _run_cursor(role, brief, root)
            _log_call(
                root,
                run_id,
                {
                    "role": role,
                    "backend": "cursor",
                    "ok": True,
                    "latency_s": round(time.perf_counter() - started, 4),
                    "tokens": _token_count(output),
                },
            )
            return output
        except Exception as exc:
            _log_call(
                root,
                run_id,
                {
                    "role": role,
                    "backend": "cursor",
                    "ok": False,
                    "latency_s": round(time.perf_counter() - started, 4),
                    "tokens": None,
                    "error": str(exc),
                },
            )
            raise
    output = _run_stub(role, brief, root, run_id)
    _log_call(
        root,
        run_id,
        {
            "role": role,
            "backend": "stub",
            "ok": True,
            "latency_s": round(time.perf_counter() - started, 4),
            "tokens": 0,
        },
    )
    return output


def _token_count(output: dict) -> int | None:
    usage = output.get("usage") or {}
    total = usage.get("total_tokens") or usage.get("output_tokens")
    return int(total) if total is not None else None


def _run_cursor(role: str, brief: Brief, root: Path) -> dict:
    binary = cursor_agent_bin()
    if binary is None:
        raise RuntimeError("cursor-agent is not on PATH")
    prompt = build_context(role, brief, root)
    command = [
        binary,
        "-p",
        "--output-format",
        "json",
        "--trust",
        "--force",
        "--workspace",
        str(root),
    ]
    if role in {"planner", "critic"}:
        command.extend(["--mode", "plan"])
    if role == "implementer":
        _set_active_role(root, "implementer")
    command.append(prompt)
    try:
        completed = subprocess.run(
            command,
            check=False,
            capture_output=True,
            text=True,
            timeout=int(os.environ.get("AMT_AGENT_TIMEOUT_S", "600")),
            cwd=root,
        )
    finally:
        if role == "implementer":
            _set_active_role(root, None)
    if completed.returncode != 0:
        detail = completed.stderr.strip() or completed.stdout.strip() or "cursor-agent failed"
        raise RuntimeError(detail)
    try:
        return json.loads(completed.stdout)
    except json.JSONDecodeError:
        return {"text": completed.stdout}


def _set_active_role(root: Path, role: str | None) -> None:
    path = root / ".cursor" / "hooks" / "state" / "active_role"
    path.parent.mkdir(parents=True, exist_ok=True)
    if role is None:
        path.unlink(missing_ok=True)
    else:
        path.write_text(role)


def _run_stub(role: str, brief: Brief, root: Path, run_id: str) -> dict:
    spec = brief.candidate or BriefCandidate()
    if role == "planner":
        proposal = Candidate(
            id=spec.id,
            family=spec.family,
            claim=spec.claim,
            operator="draft",
            kind="model",
            fit_argument="Separable toy features are linearly classifiable.",
            cheap_test="Smoke loss decreases on the toy frame.",
            kill_if="Smoke scores are non-finite or the loss does not fall.",
            est_cost=spec.est_dbu,
        )
        path = root / "runs" / run_id / "proposals.yaml"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(yaml.safe_dump({"candidates": [proposal.model_dump()]}, sort_keys=False))
        return {"candidates": [proposal.model_dump()]}
    if role == "critic":
        verdict = Verdict(
            candidate_id=spec.id,
            decision="PASS",
            reason="Toy skeleton accepts the single logistic proposal.",
        )
        path = root / "runs" / run_id / "verdict.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(verdict.model_dump_json(indent=2) + "\n")
        return {"verdicts": [verdict.model_dump()]}
    if role == "implementer":
        _set_active_role(root, "implementer")
        try:
            _write_candidate(root, spec.id, spec)
        finally:
            _set_active_role(root, None)
        return {"wrote": f"candidates/{spec.id}/model.py"}
    raise ValueError(f"unknown role {role}")


def _write_candidate(root: Path, candidate_id: str, candidate: BriefCandidate) -> None:
    directory = root / "candidates" / candidate_id
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "model.py").write_text(MODEL_PY)
    (directory / "candidate.yaml").write_text(
        f"id: {candidate_id}\n"
        f"family: {candidate.family}\n"
        f"claim: {candidate.claim}\n"
        "operator: draft\n"
    )
    (directory / "spec.md").write_text(
        f"# {candidate_id}\n\n"
        "Fit a logistic regression on the frame passed to `fit` and return "
        "finite scores from `score`.\n"
    )
