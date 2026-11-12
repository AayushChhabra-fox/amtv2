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


def _model_for(role: str, explicit: str | None) -> str | None:
    if explicit:
        return explicit
    chosen = os.environ.get(f"AMT_MODEL_{role.upper()}", "").strip()
    return chosen or None


def _timeout_s(explicit: int | None) -> int:
    if explicit is not None:
        return explicit
    return int(os.environ.get("AMT_AGENT_TIMEOUT_S", "600"))


def _token_cap(explicit: int | None) -> int | None:
    if explicit is not None:
        return explicit
    raw = os.environ.get("AMT_TOKEN_CAP", "32000").strip()
    if not raw:
        return None
    return int(raw)


def run_role(
    role: str,
    brief: Brief,
    root: Path,
    run_id: str,
    *,
    node_id: str | None = None,
    operator: str | None = None,
    parent_ids: list[str] | None = None,
    output_path: Path | None = None,
    inputs: list[str] | None = None,
    model: str | None = None,
    timeout_s: int | None = None,
    token_cap: int | None = None,
    note: str = "",
) -> dict:
    """Call one role. Every call is logged with its model, paths, timeout, and tokens."""

    started = time.perf_counter()
    chosen_model = _model_for(role, model)
    timeout = _timeout_s(timeout_s)
    cap = _token_cap(token_cap)
    prompt = build_context(
        role,
        brief,
        root,
        run_id=run_id,
        node_id=node_id,
        operator=operator,
        parent_ids=parent_ids,
        output_path=None if output_path is None else str(output_path),
        inputs=inputs,
        note=note,
    )
    mode = _backend()
    common = {
        "role": role,
        "model": chosen_model,
        "timeout_s": timeout,
        "token_cap": cap,
        "node_id": node_id,
        "operator": operator,
        "inputs": inputs or [],
        "output_path": None if output_path is None else str(output_path),
    }
    if mode == "auto":
        mode = "cursor" if cursor_agent_logged_in() else "stub"
        if mode == "stub":
            _log_call(
                root,
                run_id,
                {
                    **common,
                    "backend": "cursor",
                    "ok": False,
                    "latency_s": round(time.perf_counter() - started, 4),
                    "tokens": None,
                    "error": "cursor-agent is not logged in; toy loop used the stub",
                },
            )
            started = time.perf_counter()
    try:
        if mode == "cursor":
            output = _run_cursor(
                role, root, prompt, model=chosen_model, timeout_s=timeout
            )
        elif mode == "fake":
            output = _run_fake(role, root, run_id, node_id=node_id)
        else:
            output = _run_stub(role, brief, root, run_id)
        tokens = _token_count(output)
        if cap is not None and tokens is not None and tokens > cap:
            raise RuntimeError(f"token cap {cap} exceeded ({tokens})")
    except Exception as exc:
        _log_call(
            root,
            run_id,
            {
                **common,
                "backend": mode,
                "ok": False,
                "latency_s": round(time.perf_counter() - started, 4),
                "tokens": None,
                "error": str(exc),
            },
        )
        raise
    _log_call(
        root,
        run_id,
        {
            **common,
            "backend": mode,
            "ok": True,
            "latency_s": round(time.perf_counter() - started, 4),
            "tokens": tokens,
        },
    )
    return output


def coerce_role_output(output: object) -> object:
    """Pull a planner or critic object out of a cursor-agent JSON envelope."""

    if not isinstance(output, dict):
        return output
    if any(key in output for key in ("candidates", "proposals", "verdicts", "decision")):
        return output
    for key in ("text", "result", "output"):
        parsed = _parse_embedded(output.get(key))
        if isinstance(parsed, (dict, list)):
            return parsed
    return output


def _parse_embedded(value: object) -> object | None:
    if isinstance(value, (dict, list)):
        return value
    if not isinstance(value, str):
        return None
    text = value.strip()
    if not text:
        return None
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    start = text.find("{")
    end = text.rfind("}")
    if start >= 0 and end > start:
        try:
            return json.loads(text[start : end + 1])
        except json.JSONDecodeError:
            return None
    return None


def _token_count(output: dict) -> int | None:
    usage = output.get("usage") or {}
    total = usage.get("total_tokens") or usage.get("output_tokens")
    return int(total) if total is not None else None


def _run_cursor(
    role: str,
    root: Path,
    prompt: str,
    *,
    model: str | None,
    timeout_s: int,
) -> dict:
    binary = cursor_agent_bin()
    if binary is None:
        raise RuntimeError("cursor-agent is not on PATH")
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
    if model:
        command.extend(["--model", model])
    if role in {"planner", "critic"}:
        command.extend(["--mode", "plan"])
    if role in {"implementer", "reviewer"}:
        _set_active_role(root, role)
    command.append(prompt)
    try:
        completed = subprocess.run(
            command,
            check=False,
            capture_output=True,
            text=True,
            timeout=timeout_s,
            cwd=root,
        )
    finally:
        if role in {"implementer", "reviewer"}:
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


def _run_fake(role: str, root: Path, run_id: str, *, node_id: str | None) -> dict:
    path = root / "runs" / run_id / "fake_agent.json"
    if not path.is_file():
        raise RuntimeError(f"fake agent script is missing: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    calls = payload.get("calls")
    if not isinstance(calls, list):
        raise RuntimeError("fake agent script needs a calls list")
    cursor = int(payload.get("cursor") or 0)
    if cursor >= len(calls):
        raise RuntimeError("fake agent script is exhausted")
    call = calls[cursor]
    expected = str(call.get("role") or "")
    if expected != role:
        raise RuntimeError(f"fake agent expected {expected}, got {role}")
    scripted_node = call.get("node_id")
    if scripted_node and node_id and scripted_node != node_id:
        raise RuntimeError(f"fake agent expected node {scripted_node}, got {node_id}")
    target = scripted_node or node_id
    payload["cursor"] = cursor + 1
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    if call.get("error"):
        raise RuntimeError(str(call["error"]))
    files = call.get("files") or {}
    if isinstance(files, dict) and files:
        _write_scripted_files(root, files, role)
    elif role == "implementer" and target:
        _set_active_role(root, "implementer")
        try:
            _write_default_model(root, target)
        finally:
            _set_active_role(root, None)
    elif role == "reviewer" and target:
        _set_active_role(root, "reviewer")
        try:
            raw_verdict = str((call.get("output") or {}).get("verdict", ""))
            verdict = "FAIL" if raw_verdict.upper() == "FAIL" else "PASS"
            _write_text(root, f"candidates/{target}/review.md", f"# review\n\nVERDICT: {verdict}\n")
        finally:
            _set_active_role(root, None)
    output = call.get("output")
    if output is None:
        output = {}
    if not isinstance(output, dict):
        raise RuntimeError("fake agent output must be an object")
    usage = call.get("usage")
    if isinstance(usage, dict):
        output = {**output, "usage": usage}
    return output


def _write_scripted_files(root: Path, files: dict, role: str) -> None:
    if role in {"implementer", "reviewer"}:
        _set_active_role(root, role)
    try:
        for relative, contents in files.items():
            _write_text(root, str(relative), str(contents))
    finally:
        if role in {"implementer", "reviewer"}:
            _set_active_role(root, None)


def _write_text(root: Path, relative: str, contents: str) -> None:
    path = _candidate_path(root, relative)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(contents, encoding="utf-8")


def _candidate_path(root: Path, relative: str) -> Path:
    rel = Path(relative)
    if rel.is_absolute() or ".." in rel.parts or len(rel.parts) < 3 or rel.parts[0] != "candidates":
        raise RuntimeError(f"refusing agent write outside candidates/<id>/: {relative}")
    if rel.parts[1].startswith("."):
        raise RuntimeError(f"refusing agent write outside candidates/<id>/: {relative}")
    return root / rel


def _write_default_model(root: Path, node_id: str) -> None:
    directory = root / "candidates" / node_id
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "model.py").write_text(MODEL_PY, encoding="utf-8")


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
    if role == "reviewer":
        _set_active_role(root, "reviewer")
        try:
            _write_text(
                root,
                f"candidates/{spec.id}/review.md",
                "# review\n\nThe claimed logistic ranker is present.\n\nVERDICT: PASS\n",
            )
        finally:
            _set_active_role(root, None)
        return {"verdict": "PASS"}
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
