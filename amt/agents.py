"""Agent calls.

`cursor-agent` (installed as `agent` / `cursor-agent`, CLI 2026.11.11) is the
live backend. Flags confirmed from `cursor-agent --help`:

- `-p` / `--print` non-interactive, with write and shell tools
- `--output-format json|text|stream-json` (only with `--print`)
- `--model <model>`
- `--mode ask` for read-only roles. Plan mode ends a headless run with a
  CreatePlan call that is not saved, and `result` comes back empty.
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
import random
import shutil
import subprocess
import time
from pathlib import Path

import yaml

from amt.context import build_context
from amt.paths import candidate_dir
from amt.schemas import Brief, BriefCandidate, Candidate, Verdict

# Three tries. A transient failure waits 5s, then 15s. 45s is the next step of the same series.
_RETRY_BACKOFF_S = (5, 15, 45)
_TRANSIENT_MARKERS = ("resource_exhausted", "[internal]", "connection lost", "timed out", "timeout")


class AgentCallError(RuntimeError):
    """A cursor-agent call failed. ``retries`` is how many extra tries already ran."""

    def __init__(self, message: str, *, retries: int = 0) -> None:
        super().__init__(message)
        self.retries = retries

MODEL_PY = '''\
"""Contract-shaped ranker. Ridge on numeric feat__ columns; categories are ignored."""

from __future__ import annotations

import pickle

import numpy as np


class Ranker:
    def __init__(self) -> None:
        self.columns: list[str] = []
        self.weights: np.ndarray | None = None

    def fit(self, train, valid, ctx) -> None:
        del valid, ctx
        matrix, names = train.numeric_matrix()
        self.columns = list(names)
        labels = np.asarray(train.labels(), dtype=float)
        width = matrix.shape[1]
        if width == 0:
            self.weights = np.zeros(0, dtype=float)
            return
        gram = matrix.T @ matrix + 1e-3 * np.eye(width)
        self.weights = np.linalg.solve(gram, matrix.T @ labels)

    def score(self, frame) -> np.ndarray:
        if self.weights is None:
            raise RuntimeError("Ranker has not been fit")
        matrix, _names = frame.numeric_matrix(self.columns)
        if matrix.shape[1] == 0:
            return np.zeros(len(frame), dtype=float)
        return matrix @ self.weights

    def save(self, path: str) -> None:
        with open(path, "wb") as handle:
            pickle.dump({"columns": self.columns, "weights": self.weights}, handle)

    @classmethod
    def load(cls, path: str) -> "Ranker":
        ranker = cls()
        with open(path, "rb") as handle:
            payload = pickle.load(handle)
        ranker.columns = list(payload["columns"])
        ranker.weights = payload["weights"]
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


_WEB_TOOLS: bool | None = None


def cursor_agent_web_tools_available() -> bool:
    """True when this CLI can search the web from ask mode.

    ``--help`` does not name web search on its own. Ask mode is read-only, and
    ``--print`` says the process has every tool, so web search is in that set.
    The result is cached for the process. A missing signal is a hard failure
    for a live planner call, not a silent skip.
    """

    global _WEB_TOOLS
    if _WEB_TOOLS is not None:
        return _WEB_TOOLS
    binary = cursor_agent_bin()
    if binary is None:
        _WEB_TOOLS = False
        return False
    completed = subprocess.run(
        [binary, "--help"],
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
    )
    text = f"{completed.stdout}\n{completed.stderr}".lower()
    has_ask = "--mode" in text and "ask" in text
    has_tools = "web search" in text or "all tools" in text
    _WEB_TOOLS = completed.returncode == 0 and has_ask and has_tools
    return _WEB_TOOLS


def cursor_agent_logged_in() -> bool:
    binary = cursor_agent_bin()
    if binary is None:
        return False
    # `status` reports only a browser login; -p calls also accept CURSOR_API_KEY.
    if os.environ.get("CURSOR_API_KEY", "").strip():
        return True
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
    reply_path: str | None = None
    retries = 0
    spec_name = ""
    resolved_spec = brief.resolved_dataset_spec()
    if resolved_spec:
        spec_name = Path(resolved_spec).name
    try:
        if mode == "cursor" and role == "planner" and not cursor_agent_web_tools_available():
            raise RuntimeError(
                "planner requires cursor-agent web search in ask mode; "
                "the CLI help text does not offer it"
            )
        if mode == "cursor":
            output, retries = _run_cursor(
                role,
                root,
                prompt,
                model=chosen_model,
                timeout_s=timeout,
                run_id=run_id,
                node_id=node_id,
                spec=spec_name,
            )
        elif mode == "fake":
            output, retries = _call_with_retry(
                lambda: _run_fake(role, root, run_id, node_id=node_id)
            )
        else:
            output = _run_stub(role, brief, root, run_id)
        if role in {"planner", "critic"}:
            reply_path = _save_reply(root, run_id, role, output)
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
                "references_count": _references_count(role, None),
                "reply_path": reply_path,
                "retries": getattr(exc, "retries", retries),
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
            "references_count": _references_count(role, output),
            "reply_path": reply_path,
            "retries": retries,
        },
    )
    return output


def _save_reply(root: Path, run_id: str, role: str, output: object) -> str:
    """Keep the raw reply, so one the orchestrator cannot parse can still be read."""

    directory = root / "runs" / run_id / "agent_replies"
    directory.mkdir(parents=True, exist_ok=True)
    number = sum(1 for _ in directory.glob("*.json")) + 1
    path = directory / f"{number:03d}_{role}.json"
    path.write_text(json.dumps(output, indent=2, default=str) + "\n", encoding="utf-8")
    return str(path.relative_to(root))


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


def _references_count(role: str, output: object) -> int | None:
    """How many cited URLs a planner call returned. Other roles log null."""

    if role != "planner":
        return None
    payload = coerce_role_output(output) if output is not None else None
    if not isinstance(payload, dict):
        return 0
    candidates = payload.get("candidates") or payload.get("proposals") or []
    if not isinstance(candidates, list):
        return 0
    count = 0
    for item in candidates:
        if not isinstance(item, dict):
            continue
        refs = item.get("references") or []
        if isinstance(refs, list):
            count += len(refs)
    return count


def _token_count(output: dict) -> int | None:
    usage = output.get("usage") or {}
    total = usage.get("total_tokens") or usage.get("output_tokens")
    return int(total) if total is not None else None


def _is_transient(detail: str) -> bool:
    text = detail.lower()
    return any(marker in text for marker in _TRANSIENT_MARKERS)


def _call_with_retry(operation):
    """Retry a transient agent failure. The delays are 5s, then 15s, then 45s."""

    last = "agent call failed"
    for attempt, delay in enumerate(_RETRY_BACKOFF_S):
        try:
            return operation(), attempt
        except AgentCallError as exc:
            last = str(exc)
            if not _is_transient(last) or attempt + 1 == len(_RETRY_BACKOFF_S):
                raise AgentCallError(last, retries=attempt) from exc
            time.sleep(delay + random.uniform(0, delay * 0.2))
    raise AgentCallError(last, retries=len(_RETRY_BACKOFF_S) - 1)


def _lease_path(root: Path, pid: int) -> Path:
    return root / ".cursor" / "hooks" / "state" / "leases" / f"{pid}.json"


def _write_lease(
    root: Path, pid: int, role: str, run_id: str, directory: str, spec: str
) -> Path:
    path = _lease_path(root, pid)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "pid": pid,
        "role": role,
        "run_id": run_id,
        "candidate_dir": directory,
        "spec": spec,
        "ts": time.time(),
    }
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _agent_env(role: str, run_id: str, directory: str, spec: str) -> dict[str, str]:
    env = os.environ.copy()
    env["AMT_ROLE"] = role
    env["AMT_RUN_ID"] = run_id
    if directory:
        env["AMT_CANDIDATE_DIR"] = directory
    if spec:
        env["AMT_SPEC"] = spec
    return env


def _run_cursor_once(
    command: list[str],
    *,
    root: Path,
    role: str,
    run_id: str,
    directory: str,
    spec: str,
    timeout_s: int,
    env: dict[str, str],
) -> dict:
    proc = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        cwd=root,
        env=env,
    )
    lease = None
    try:
        if role in {"implementer", "reviewer", "dataset_fixer"}:
            lease = _write_lease(root, proc.pid, role, run_id, directory, spec)
        try:
            stdout, stderr = proc.communicate(timeout=timeout_s)
        except subprocess.TimeoutExpired as exc:
            proc.kill()
            proc.communicate()
            raise AgentCallError(f"cursor-agent timed out after {timeout_s}s") from exc
        if proc.returncode != 0:
            detail = (stderr or stdout or "cursor-agent failed").strip()
            raise AgentCallError(detail)
        try:
            payload = json.loads(stdout)
        except json.JSONDecodeError:
            payload = {"text": stdout}
        if not isinstance(payload, dict):
            return {"text": stdout}
        return payload
    finally:
        if lease is not None:
            lease.unlink(missing_ok=True)


def _run_cursor(
    role: str,
    root: Path,
    prompt: str,
    *,
    model: str | None,
    timeout_s: int,
    run_id: str,
    node_id: str | None,
    spec: str,
) -> tuple[dict, int]:
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
        command.extend(["--mode", "ask"])
    command.append(prompt)
    directory = ""
    if node_id:
        directory = str(candidate_dir(root, run_id, node_id))
    env = _agent_env(role, run_id, directory, spec)
    last = "cursor-agent failed"
    for attempt, delay in enumerate(_RETRY_BACKOFF_S):
        try:
            output = _run_cursor_once(
                command,
                root=root,
                role=role,
                run_id=run_id,
                directory=directory,
                spec=spec,
                timeout_s=timeout_s,
                env=env,
            )
        except AgentCallError as exc:
            last = str(exc)
            transient = _is_transient(last)
            if not transient or attempt + 1 == len(_RETRY_BACKOFF_S):
                raise AgentCallError(last, retries=attempt) from exc
            time.sleep(delay + random.uniform(0, delay * 0.2))
            continue
        return output, attempt
    raise AgentCallError(last, retries=len(_RETRY_BACKOFF_S) - 1)


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
        message = str(call["error"])
        if _is_transient(message):
            raise AgentCallError(message)
        raise RuntimeError(message)
    files = call.get("files") or {}
    if isinstance(files, dict) and files:
        _write_scripted_files(root, run_id, files)
    elif role == "implementer" and target:
        _write_default_model(root, run_id, target)
    elif role == "reviewer" and target:
        raw_verdict = str((call.get("output") or {}).get("verdict", ""))
        verdict = "FAIL" if raw_verdict.upper() == "FAIL" else "PASS"
        _write_text(
            root,
            run_id,
            f"runs/{run_id}/candidates/{target}/review.md",
            f"# review\n\nVERDICT: {verdict}\n",
        )
    output = call.get("output")
    if output is None:
        output = {}
    if not isinstance(output, dict):
        raise RuntimeError("fake agent output must be an object")
    usage = call.get("usage")
    if isinstance(usage, dict):
        output = {**output, "usage": usage}
    return output


def _write_scripted_files(root: Path, run_id: str, files: dict) -> None:
    for relative, contents in files.items():
        _write_text(root, run_id, str(relative), str(contents))


def _write_text(root: Path, run_id: str, relative: str, contents: str) -> None:
    path = _candidate_path(root, run_id, relative)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(contents, encoding="utf-8")


def _candidate_path(root: Path, run_id: str, relative: str) -> Path:
    rel = Path(relative)
    parts = rel.parts
    allowed = f"runs/{run_id}/candidates/<id>/"
    if (
        rel.is_absolute()
        or ".." in parts
        or len(parts) < 5
        or parts[0] != "runs"
        or parts[1] != run_id
        or parts[2] != "candidates"
        or not parts[3]
        or parts[3].startswith(".")
    ):
        raise RuntimeError(f"refusing agent write outside {allowed}: {relative}")
    return root / rel


def _write_default_model(root: Path, run_id: str, node_id: str) -> None:
    directory = candidate_dir(root, run_id, node_id)
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
            fit_argument="Numeric feat columns are linearly related to the graded label.",
            cheap_test="Smoke scores are finite and deterministic on the contract frame.",
            kill_if="Smoke scores are non-finite or load does not reproduce them.",
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
            reason="The single contract-shaped ranker passes the critic.",
        )
        path = root / "runs" / run_id / "verdict.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(verdict.model_dump_json(indent=2) + "\n")
        return {"verdicts": [verdict.model_dump()]}
    if role == "implementer":
        _write_candidate(root, run_id, spec.id, spec)
        return {"wrote": f"runs/{run_id}/candidates/{spec.id}/model.py"}
    if role == "reviewer":
        _write_text(
            root,
            run_id,
            f"runs/{run_id}/candidates/{spec.id}/review.md",
            "# review\n\nThe claimed Ranker is present.\n\nVERDICT: PASS\n",
        )
        return {"verdict": "PASS"}
    raise ValueError(f"unknown role {role}")


def _write_candidate(root: Path, run_id: str, candidate_id: str, candidate: BriefCandidate) -> None:
    directory = candidate_dir(root, run_id, candidate_id)
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
        "Fit a ridge ranker on numeric feat columns and return finite scores.\n"
    )
