#!/usr/bin/env python3
"""Block implementer and reviewer writes outside one candidate directory.

`.cursor/hooks.json` is schema version 1 from the Cursor command-hook format:
each entry has a command, and preToolUse / beforeShellExecution set failClosed
so a crash denies the write. preToolUse uses a tool-name matcher.

Cursor's preToolUse payload does not include the subagent name. subagentStart
does (`subagent_type`). This script records an active IDE writer on
subagentStart and clears it on subagentStop. Each record is
`{subagent_id: {"pid": ppid, "ts": now, "role": ...}}`. Every read drops
entries whose pid is dead or whose timestamp is older than LEASE_TTL_S, so a
crashed subagent cannot lock the workspace until the IDE restarts.

`cursor-agent -p` is a different process. The parent passes AMT_ROLE,
AMT_RUN_ID, AMT_CANDIDATE_DIR, and AMT_SPEC in the child environment. When
those variables reach this hook they are authoritative. When Cursor strips
them, the parent also writes `.cursor/hooks/state/leases/<pid>.json` and this
hook applies a lease only when that pid is an ancestor of the hook process
and `os.kill(pid, 0)` succeeds. Other terminals are not descendants, so they
stay unrestricted.

Allowed writes are `runs/<run_id>/candidates/<node_id>/...`. When
AMT_CANDIDATE_DIR or a matching lease names a directory, the write must sit
under that exact prefix. Those roles cannot write `harness/`, `amt/`,
`briefs/`, `datasets/`, `resources/`, or `databricks.yml`.

Python 3.9 compatible: the hook runs with the system interpreter.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

PROTECTED_PREFIXES = ("harness/", "amt/", "briefs/", "datasets/", "resources/")
PROTECTED_FILES = {"databricks.yml"}
RESTRICTED_ROLES = {"implementer", "reviewer"}
STATE_NAME = "active_implementers.json"
LEASE_TTL_S = 2 * 60 * 60


def project_root():
    env = os.environ.get("CURSOR_PROJECT_DIR") or os.environ.get("CLAUDE_PROJECT_DIR")
    if env:
        return Path(env)
    return Path(__file__).resolve().parents[2]


def state_path():
    return project_root() / ".cursor" / "hooks" / "state" / STATE_NAME


def leases_dir():
    return project_root() / ".cursor" / "hooks" / "state" / "leases"


def _read_json(path):
    try:
        payload = json.loads(path.read_text())
    except (OSError, ValueError):
        return None
    return payload


def _pid_alive(pid):
    try:
        number = int(pid)
    except (TypeError, ValueError):
        return False
    if number <= 0:
        return False
    try:
        os.kill(number, 0)
    except OSError:
        return False
    return True


def _parent_pid(pid):
    try:
        completed = subprocess.run(
            ["ps", "-o", "ppid=", "-p", str(pid)],
            check=False,
            capture_output=True,
            text=True,
        )
    except OSError:
        return 0
    if completed.returncode != 0:
        return 0
    text = (completed.stdout or "").strip()
    try:
        return int(text)
    except ValueError:
        return 0


def _lease_current(data, now):
    if not isinstance(data, dict):
        return False
    try:
        pid = int(data.get("pid"))
        ts = float(data.get("ts"))
    except (TypeError, ValueError):
        return False
    if not _pid_alive(pid):
        return False
    return now - ts <= LEASE_TTL_S


def _prune_leases():
    directory = leases_dir()
    if not directory.is_dir():
        return
    now = time.time()
    for path in directory.glob("*.json"):
        if not _lease_current(_read_json(path), now):
            try:
                path.unlink()
            except OSError:
                pass


def _ancestor_lease():
    pid = os.getppid()
    seen = set()
    now = time.time()
    while pid > 1 and pid not in seen:
        seen.add(pid)
        path = leases_dir() / ("%s.json" % pid)
        data = _read_json(path) if path.is_file() else None
        if _lease_current(data, now):
            return data
        pid = _parent_pid(pid)
    return None


def load_state():
    path = state_path()
    if not path.is_file():
        return {}
    payload = _read_json(path)
    raw = {}
    if isinstance(payload, dict) and isinstance(payload.get("agents"), dict):
        raw = payload["agents"]
    now = time.time()
    kept = {}
    for key, value in raw.items():
        if not isinstance(value, dict) or not _lease_current(value, now):
            continue
        kept[str(key)] = {
            "pid": int(value["pid"]),
            "ts": float(value["ts"]),
            "role": str(value.get("role") or ""),
        }
    canonical = {"agents": kept}
    if payload != canonical:
        save_state(canonical)
    return kept


def save_state(payload):
    path = state_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload))


def _env_role():
    return os.environ.get("AMT_ROLE", "").strip()


def implementer_active(event):
    role = _env_role()
    if role:
        return role in RESTRICTED_ROLES
    lease = _ancestor_lease()
    if lease and str(lease.get("role") or "") in RESTRICTED_ROLES:
        return True
    subagent_type = str(event.get("subagent_type") or "")
    if subagent_type in RESTRICTED_ROLES:
        return True
    return bool(load_state())


def _restricted_role(event):
    subagent_type = str(event.get("subagent_type") or "")
    task = str(event.get("task") or "")
    if subagent_type in RESTRICTED_ROLES:
        return subagent_type
    if task.startswith("implementer:"):
        return "implementer"
    if task.startswith("reviewer:"):
        return "reviewer"
    return ""


def mark_start(event):
    role = _restricted_role(event)
    if not role:
        return
    agents = load_state()
    subagent_id = str(event.get("subagent_id") or role)
    agents[subagent_id] = {"pid": os.getppid(), "ts": time.time(), "role": role}
    save_state({"agents": agents})


def mark_stop(event):
    agents = load_state()
    subagent_id = str(event.get("subagent_id") or "")
    agents.pop(subagent_id, None)
    save_state({"agents": agents})


def relative_path(raw, roots):
    text = str(raw).replace("\\", "/")
    for root in roots:
        root_text = str(root).replace("\\", "/").rstrip("/")
        if text.startswith(root_text + "/"):
            text = text[len(root_text) + 1 :]
            break
    if text.startswith("./"):
        text = text[2:]
    return text


def _forced_candidate_dir():
    forced = os.environ.get("AMT_CANDIDATE_DIR", "").strip()
    if forced:
        return forced
    if _env_role():
        return ""
    lease = _ancestor_lease()
    if not lease:
        return ""
    return str(lease.get("candidate_dir") or "").strip()


def _within(raw, forced, roots):
    base = Path(forced)
    if not base.is_absolute():
        base = project_root() / base
    try:
        base = base.resolve()
    except OSError:
        return False
    raw_path = Path(str(raw))
    candidates = [raw_path] if raw_path.is_absolute() else []
    for root in roots:
        candidates.append(Path(root) / raw_path)
    for candidate in candidates:
        try:
            resolved = candidate.resolve()
        except OSError:
            continue
        if resolved == base or base in resolved.parents:
            return True
    return False


def candidate_write_ok(rel, raw, roots):
    parts = [part for part in rel.split("/") if part not in ("", ".")]
    if ".." in parts:
        return False
    forced = _forced_candidate_dir()
    if forced:
        return _within(raw, forced, roots)
    if len(parts) < 4 or parts[0] != "runs" or parts[2] != "candidates":
        return False
    run_id = parts[1]
    node_id = parts[3]
    if not run_id or run_id.startswith(".") or not node_id or node_id.startswith("."):
        return False
    return True


def write_paths(event):
    tool_input = event.get("tool_input") or {}
    if isinstance(tool_input, str):
        try:
            tool_input = json.loads(tool_input)
        except ValueError:
            tool_input = {}
    if not isinstance(tool_input, dict):
        return []
    found = []
    for key in ("path", "file_path", "target_file", "notebook_path", "target_notebook"):
        if tool_input.get(key):
            found.append(tool_input[key])
    return found


def roots_from(event):
    roots = list(event.get("workspace_roots") or [])
    roots.append(str(project_root()))
    cwd = event.get("cwd")
    if cwd:
        roots.append(cwd)
    return roots


def is_protected(rel):
    name = rel.rstrip("/")
    if name in PROTECTED_FILES or name.endswith("/databricks.yml"):
        return True
    return name.startswith(PROTECTED_PREFIXES)


def decide_file_write(event):
    if not implementer_active(event):
        return allow()
    paths = write_paths(event)
    if not paths:
        return deny("Implementer write had no path.")
    root_list = roots_from(event)
    for raw in paths:
        rel = relative_path(raw, root_list)
        if is_protected(rel) or not candidate_write_ok(rel, raw, root_list):
            return deny(
                "Implementer may only write under runs/<run_id>/candidates/<id>/. Blocked: "
                + rel
            )
    return allow()


def decide_shell(event):
    if not implementer_active(event):
        return allow()
    command = str(event.get("command") or "")
    targets = re.findall(r">>?\s*([^\s;|&]+)", command)
    copy = re.search(r"\b(cp|mv|tee|install)\b(.*)$", command)
    if copy:
        tokens = copy.group(2).split()
        if tokens:
            targets.append(tokens[-1])
    write_markers = (">", "tee ", "cp ", "mv ", "install ", "sed -i")
    if not targets and not any(marker in command for marker in write_markers):
        return allow()
    if not targets:
        return deny(
            "Implementer shell write could not be scoped to runs/<run_id>/candidates/<id>/. "
        )
    root_list = roots_from(event)
    for target in targets:
        raw = target.strip("\"'")
        rel = relative_path(raw, root_list)
        if is_protected(rel) or not candidate_write_ok(rel, raw, root_list):
            return deny("Implementer shell write blocked: " + rel)
    return allow()


def allow():
    return {"permission": "allow"}


def deny(message):
    return {
        "permission": "deny",
        "user_message": message,
        "agent_message": message,
    }


def main():
    _prune_leases()
    raw = sys.stdin.read()
    try:
        event = json.loads(raw) if raw.strip() else {}
    except ValueError:
        event = {}
    name = str(event.get("hook_event_name") or "")
    if name == "subagentStart":
        mark_start(event)
        result = allow()
    elif name == "subagentStop":
        mark_stop(event)
        result = {}
    elif name == "beforeShellExecution":
        result = decide_shell(event)
    elif name == "preToolUse":
        result = decide_file_write(event)
    else:
        result = allow()
    sys.stdout.write(json.dumps(result))


if __name__ == "__main__":
    main()
