#!/usr/bin/env python3
"""Block implementer writes outside candidates/<id>/.

Cursor's preToolUse payload does not include the subagent name. subagentStart
does (`subagent_type`). This script records an active implementer on
subagentStart and clears it on subagentStop. preToolUse and beforeShellExecution
enforce the write scope only while that record, `.cursor/hooks/state/active_role`,
or AMT_ROLE=implementer says the implementer is the writer.

Python 3.9 compatible: the hook runs with the system interpreter.
"""

from __future__ import annotations

import json
import os
import re
import sys
from pathlib import Path

PROTECTED_PREFIXES = ("harness/", "amt/", "briefs/", "datasets/", "resources/")
PROTECTED_FILES = {"databricks.yml"}
STATE_NAME = "active_implementers.json"


def project_root():
    env = os.environ.get("CURSOR_PROJECT_DIR") or os.environ.get("CLAUDE_PROJECT_DIR")
    if env:
        return Path(env)
    return Path(__file__).resolve().parents[2]


def state_path():
    return project_root() / ".cursor" / "hooks" / "state" / STATE_NAME


def role_path():
    return project_root() / ".cursor" / "hooks" / "state" / "active_role"


def load_state():
    path = state_path()
    if not path.is_file():
        return {"ids": []}
    try:
        payload = json.loads(path.read_text())
    except (OSError, ValueError):
        return {"ids": []}
    if not isinstance(payload, dict):
        return {"ids": []}
    ids = payload.get("ids") or []
    return {"ids": [str(item) for item in ids]}


def save_state(payload):
    path = state_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload))


def implementer_active(event):
    if os.environ.get("AMT_ROLE") == "implementer":
        return True
    role_file = role_path()
    if role_file.is_file() and role_file.read_text().strip() == "implementer":
        return True
    subagent_type = str(event.get("subagent_type") or "")
    if subagent_type == "implementer":
        return True
    return bool(load_state()["ids"])


def mark_start(event):
    subagent_type = str(event.get("subagent_type") or "")
    task = str(event.get("task") or "")
    if subagent_type != "implementer" and not task.startswith("implementer:"):
        return
    payload = load_state()
    subagent_id = str(event.get("subagent_id") or "implementer")
    if subagent_id not in payload["ids"]:
        payload["ids"].append(subagent_id)
    save_state(payload)


def mark_stop(event):
    payload = load_state()
    subagent_id = str(event.get("subagent_id") or "")
    payload["ids"] = [item for item in payload["ids"] if item != subagent_id]
    save_state(payload)


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


def candidate_write_ok(rel):
    parts = [part for part in rel.split("/") if part not in ("", ".")]
    if ".." in parts:
        return False
    if len(parts) < 2 or parts[0] != "candidates":
        return False
    candidate_id = parts[1]
    if not candidate_id or candidate_id.startswith("."):
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


def decide_file_write(event):
    if not implementer_active(event):
        return allow()
    paths = write_paths(event)
    if not paths:
        return deny("Implementer write had no path.")
    root_list = roots_from(event)
    for raw in paths:
        rel = relative_path(raw, root_list)
        if not candidate_write_ok(rel):
            return deny("Implementer may only write under candidates/<id>/. Blocked: " + rel)
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
        return deny("Implementer shell write could not be scoped to candidates/<id>/. ")
    root_list = roots_from(event)
    for target in targets:
        rel = relative_path(target.strip("\"'"), root_list)
        if not candidate_write_ok(rel):
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
