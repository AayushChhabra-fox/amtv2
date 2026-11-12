import json
import subprocess
from pathlib import Path

HOOK = Path(__file__).resolve().parents[1] / ".cursor" / "hooks" / "implementer_scope.py"


def _run(tmp_path: Path, event: dict, extra_env: dict | None = None) -> dict:
    env = {"CURSOR_PROJECT_DIR": str(tmp_path), "PATH": "/usr/bin:/bin"}
    if extra_env:
        env.update(extra_env)
    completed = subprocess.run(
        ["python3", str(HOOK)],
        input=json.dumps(event),
        text=True,
        capture_output=True,
        check=False,
        env=env,
    )
    assert completed.returncode == 0, completed.stderr
    return json.loads(completed.stdout)


def test_implementer_write_outside_candidates_is_denied(tmp_path: Path) -> None:
    start = _run(
        tmp_path,
        {
            "hook_event_name": "subagentStart",
            "subagent_type": "implementer",
            "subagent_id": "impl-1",
        },
    )
    assert start["permission"] == "allow"
    blocked = _run(
        tmp_path,
        {
            "hook_event_name": "preToolUse",
            "tool_name": "Write",
            "tool_input": {"path": str(tmp_path / "amt" / "schemas.py")},
            "workspace_roots": [str(tmp_path)],
            "cwd": str(tmp_path),
        },
    )
    assert blocked["permission"] == "deny"
    allowed = _run(
        tmp_path,
        {
            "hook_event_name": "preToolUse",
            "tool_name": "Write",
            "tool_input": {"path": str(tmp_path / "candidates" / "c00" / "model.py")},
            "workspace_roots": [str(tmp_path)],
            "cwd": str(tmp_path),
        },
    )
    assert allowed["permission"] == "allow"
    shell_blocked = _run(
        tmp_path,
        {
            "hook_event_name": "beforeShellExecution",
            "command": "cp model.py amt/stolen.py",
            "cwd": str(tmp_path),
            "workspace_roots": [str(tmp_path)],
        },
    )
    assert shell_blocked["permission"] == "deny"


def test_non_implementer_can_write_control_files(tmp_path: Path) -> None:
    decision = _run(
        tmp_path,
        {
            "hook_event_name": "preToolUse",
            "tool_name": "Write",
            "tool_input": {"path": str(tmp_path / "amt" / "orchestrator.py")},
            "workspace_roots": [str(tmp_path)],
        },
    )
    assert decision["permission"] == "allow"


def test_reviewer_cannot_write_protected_paths(tmp_path: Path) -> None:
    start = _run(
        tmp_path,
        {
            "hook_event_name": "subagentStart",
            "subagent_type": "reviewer",
            "subagent_id": "rev-1",
        },
    )
    assert start["permission"] == "allow"
    blocked = _run(
        tmp_path,
        {
            "hook_event_name": "preToolUse",
            "tool_name": "Write",
            "tool_input": {"path": "harness/train.py"},
            "workspace_roots": [str(tmp_path)],
        },
    )
    assert blocked["permission"] == "deny"
    allowed = _run(
        tmp_path,
        {
            "hook_event_name": "preToolUse",
            "tool_name": "Write",
            "tool_input": {"path": "candidates/c00/review.md"},
            "workspace_roots": [str(tmp_path)],
        },
    )
    assert allowed["permission"] == "allow"


def test_hooks_json_uses_the_command_hook_schema() -> None:
    payload = json.loads((HOOK.parents[1] / "hooks.json").read_text(encoding="utf-8"))
    assert payload["version"] == 1
    assert payload["hooks"]["preToolUse"][0]["failClosed"] is True
    assert payload["hooks"]["beforeShellExecution"][0]["failClosed"] is True
    assert payload["hooks"]["preToolUse"][0]["command"].endswith("implementer_scope.py")


def test_subagent_stop_clears_the_block(tmp_path: Path) -> None:
    _run(
        tmp_path,
        {
            "hook_event_name": "subagentStart",
            "subagent_type": "implementer",
            "subagent_id": "impl-2",
        },
    )
    _run(
        tmp_path,
        {
            "hook_event_name": "subagentStop",
            "subagent_type": "implementer",
            "subagent_id": "impl-2",
        },
    )
    decision = _run(
        tmp_path,
        {
            "hook_event_name": "preToolUse",
            "tool_name": "Write",
            "tool_input": {"path": "harness/train.py"},
            "workspace_roots": [str(tmp_path)],
        },
    )
    assert decision["permission"] == "allow"
