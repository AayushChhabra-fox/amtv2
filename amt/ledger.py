"""Append-only ledger at ``ledger/results.jsonl``.

Only the control layer under ``amt/`` (and the test suite) may append. There
is no update or truncate. Agents never write this file.
"""

from __future__ import annotations

import fcntl
import inspect
import json
import os
from pathlib import Path

from amt.schemas import LedgerRow


class LedgerWriteError(PermissionError):
    """A caller outside ``amt/`` tried to append a ledger row."""


def ledger_path(root: Path) -> Path:
    return root / "ledger" / "results.jsonl"


def append(root: Path, row: LedgerRow) -> None:
    """Append one JSON line. Earlier lines are left untouched."""
    _assert_control_writer()
    path = ledger_path(root)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = (row.model_dump_json() + "\n").encode("utf-8")
    flags = os.O_WRONLY | os.O_CREAT | os.O_APPEND
    descriptor = os.open(path, flags, 0o644)
    try:
        fcntl.flock(descriptor, fcntl.LOCK_EX)
        view = memoryview(payload)
        while view:
            written = os.write(descriptor, view)
            if written <= 0:
                raise OSError(f"short write to {path}")
            view = view[written:]
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def read_rows(root: Path) -> list[LedgerRow]:
    path = ledger_path(root)
    if not path.is_file():
        return []
    rows: list[LedgerRow] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(LedgerRow.model_validate(json.loads(line)))
    return rows


def _assert_control_writer() -> None:
    frames = inspect.stack()
    if len(frames) < 3:
        raise LedgerWriteError("refused ledger write without a Python caller")
    caller = Path(frames[2].filename).resolve()
    package = Path(__file__).resolve().parent
    tests = package.parent / "tests"
    if package in caller.parents or tests in caller.parents:
        return
    raise LedgerWriteError(
        f"ledger/results.jsonl is append-only and only amt/ may write it (refused caller {caller})"
    )
