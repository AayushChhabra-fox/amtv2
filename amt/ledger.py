"""Append-only ledger. Only control-layer code writes this file."""

from __future__ import annotations

import json
from pathlib import Path

from amt.schemas import LedgerRow


def ledger_path(root: Path) -> Path:
    return root / "ledger" / "results.jsonl"


def append(root: Path, row: LedgerRow) -> None:
    path = ledger_path(root)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(row.model_dump_json() + "\n")


def read_rows(root: Path) -> list[LedgerRow]:
    path = ledger_path(root)
    if not path.is_file():
        return []
    rows: list[LedgerRow] = []
    for line in path.read_text().splitlines():
        if line.strip():
            rows.append(LedgerRow.model_validate(json.loads(line)))
    return rows
