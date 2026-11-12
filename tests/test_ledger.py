import importlib.util
from pathlib import Path

import pytest
from pydantic import ValidationError

from amt.ledger import append, read_rows
from amt.schemas import DeltaVsBaseline, LedgerRow


def _row(**overrides: object) -> LedgerRow:
    payload: dict[str, object] = {
        "run_id": "toy",
        "node_id": "c00",
        "operator": "draft",
        "family": "linear",
        "claim": "logistic",
        "rung": "R0",
        "seed": 0,
        "status": "kept",
        "git_sha": "abc",
        "job_run_id": "1",
        "mlflow_run_id": "m1",
        "metric": 0.5,
        "delta_vs_baseline": {"point": 0.02, "ci_lo": 0.0, "ci_hi": 0.04},
    }
    payload.update(overrides)
    return LedgerRow.model_validate(payload)


def test_append_round_trip_preserves_delta(tmp_path: Path) -> None:
    row = _row()
    assert row.delta_vs_baseline == DeltaVsBaseline(point=0.02, ci_lo=0.0, ci_hi=0.04)
    append(tmp_path, row)
    assert read_rows(tmp_path) == [row]


def test_append_does_not_rewrite_earlier_rows(tmp_path: Path) -> None:
    first = _row(claim='say "hi"\nnext')
    append(tmp_path, first)
    path = tmp_path / "ledger" / "results.jsonl"
    before = path.read_bytes()
    append(tmp_path, _row(node_id="c01"))
    after = path.read_bytes()
    assert after.startswith(before)
    assert before.count(b"\n") == 1
    assert after.count(b"\n") == 2
    rows = read_rows(tmp_path)
    assert [row.node_id for row in rows] == ["c00", "c01"]
    assert rows[0] == first
    assert rows[0].claim == 'say "hi"\nnext'


def test_corrupt_line_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "ledger" / "results.jsonl"
    path.parent.mkdir(parents=True)
    path.write_text('{"run_id": "toy"}\n', encoding="utf-8")
    with pytest.raises(ValidationError):
        read_rows(tmp_path)


def test_failed_row_requires_a_reason() -> None:
    with pytest.raises(ValidationError):
        _row(status="failed", kill_reason=None)
    failed = _row(status="failed", kill_reason="no metric")
    assert failed.kill_reason == "no metric"


def test_only_amt_may_append(tmp_path: Path) -> None:
    outsider = tmp_path / "outsider.py"
    outsider.write_text(
        "from amt.ledger import append\n\ndef go(root, row):\n    append(root, row)\n",
        encoding="utf-8",
    )
    spec = importlib.util.spec_from_file_location("outsider_writer", outsider)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    with pytest.raises(PermissionError):
        module.go(tmp_path, _row())
    assert not (tmp_path / "ledger" / "results.jsonl").exists()
