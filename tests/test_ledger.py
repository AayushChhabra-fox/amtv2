import json
from pathlib import Path

from amt.ledger import append, read_rows
from amt.schemas import Brief, LedgerRow


def test_brief_and_ledger_round_trip(tmp_path: Path) -> None:
    brief = Brief.model_validate(
        {
            "name": "toy",
            "run_id": "toy",
            "approval_mode": "none",
            "task": {"query_unit": "one row"},
            "budget": {"total_dbu": 10},
            "extra_field": "kept",
        }
    )
    assert brief.name == "toy"
    assert brief.model_extra["extra_field"] == "kept"

    row = LedgerRow(
        run_id="toy",
        node_id="c00",
        operator="draft",
        family="linear",
        claim="logistic",
        rung="R0",
        seed=0,
        status="kept",
        git_sha="abc",
        job_run_id="1",
        mlflow_run_id="m1",
        metric=0.5,
    )
    append(tmp_path, row)
    loaded = read_rows(tmp_path)
    assert loaded == [row]
    raw = (tmp_path / "ledger" / "results.jsonl").read_text().strip()
    assert json.loads(raw)["node_id"] == "c00"
