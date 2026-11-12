"""Status table, kill reasons, and the trace audit."""

import json
from pathlib import Path

from amt.ledger import append
from amt.schemas import DeltaVsBaseline, LedgerRow
from amt.status import audit_rows, format_status, main


def _row(**overrides: object) -> LedgerRow:
    payload: dict = {
        "run_id": "toy",
        "node_id": "c1",
        "operator": "draft",
        "family": "trees",
        "claim": "c1",
        "rung": "R1",
        "seed": 0,
        "status": "kept",
        "git_sha": "abc123",
        "job_run_id": "fake-1",
        "mlflow_run_id": "m1",
        "metric": 0.42,
        "delta_vs_baseline": {"point": 0.02, "ci_lo": 0.0, "ci_hi": 0.04},
    }
    payload.update(overrides)
    return LedgerRow.model_validate(payload)


def _state(root: Path) -> None:
    path = root / "runs" / "toy" / "state.json"
    path.parent.mkdir(parents=True)
    path.write_text(
        json.dumps(
            {
                "stage": "PRUNE",
                "round": 1,
                "live_nodes": ["c1", "c2"],
                "spent_dbu": 12.5,
                "budget_dbu": 100,
            }
        )
    )


def test_status_shows_spend_live_nodes_and_kills(tmp_path: Path) -> None:
    _state(tmp_path)
    append(tmp_path, _row())
    append(
        tmp_path,
        _row(
            node_id="c2",
            status="killed",
            kill_reason="below floor at R1 vs lgbm_reference",
            delta_vs_baseline=DeltaVsBaseline(point=-0.03, ci_lo=-0.05, ci_hi=-0.02),
            job_run_id="fake-2",
            mlflow_run_id="m2",
        ),
    )
    text = format_status(tmp_path, "toy", audit=True)
    assert "spend 12.500 / 100.000 DBU" in text
    assert "live nodes" in text
    assert "c1" in text
    assert "c2" in text
    assert "below floor at R1 vs lgbm_reference" in text
    assert "audit PASS" in text
    assert audit_rows(tmp_path, "toy") == []
    assert main(["--run-id", "toy", "--audit", "--root", str(tmp_path)]) == 0


def test_audit_fails_when_the_trace_is_incomplete(tmp_path: Path) -> None:
    append(tmp_path, _row(git_sha="unknown", job_run_id=None, mlflow_run_id=None))
    text = format_status(tmp_path, "toy", audit=True)
    assert "audit FAIL" in text
    assert "missing job_run_id" in text
    assert "missing mlflow_run_id" in text
    assert "missing git_sha" in text
    assert main(["--run-id", "toy", "--audit", "--root", str(tmp_path)]) == 1
