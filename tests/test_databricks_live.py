"""One live R0 popularity cell. Skipped unless AMT_LIVE_DATABRICKS=1."""

import os

import pytest

pytestmark = pytest.mark.databricks


@pytest.mark.skipif(
    os.environ.get("AMT_LIVE_DATABRICKS") != "1",
    reason="set AMT_LIVE_DATABRICKS=1",
)
def test_one_popularity_cell_logs_graded_ndcg_and_deltas() -> None:
    from pathlib import Path

    from amt.jobs_backend import DatabricksJobs
    from amt.schemas import MLFLOW_TAG_KEYS

    root = Path(__file__).resolve().parents[1]
    backend = DatabricksJobs(root)
    params = {
        "candidate_id": "popularity",
        "node_id": "popularity",
        "parent_id": "",
        "family": "popularity",
        "operator": "draft",
        "rung": "R0",
        "seed": "0",
        "split": "search_val",
        "batch": "live-smoke",
        "spec": "datasets/nfy_article_rerank_demo",
        "baseline": "popularity",
        "reference_score": "production",
        "paired_reference": "production",
        "model": "",
        "catalog": "product_api_dev",
        "schema": "amt",
        "dataset_dir": "",
        "experiment_name": "/Shared/amt/nfy_demo",
        "git_sha": "live",
        "run_id": "live-smoke",
        "baseline_scores": "",
        "write_scores": "",
        "root": "",
        "shared_cluster": "",
        "matrix": "",
        "device": "cpu",
    }
    run_id = backend.submit(
        job="train",
        job_key="live-smoke",
        params=params,
        idempotency_token="live-smoke",
    )
    result = backend.wait(run_id)
    assert result["result_state"] == "SUCCESS"
    found = backend.find_mlflow_run(
        tags={"run_id": "live-smoke", "node_id": "popularity", "rung": "R0", "seed": "0"}
    )
    assert found is not None
    assert found["metric"] is not None
    assert found.get("delta_point") is not None
    assert found.get("delta_ci_lo") is not None
    assert found.get("delta_ci_hi") is not None
    for key in MLFLOW_TAG_KEYS:
        assert key in (found.get("tags") or {})
