"""Job submission. Fake backend for tests; Databricks Jobs API for the toy run.

Submission is idempotent on `candidate_id/rung/seed`. The Databricks path sends
that key as `idempotency_token` on `runs/run-now`, which returns the original
run id for about 48 hours.
"""

from __future__ import annotations

import json
import os
import subprocess
import time
from pathlib import Path

EXPERIMENT_NAME = "/Shared/amt/nfy_article_rerank"


def databricks_bin() -> str:
    local = Path.home() / ".local" / "bin" / "databricks"
    if local.is_file():
        return str(local)
    return "databricks"


def _run_json(args: list[str], *, cwd: Path) -> dict:
    completed = subprocess.run(
        args,
        cwd=cwd,
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        detail = (completed.stderr or completed.stdout or "databricks command failed").strip()
        raise RuntimeError(detail)
    text = completed.stdout.strip()
    if not text:
        return {}
    return json.loads(text)


class FakeJobs:
    def __init__(self, root: Path) -> None:
        self.root = root
        self.path = root / ".amt" / "fake_jobs.json"

    def _load(self) -> dict:
        if not self.path.is_file():
            return {"submits": {}, "new_submit_count": 0}
        return json.loads(self.path.read_text())

    def _save(self, payload: dict) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(payload, indent=2))

    def submit(self, *, job_key: str, params: dict, idempotency_token: str) -> str:
        payload = self._load()
        existing = payload["submits"].get(idempotency_token)
        if existing:
            return existing["job_run_id"]
        payload["new_submit_count"] = int(payload["new_submit_count"]) + 1
        job_run_id = f"fake-{payload['new_submit_count']}"
        mlflow_run_id = f"fake-mlflow-{payload['new_submit_count']}"
        payload["submits"][idempotency_token] = {
            "job_key": job_key,
            "job_run_id": job_run_id,
            "mlflow_run_id": mlflow_run_id,
            "params": params,
            "state": "TERMINATED",
            "result": "SUCCESS",
        }
        self._save(payload)
        return job_run_id

    def wait(self, job_run_id: str) -> dict:
        payload = self._load()
        for record in payload["submits"].values():
            if record["job_run_id"] == job_run_id:
                return {"life_cycle_state": "TERMINATED", "result_state": "SUCCESS"}
        raise KeyError(job_run_id)

    def find_mlflow_run(self, *, tags: dict) -> dict | None:
        payload = self._load()
        for record in payload["submits"].values():
            params = record["params"]
            if all(str(params.get(key)) == str(value) for key, value in tags.items()):
                return {
                    "mlflow_run_id": record["mlflow_run_id"],
                    "metric": 0.5,
                    "tags": params,
                }
        return None

    def submit_count(self) -> int:
        return int(self._load()["new_submit_count"])


class DatabricksJobs:
    def __init__(self, root: Path) -> None:
        self.root = root

    def submit(self, *, job_key: str, params: dict, idempotency_token: str) -> str:
        del job_key
        job_id = resolve_job_id(self.root)
        body = {
            "job_id": int(job_id),
            "idempotency_token": idempotency_token,
            "job_parameters": {key: str(value) for key, value in params.items()},
        }
        response = _run_json(
            [databricks_bin(), "api", "post", "/api/2.1/jobs/run-now", "--json", json.dumps(body)],
            cwd=self.root,
        )
        run_id = response.get("run_id")
        if run_id is None:
            raise RuntimeError(f"run-now returned no run_id: {response}")
        return str(run_id)

    def wait(self, job_run_id: str, timeout_s: int = 1500) -> dict:
        deadline = time.time() + timeout_s
        terminal = {"TERMINATED", "SKIPPED", "INTERNAL_ERROR"}
        while time.time() < deadline:
            response = _run_json(
                [
                    databricks_bin(),
                    "api",
                    "get",
                    f"/api/2.1/jobs/runs/get?run_id={job_run_id}",
                ],
                cwd=self.root,
            )
            state = response.get("state") or {}
            life = state.get("life_cycle_state")
            if life in terminal:
                return {
                    "life_cycle_state": life,
                    "result_state": state.get("result_state"),
                    "state_message": state.get("state_message"),
                }
            time.sleep(10)
        raise TimeoutError(f"job run {job_run_id} did not finish within {timeout_s}s")

    def find_mlflow_run(self, *, tags: dict) -> dict | None:
        experiment_id = experiment_id_by_name(self.root)
        clauses = [f"tags.{key} = '{value}'" for key, value in tags.items()]
        body = {
            "experiment_ids": [experiment_id],
            "filter": " and ".join(clauses),
            "max_results": 5,
            "order_by": ["attributes.start_time DESC"],
        }
        response = _run_json(
            [
                databricks_bin(),
                "api",
                "post",
                "/api/2.0/mlflow/runs/search",
                "--json",
                json.dumps(body),
            ],
            cwd=self.root,
        )
        runs = response.get("runs") or []
        if not runs:
            return None
        info = runs[0].get("info") or {}
        data = runs[0].get("data") or {}
        metrics = {item["key"]: item["value"] for item in data.get("metrics", [])}
        tags = {item["key"]: item["value"] for item in data.get("tags", [])}
        return {
            "mlflow_run_id": info.get("run_id"),
            "metric": metrics.get("ndcg_at_10"),
            "tags": tags,
        }


def make_backend(root: Path) -> FakeJobs | DatabricksJobs:
    if os.environ.get("AMT_JOBS_BACKEND", "databricks").strip().lower() == "fake":
        return FakeJobs(root)
    return DatabricksJobs(root)


def resolve_job_id(root: Path) -> str:
    override = os.environ.get("AMT_HELLO_TRAIN_JOB_ID")
    if override:
        return override
    cache = root / ".amt" / "hello_train_job_id"
    if cache.is_file():
        return cache.read_text().strip()
    deploy_hello_train(root)
    summary = _run_json(
        [databricks_bin(), "bundle", "summary", "-o", "json", "-t", "dev"],
        cwd=root,
    )
    job_id = _dig_job_id(summary)
    if not job_id:
        raise RuntimeError("bundle summary did not include hello_train job id")
    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_text(job_id)
    return job_id


def deploy_hello_train(root: Path) -> None:
    completed = subprocess.run(
        [databricks_bin(), "bundle", "deploy", "-t", "dev", "--auto-approve"],
        cwd=root,
        check=False,
        text=True,
    )
    if completed.returncode != 0:
        raise RuntimeError("bundle deploy failed")


def experiment_id_by_name(root: Path, name: str = EXPERIMENT_NAME) -> str:
    response = _run_json(
        [databricks_bin(), "experiments", "get-by-name", name, "-o", "json"],
        cwd=root,
    )
    experiment = response.get("experiment") or response
    experiment_id = experiment.get("experiment_id")
    if not experiment_id:
        raise RuntimeError(f"experiment {name} has no id")
    return str(experiment_id)


def _dig_job_id(summary: dict) -> str | None:
    jobs = (summary.get("resources") or {}).get("jobs") or summary.get("jobs") or {}
    if isinstance(jobs, dict):
        hello = jobs.get("hello_train") or {}
        if isinstance(hello, dict):
            for key in ("id", "job_id"):
                if hello.get(key):
                    return str(hello[key])
            url = str(hello.get("url") or "")
            if "/jobs/" in url:
                return url.rstrip("/").split("/jobs/")[-1].split("/")[0].split("?")[0]
    return None
