"""Job submission. Fake backend for tests; Databricks Jobs API otherwise.

Submission is idempotent on the caller's token. ``runs/run-now`` and
``runs/submit`` both honour ``idempotency_token`` for about 48 hours.
``job`` selects the bundle job (``train``, ``confirm``, ``final_holdout``,
``build_dataset``, ``hello_train``). The state key stays ``job_key``.
"""

from __future__ import annotations

import json
import os
import subprocess
import time
from pathlib import Path

import yaml

from amt.config import load_settings

EXPERIMENT_NAME = "/Shared/amt/nfy_article_rerank"
JOB_NAMES = ("hello_train", "train", "confirm", "final_holdout", "build_dataset")
REPO = Path(__file__).resolve().parents[1]


def databricks_bin() -> str:
    local = Path.home() / ".local" / "bin" / "databricks"
    if local.is_file():
        return str(local)
    return "databricks"


def declared_parameters(job: str) -> set[str]:
    """Parameter names from ``resources/<job>.job.yml``. Empty when the file is absent."""

    path = REPO / "resources" / f"{job}.job.yml"
    if not path.is_file():
        return set()
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    job_def = (payload.get("resources") or {}).get("jobs", {}).get(job) or {}
    return {str(item["name"]) for item in job_def.get("parameters") or []}


def _check_params(job: str, params: dict) -> None:
    allowed = declared_parameters(job)
    if not allowed:
        return
    extra = sorted(set(params) - allowed)
    if extra:
        raise ValueError(f"{job} does not declare parameters {extra}")


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
            return {"submits": {}, "new_submit_count": 0, "cells": []}
        payload = json.loads(self.path.read_text())
        payload.setdefault("cells", [])
        return payload

    def _save(self, payload: dict) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(payload, indent=2))

    def submit(
        self,
        *,
        job: str = "hello_train",
        job_key: str,
        params: dict,
        idempotency_token: str,
    ) -> str:
        _check_params(job, params)
        payload = self._load()
        existing = payload["submits"].get(idempotency_token)
        if existing:
            return existing["job_run_id"]
        payload["new_submit_count"] = int(payload["new_submit_count"]) + 1
        job_run_id = f"fake-{payload['new_submit_count']}"
        mlflow_run_id = f"fake-mlflow-{payload['new_submit_count']}"
        payload["submits"][idempotency_token] = {
            "job": job,
            "job_key": job_key,
            "job_run_id": job_run_id,
            "mlflow_run_id": mlflow_run_id,
            "params": params,
            "state": "TERMINATED",
            "result": "SUCCESS",
        }
        _expand_matrix(self.root, payload, params, job_run_id)
        self._save(payload)
        return job_run_id

    def wait(self, job_run_id: str, timeout_s: int | None = None) -> dict:
        del timeout_s
        payload = self._load()
        for record in payload["submits"].values():
            if record["job_run_id"] == job_run_id:
                return {"life_cycle_state": "TERMINATED", "result_state": "SUCCESS"}
        raise KeyError(job_run_id)

    def find_mlflow_run(self, *, tags: dict) -> dict | None:
        found = self.search_runs(tags=tags)
        return found[0] if found else None

    def search_runs(self, *, tags: dict) -> list[dict]:
        payload = self._load()
        metrics_table = _fake_metrics(self.root)
        matches: list[dict] = []
        for record in payload["submits"].values():
            params = record["params"]
            if _tags_match(params, tags):
                matches.append(_fake_found(record, params, metrics_table))
        for cell in payload["cells"]:
            if _tags_match(cell, tags):
                matches.append(dict(cell))
        return matches

    def submit_count(self) -> int:
        return int(self._load()["new_submit_count"])

    def submitted_jobs(self) -> list[str]:
        payload = self._load()
        return [str(record.get("job") or "") for record in payload["submits"].values()]


class DatabricksJobs:
    def __init__(self, root: Path) -> None:
        self.root = root
        self.settings = load_settings(root)

    def submit(
        self,
        *,
        job: str = "hello_train",
        job_key: str,
        params: dict,
        idempotency_token: str,
    ) -> str:
        del job_key
        _check_params(job, params)
        params = _bundle_relative_spec(self.root, params)
        from amt.deploy import assert_fresh

        assert_fresh(self.root)
        self._ensure_cluster()
        if self.settings.fixed_cluster or _gpu_existing(params, self.settings):
            return self._submit_existing(job, params, idempotency_token)
        job_id = resolve_job_ids(self.root)[job]
        body = {
            "job_id": int(job_id),
            "idempotency_token": idempotency_token,
            "job_parameters": {
                key: "" if value is None else str(value) for key, value in params.items()
            },
        }
        response = _run_json(
            [databricks_bin(), "api", "post", "/api/2.1/jobs/run-now", "--json", json.dumps(body)],
            cwd=self.root,
        )
        run_id = response.get("run_id")
        if run_id is None:
            raise RuntimeError(f"run-now returned no run_id: {response}")
        return str(run_id)

    def _submit_existing(self, job: str, params: dict, idempotency_token: str) -> str:
        summary = _bundle_summary(self.root)
        workspace = _workspace_path(summary)
        body = existing_cluster_payload(
            job,
            params,
            workspace_path=workspace,
            cluster_id=_cluster_for(params, self.settings),
            idempotency_token=idempotency_token,
            settings=self.settings,
        )
        response = _run_json(
            [
                databricks_bin(),
                "api",
                "post",
                "/api/2.1/jobs/runs/submit",
                "--json",
                json.dumps(body),
            ],
            cwd=self.root,
        )
        run_id = response.get("run_id")
        if run_id is None:
            raise RuntimeError(f"runs/submit returned no run_id: {response}")
        return str(run_id)

    def _ensure_cluster(self) -> None:
        """Start the fixed cluster if needed and return only once it is RUNNING."""

        cluster_id = self.settings.cluster_id
        if not cluster_id:
            return
        deadline = time.time() + 1800
        started = False
        while time.time() < deadline:
            response = _run_json(
                [databricks_bin(), "api", "get", f"/api/2.0/clusters/get?cluster_id={cluster_id}"],
                cwd=self.root,
            )
            state = str(response.get("state") or "")
            if state == "RUNNING":
                return
            if state in {"PENDING", "RESTARTING", "RESIZING"}:
                time.sleep(15)
                continue
            if state in {"TERMINATING"}:
                time.sleep(15)
                continue
            if not self.settings.cluster_autostart:
                raise RuntimeError(f"cluster is {state or 'unknown'}")
            if not started:
                _run_json(
                    [
                        databricks_bin(),
                        "api",
                        "post",
                        "/api/2.0/clusters/start",
                        "--json",
                        json.dumps({"cluster_id": cluster_id}),
                    ],
                    cwd=self.root,
                )
                started = True
            time.sleep(15)
        raise RuntimeError("cluster did not reach RUNNING")

    def wait(self, job_run_id: str, timeout_s: int | None = None) -> dict:
        timeout_s = self.settings.job_wait_timeout_s if timeout_s is None else timeout_s
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
        found = self.search_runs(tags=tags)
        return found[0] if found else None

    def search_runs(self, *, tags: dict) -> list[dict]:
        experiment_id = experiment_id_by_name(self.root, _experiment_from_tags(tags))
        clauses = [f"tags.{key} = '{value}'" for key, value in tags.items() if value != ""]
        body = {
            "experiment_ids": [experiment_id],
            "filter": " and ".join(clauses) if clauses else "",
            "max_results": 50,
            "order_by": ["attributes.start_time DESC"],
        }
        if not body["filter"]:
            del body["filter"]
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
        found: list[dict] = []
        for run in response.get("runs") or []:
            info = run.get("info") or {}
            data = run.get("data") or {}
            metrics = {item["key"]: item["value"] for item in data.get("metrics", [])}
            run_tags = {item["key"]: item["value"] for item in data.get("tags", [])}
            found.append(_metric_row(info.get("run_id"), metrics, run_tags))
        return found


def _bundle_relative_spec(root: Path, params: dict) -> dict:
    """Jobs resolve ``spec`` against the deployed bundle, so a local absolute path
    under the repo is sent relative to it."""

    spec = str(params.get("spec") or "")
    if not spec.startswith("/"):
        return params
    try:
        relative = Path(spec).resolve().relative_to(Path(root).resolve())
    except ValueError:
        return params
    return {**params, "spec": relative.as_posix()}


def _gpu_existing(params: dict, settings) -> bool:
    return str(params.get("device") or "") == "gpu" and bool(settings.gpu_cluster_id)


def _cluster_for(params: dict, settings) -> str:
    if str(params.get("device") or "") == "gpu" and settings.gpu_cluster_id:
        return settings.gpu_cluster_id
    if not settings.cluster_id:
        raise RuntimeError("fixed-cluster submit requires AMT_CLUSTER_ID or AMT_GPU_CLUSTER_ID")
    return settings.cluster_id


def existing_cluster_payload(
    job: str,
    params: dict,
    *,
    workspace_path: str,
    cluster_id: str,
    idempotency_token: str,
    settings,
) -> dict:
    """One-time run on an existing cluster. No ``new_cluster`` and no wheel library."""

    path = REPO / "resources" / f"{job}.job.yml"
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    job_def = payload["resources"]["jobs"][job]
    task = job_def["tasks"][0]
    python_rel = str(task["spark_python_task"]["python_file"])
    if python_rel.startswith("../"):
        python_rel = python_rel[3:]
    remote_python = workspace_path.rstrip("/") + "/" + python_rel
    defaults = {
        item["name"]: str(item.get("default") or "") for item in job_def.get("parameters") or []
    }
    rendered: list[str] = []
    for token in task["spark_python_task"].get("parameters") or []:
        text = str(token)
        if text.startswith("{{job.parameters.") and text.endswith("}}"):
            name = text[len("{{job.parameters.") : -2]
            rendered.append(str(params.get(name, defaults.get(name, ""))))
        else:
            rendered.append(text)
    if settings.fixed_cluster or params.get("device") == "gpu":
        if "--root" not in rendered:
            rendered.extend(["--root", workspace_path])
        else:
            index = rendered.index("--root") + 1
            if index < len(rendered) and not rendered[index]:
                rendered[index] = workspace_path
    return {
        "run_name": f"amt_{job}",
        "idempotency_token": idempotency_token,
        "tasks": [
            {
                "task_key": task["task_key"],
                "existing_cluster_id": cluster_id,
                "spark_python_task": {"python_file": remote_python, "parameters": rendered},
            }
        ],
    }


def _expand_matrix(root: Path, payload: dict, params: dict, job_run_id: str) -> None:
    matrix = params.get("matrix")
    if not matrix:
        return
    path = Path(str(matrix))
    if not path.is_file():
        return
    cells = json.loads(path.read_text(encoding="utf-8"))
    metrics_table = _fake_metrics(root)
    for cell in cells:
        if not isinstance(cell, dict):
            continue
        merged = dict(params)
        merged.update({key: str(value) for key, value in cell.items()})
        cell_id = merged.get("candidate_id", "cell")
        merged["mlflow_run_id"] = f"{job_run_id}-{cell_id}-{merged.get('seed', '0')}"
        base = metrics_table.get(str(merged.get("candidate_id"))) or metrics_table.get("*") or {}
        metric = float(base.get("ndcg@10_graded", 0.42))
        record = {
            "mlflow_run_id": merged["mlflow_run_id"],
            "metric": metric,
            "tags": merged,
            "run_id": merged.get("run_id") or merged.get("batch"),
            "node_id": merged.get("node_id") or merged.get("candidate_id"),
            "candidate_id": merged.get("candidate_id"),
            "rung": merged.get("rung"),
            "seed": str(merged.get("seed")),
            "batch": merged.get("batch"),
            "split": merged.get("split") or "search_val",
            "delta_point": base.get("delta_point", 0.01),
            "delta_ci_lo": base.get("delta_ci_lo", 0.0),
            "delta_ci_hi": base.get("delta_ci_hi", 0.02),
            "delta_lgbm_point": base.get("delta_lgbm_point", 0.0),
            "delta_lgbm_ci_lo": base.get("delta_lgbm_ci_lo", -0.01),
            "delta_lgbm_ci_hi": base.get("delta_lgbm_ci_hi", 0.01),
            "latency_p95_ms": base.get("latency_p95_ms", 5.0),
        }
        payload["cells"].append(record)


def _tags_match(params: dict, tags: dict) -> bool:
    for key, value in tags.items():
        if str(params.get(key, "")) != str(value):
            return False
    return True


def _fake_found(record: dict, params: dict, metrics_table: dict) -> dict:
    metrics = dict(metrics_table.get("*") or {})
    metrics.update(metrics_table.get(str(params.get("candidate_id"))) or {})
    found = {
        "mlflow_run_id": record["mlflow_run_id"],
        "metric": metrics.get("ndcg@10_graded", metrics.get("ndcg_at_10", 0.5)),
        "tags": params,
    }
    for key in (
        "delta_point",
        "delta_ci_lo",
        "delta_ci_hi",
        "delta_lgbm_point",
        "delta_lgbm_ci_lo",
        "delta_lgbm_ci_hi",
        "latency_p95_ms",
    ):
        if key in metrics:
            found[key] = metrics[key]
    return found


def _metric_row(mlflow_run_id: str | None, metrics: dict, tags: dict) -> dict:
    metric = metrics.get("ndcg@10_graded", metrics.get("ndcg_at_10"))
    found = {"mlflow_run_id": mlflow_run_id, "metric": metric, "tags": tags}
    for key in (
        "delta_point",
        "delta_ci_lo",
        "delta_ci_hi",
        "delta_lgbm_point",
        "delta_lgbm_ci_lo",
        "delta_lgbm_ci_hi",
        "latency_p95_ms",
    ):
        if key in metrics:
            found[key] = metrics[key]
    return found


def _experiment_from_tags(tags: dict) -> str:
    return str(tags.get("experiment") or EXPERIMENT_NAME)


def _fake_metrics(root: Path) -> dict:
    path = root / ".amt" / "fake_metrics.json"
    if not path.is_file():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        return {}
    return payload


def make_backend(root: Path) -> FakeJobs | DatabricksJobs:
    if os.environ.get("AMT_JOBS_BACKEND", "databricks").strip().lower() == "fake":
        return FakeJobs(root)
    return DatabricksJobs(root)


def resolve_job_ids(root: Path) -> dict[str, str]:
    """Bundle job ids. ``AMT_<JOB>_JOB_ID`` overrides the summary."""

    found: dict[str, str] = {}
    cache = root / ".amt" / "job_ids.json"
    if cache.is_file():
        found.update(json.loads(cache.read_text(encoding="utf-8")))
    else:
        summary = _bundle_summary(root)
        for name in JOB_NAMES:
            job_id = _dig_named_job(summary, name)
            if job_id:
                found[name] = job_id
        workspace = _workspace_path(summary)
        if workspace:
            found["workspace_path"] = workspace
        cache.parent.mkdir(parents=True, exist_ok=True)
        cache.write_text(json.dumps(found, indent=2) + "\n", encoding="utf-8")
    for name in JOB_NAMES:
        override = os.environ.get(f"AMT_{name.upper()}_JOB_ID", "").strip()
        if override:
            found[name] = override
    missing = [name for name in JOB_NAMES if name not in found]
    if missing:
        raise RuntimeError(
            f"bundle summary did not include {', '.join(missing)}; run amt deploy"
        )
    return found


def resolve_job_id(root: Path) -> str:
    """Walking-skeleton helper. Prefer ``resolve_job_ids``."""

    return resolve_job_ids(root)["hello_train"]


def _bundle_summary(root: Path) -> dict:
    target = load_settings(root).bundle_target
    return _run_json(
        [databricks_bin(), "bundle", "summary", "-o", "json", "-t", target],
        cwd=root,
    )


def _workspace_path(summary: dict) -> str:
    workspace = summary.get("workspace") or {}
    return str(workspace.get("file_path") or "")


def _dig_named_job(summary: dict, name: str) -> str | None:
    jobs = (summary.get("resources") or {}).get("jobs") or summary.get("jobs") or {}
    if not isinstance(jobs, dict):
        return None
    hello = jobs.get(name) or {}
    if not isinstance(hello, dict):
        return None
    for key in ("id", "job_id"):
        if hello.get(key):
            return str(hello[key])
    url = str(hello.get("url") or "")
    if "/jobs/" in url:
        return url.rstrip("/").split("/jobs/")[-1].split("/")[0].split("?")[0]
    return None


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
