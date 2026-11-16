"""Submit and collect the round-0 and ladder matrices.

Each rung is one ``train`` job with ``--matrix``, so the cluster starts once
per rung. Resubmitting a rung that already has a run id does nothing.
"""

from __future__ import annotations

import json
from pathlib import Path

from amt import volume
from amt.config import load_settings
from amt.deploy import _git_sha, freshness_token
from amt.gates import write_ladder_report, write_noise_report
from amt.job_params import build_job_params
from amt.jobs_backend import FakeJobs, make_backend
from amt.ladder import build_ladder_report, ladder_jobs
from amt.noise import build_noise_report, round0_jobs
from amt.schemas import Brief
from amt.scores import scores_path
from amt.volume import VolumeError
from harness.ladder import LADDER_DESIGNS


def submit_noise(root: Path, brief: Brief, spec_path: str, reference_scores: list[str]) -> Path:
    jobs = round0_jobs(reference_scores)
    run_name = f"round0-{brief.name or 'brief'}"
    return _submit_matrix(
        root,
        run_name=run_name,
        spec_path=spec_path,
        brief_name=brief.name or "brief",
        jobs=jobs,
        write_for={"lgbm_reference"},
        spec_name=Path(spec_path).name,
    )


def collect_noise(
    root: Path,
    brief: Brief,
    brief_path: Path,
    dataset: str,
    reference_scores: list[str],
) -> tuple[str, Path]:
    run_name = f"round0-{brief.name or 'brief'}"
    rows = _collect(root, run_name)
    cells = [_noise_cell(row) for row in rows]
    path = root / "runs" / run_name / "round0.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(cells, indent=2) + "\n", encoding="utf-8")
    from amt.noise import load_cells

    report = build_noise_report(
        load_cells(cells),
        brief,
        dataset=dataset,
        reference_scores=reference_scores,
    )
    written = write_noise_report(root, brief, brief_path, report)
    return report.status, written


def submit_ladder(root: Path, dataset: str, spec_path: str) -> Path:
    return _submit_matrix(
        root,
        run_name=f"ladder-{dataset}",
        spec_path=spec_path,
        brief_name=dataset,
        jobs=ladder_jobs(),
        write_for=set(),
        spec_name=dataset,
    )


def collect_ladder(
    root: Path,
    dataset: str,
    *,
    skip_r1: list[str],
    r1_train_fraction: float | None,
    report_name: str | None = None,
) -> tuple[str, Path]:
    run_name = f"ladder-{dataset}"
    current = {design.design_id for design in LADDER_DESIGNS}
    rows = [row for row in _search(root, run_name) if _cell_id(row.get("tags") or {}) in current]
    cells = [_ladder_cell(row) for row in _newest(rows, scored=True)]
    path = root / "runs" / run_name / "ladder.json"
    path.write_text(json.dumps(cells, indent=2) + "\n", encoding="utf-8")
    from amt.ladder import load_cells

    report = build_ladder_report(
        load_cells(cells),
        dataset=dataset,
        skip_r1=skip_r1,
        r1_train_fraction=r1_train_fraction,
        unsupported=_unsupported_families(_newest(rows, scored=False)),
    )
    written = write_ladder_report(root, report, name=report_name)
    return report.status, written


def submit_dataset(
    root: Path,
    spec: str,
    *,
    preflight_slice: str = "",
    positive_grades: str = "",
) -> str:
    settings = load_settings(root)
    backend = make_backend(root)
    mode = preflight_slice or "full"
    token = f"build:{spec}:{freshness_token(root, Path(spec).name)}:{mode}"
    state = _load_state(root, f"build-{Path(spec).name}")
    cached = state.get(token)
    if cached and _run_is_reusable(root, str(cached)):
        return str(cached)
    if cached:
        token = f"{token}:retry"
    params = {
        "spec": spec,
        "catalog": settings.catalog,
        "schema": settings.schema_name,
        "output_dir": "",
        "preflight_slice": preflight_slice,
        "positive_grades": positive_grades,
    }
    run_id = backend.submit(
        job="build_dataset",
        job_key=token,
        params=params,
        idempotency_token=token,
    )
    state[token] = run_id
    _save_state(root, f"build-{Path(spec).name}", state)
    spec_name, spec_hash = _spec_identity(root, spec)
    _remember_run(
        root,
        run_id,
        {
            "spec": spec,
            "spec_name": spec_name,
            "spec_hash": spec_hash,
            "catalog": settings.catalog,
            "schema": settings.schema_name,
            "preflight_slice": preflight_slice,
            "positive_grades": positive_grades,
        },
    )
    return run_id


def fetch_job_failure(root: Path, run_id: str) -> dict:
    """Read ``failure.json`` for a build run, or the Jobs get-output error."""

    direct = _read_failure_file(root, run_id)
    if direct is not None:
        return direct
    return _failure_from_output(root, run_id)


def wait_job(root: Path, job_run_id: str) -> dict:
    return make_backend(root).wait(job_run_id)


def _run_is_reusable(root: Path, run_id: str) -> bool:
    """A cached run id is reused only while it is still going or already succeeded."""

    from amt.jobs_backend import DatabricksJobs, _run_json, databricks_bin

    backend = make_backend(root)
    if not isinstance(backend, DatabricksJobs):
        return True
    response = _run_json(
        [databricks_bin(), "api", "get", f"/api/2.1/jobs/runs/get?run_id={run_id}"],
        cwd=root,
    )
    state = response.get("state") or {}
    life = str(state.get("life_cycle_state") or "")
    if life in {"PENDING", "RUNNING", "BLOCKED", "QUEUED", "WAITING", "TERMINATING"}:
        return True
    return life == "TERMINATED" and state.get("result_state") == "SUCCESS"


def _submit_matrix(
    root: Path,
    *,
    run_name: str,
    spec_path: str,
    brief_name: str,
    jobs: list,
    write_for: set[str],
    spec_name: str,
) -> Path:
    settings = load_settings(root)
    backend = make_backend(root)
    state = _load_state(root, run_name)
    by_rung: dict[str, list] = {}
    for job in jobs:
        by_rung.setdefault(job.rung, []).append(job)
    directory = root / "runs" / run_name
    directory.mkdir(parents=True, exist_ok=True)
    stamp = freshness_token(root, spec_name)
    for rung, group in by_rung.items():
        token = f"{run_name}:{rung}:{stamp}"
        cached = state.get(token)
        if cached and _run_is_reusable(root, str(cached)):
            continue
        idempotency_token = f"{token}:retry-{cached}" if cached else token
        cells = []
        for job in group:
            cell = job.params()
            cell.setdefault("node_id", job.candidate_id)
            if job.candidate_id in write_for:
                cell["write_scores"] = scores_path(
                    root, spec_name, job.candidate_id, job.rung, job.seed, settings=settings
                )
            cells.append(cell)
        matrix = directory / f"matrix-{rung}.json"
        matrix.write_text(json.dumps(cells, indent=2) + "\n", encoding="utf-8")
        matrix_path = str(matrix)
        if settings.jobs_backend != "fake":
            matrix_path = settings.volume("matrices", run_name, f"{stamp}-{rung}.json")
            volume.put(root, matrix, matrix_path, overwrite=True)
        params = _base_params(
            settings,
            spec_path=spec_path,
            brief_name=brief_name,
            run_name=run_name,
            rung=rung,
            matrix=matrix_path,
            git_sha=_git_sha(root),
        )
        run_id = backend.submit(
            job="train",
            job_key=token,
            params=params,
            idempotency_token=idempotency_token,
        )
        state[token] = run_id
        _save_state(root, run_name, state)
    return directory / "state.json"


def _base_params(
    settings,
    *,
    spec_path: str,
    brief_name: str,
    run_name: str,
    rung: str,
    matrix: str,
    git_sha: str,
) -> dict[str, str]:
    return build_job_params(
        "train",
        candidate_id="matrix",
        node_id="matrix",
        parent_id="",
        family="matrix",
        operator="draft",
        rung=rung,
        seed="0",
        split="search_val",
        batch=run_name,
        spec=spec_path,
        baseline="",
        reference_score="production",
        paired_reference="production",
        model="",
        catalog=settings.catalog,
        schema=settings.schema_name,
        dataset_dir="",
        experiment_name=settings.experiment_for(brief_name),
        git_sha=git_sha,
        run_id=run_name,
        baseline_scores="",
        write_scores="",
        root="",
        shared_cluster="true" if settings.fixed_cluster else "",
        matrix=matrix,
        device="cpu",
    )


def _collect(root: Path, run_name: str) -> list[dict]:
    return _newest(_search(root, run_name), scored=True)


def _search(root: Path, run_name: str) -> list[dict]:
    backend = make_backend(root)
    state = _load_state(root, run_name)
    for run_id in state.values():
        backend.wait(str(run_id))
    return backend.search_runs(tags={"batch": run_name})  # type: ignore[attr-defined]


def _newest(rows: list[dict], *, scored: bool) -> list[dict]:
    """A resubmitted rung logs every cell again, and a failed cell logs no metric:
    keep the newest run per cell, only among scored runs when ``scored``."""

    latest: dict[tuple[str, str, str], dict] = {}
    for row in rows:
        tags = row.get("tags") or {}
        if _cell_id(tags) == "matrix" or (scored and row.get("metric") is None):
            continue
        key = _cell_key(row)
        kept = latest.get(key)
        if kept is None or int(row.get("start_time") or 0) >= int(kept.get("start_time") or 0):
            latest[key] = row
    return list(latest.values())


def _cell_key(row: dict) -> tuple[str, str, str]:
    tags = row.get("tags") or {}
    return (_cell_id(tags), str(tags.get("rung")), str(tags.get("seed")))


def _unsupported_families(rows: list[dict]) -> dict[str, str]:
    """Families whose every ladder cell last reported the dataset unsupported."""

    newest = {_cell_key(row): row for row in rows}
    by_family: dict[str, list[dict | None]] = {}
    for job in ladder_jobs():
        key = (job.candidate_id, job.rung, str(job.seed))
        by_family.setdefault(job.family, []).append(newest.get(key))
    found: dict[str, str] = {}
    for family, cells in by_family.items():
        tags = [(row or {}).get("tags") or {} for row in cells]
        if all(tag.get("status") == "unsupported" for tag in tags):
            found[family] = str(tags[0].get("failure_reason") or "unsupported")
    return found


def _cell_id(tags: dict) -> str:
    """Matrix cells logged before each set its own node_id carry the id only in family."""

    ids = [str(tags.get(key) or "") for key in ("candidate_id", "node_id")]
    for value in ids:
        if value and value != "matrix":
            return value
    family = str(tags.get("family") or "").removeprefix("ref.")
    return family or ("matrix" if "matrix" in ids else "")


def _noise_cell(row: dict) -> dict:
    tags = row.get("tags") or row
    cell = {
        "baseline_id": _cell_id(tags),
        "rung": tags.get("rung"),
        "seed": int(tags.get("seed") or 0),
        "split": "search_val",
        "metric": float(row.get("metric") or 0.0),
    }
    if cell["baseline_id"] != "production" and row.get("delta_point") is not None:
        cell["delta_point"] = float(row["delta_point"])
        cell["delta_ci_lo"] = float(row.get("delta_ci_lo") or 0.0)
        cell["delta_ci_hi"] = float(row.get("delta_ci_hi") or 0.0)
    if row.get("latency_p95_ms") is not None:
        cell["latency_p95_ms"] = float(row["latency_p95_ms"])
    return cell


def _ladder_cell(row: dict) -> dict:
    tags = row.get("tags") or row
    return {
        "design_id": _cell_id(tags),
        "rung": tags.get("rung"),
        "seed": int(tags.get("seed") or 0),
        "split": "search_val",
        "metric": float(row.get("metric") or 0.0),
    }


def _run_record_path(root: Path, run_id: str) -> Path:
    return root / "runs" / "jobs" / f"{run_id}.json"


def _remember_run(root: Path, run_id: str, payload: dict) -> None:
    path = _run_record_path(root, run_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def _load_run_record(root: Path, run_id: str) -> dict:
    path = _run_record_path(root, run_id)
    if not path.is_file():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload if isinstance(payload, dict) else {}


def _spec_identity(root: Path, spec: str) -> tuple[str, str]:
    from harness.data.contract import load_spec

    path = Path(spec)
    if not path.is_absolute():
        path = root / path
    if not path.exists():
        return Path(spec).name, ""
    loaded = load_spec(path)
    return loaded.name, loaded.spec_hash()


def _failure_file(root: Path, run_id: str) -> Path | None:
    record = _load_run_record(root, run_id)
    name = str(record.get("spec_name") or "")
    digest = str(record.get("spec_hash") or "")
    catalog = str(record.get("catalog") or "")
    schema = str(record.get("schema") or "")
    if not name or not digest or not catalog or not schema:
        return None
    from harness.data.build_dataset import failure_dir

    return failure_dir(catalog, schema, name, digest) / "failure.json"


def _read_failure_file(root: Path, run_id: str) -> dict | None:
    remote_path = _failure_file(root, run_id)
    if remote_path is None:
        return None
    if remote_path.is_file():
        return json.loads(remote_path.read_text(encoding="utf-8"))
    remote = remote_path.as_posix()
    if not remote.startswith("/Volumes"):
        return None
    downloaded = root / "runs" / "jobs" / f"{run_id}-failure.json"
    try:
        volume.get(root, remote, downloaded)
    except VolumeError:
        return None
    if not downloaded.is_file():
        return None
    return json.loads(downloaded.read_text(encoding="utf-8"))


def _task_run_id(root: Path, run_id: str) -> str:
    """Resolve get-output to a single task. Multi-task parent runs refuse it."""

    from amt.jobs_backend import _run_json, databricks_bin

    response = _run_json(
        [databricks_bin(), "api", "get", f"/api/2.1/jobs/runs/get?run_id={run_id}"],
        cwd=root,
    )
    tasks = response.get("tasks") or []
    if not tasks:
        return run_id
    task_id = tasks[0].get("run_id")
    return str(task_id) if task_id is not None else run_id


def _run_output(root: Path, run_id: str) -> dict:
    backend = make_backend(root)
    if isinstance(backend, FakeJobs):
        path = root / "runs" / "jobs" / f"{run_id}-output.json"
        if not path.is_file():
            return {}
        payload = json.loads(path.read_text(encoding="utf-8"))
        return payload if isinstance(payload, dict) else {}
    from amt.jobs_backend import _run_json, databricks_bin

    output_id = _task_run_id(root, run_id)
    return _run_json(
        [databricks_bin(), "api", "get", f"/api/2.1/jobs/runs/get-output?run_id={output_id}"],
        cwd=root,
    )


def _failure_from_output(root: Path, run_id: str) -> dict:
    response = _run_output(root, run_id)
    error = response.get("error") or ""
    if isinstance(error, dict):
        message = str(error.get("message") or error)
        exc_type = str(error.get("error_class") or error.get("type") or "JobOutput")
    else:
        message = str(error)
        exc_type = "JobOutput"
    if not message:
        message = str(response.get("metadata") or f"no failure.json for {run_id}")
    return {
        "exception_type": exc_type,
        "message": message,
        "traceback": str(response.get("error_trace") or ""),
        "kind": "",
        "examples": [],
        "fragment": "",
        "fragment_path": "",
        "source_schemas": {},
        "source": "jobs/runs/get-output",
    }


def _load_state(root: Path, run_name: str) -> dict:
    path = root / "runs" / run_name / "state.json"
    if not path.is_file():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _save_state(root: Path, run_name: str, state: dict) -> None:
    path = root / "runs" / run_name / "state.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(state, indent=2) + "\n", encoding="utf-8")
