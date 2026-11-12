"""Submit and collect the round-0 and ladder matrices.

Each rung is one ``train`` job with ``--matrix``, so the cluster starts once
per rung. Resubmitting a rung that already has a run id does nothing.
"""

from __future__ import annotations

import json
from pathlib import Path

from amt import volume
from amt.config import load_settings
from amt.deploy import _git_sha, tree_hash
from amt.jobs_backend import make_backend
from amt.ladder import build_ladder_report, ladder_jobs
from amt.noise import build_noise_report, round0_jobs
from amt.schemas import Brief
from amt.scores import scores_path
from amt.signoff import write_ladder_report, write_noise_report


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
) -> tuple[str, Path]:
    run_name = f"ladder-{dataset}"
    rows = _search(root, run_name)
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
    written = write_ladder_report(root, report)
    return report.status, written


def submit_dataset(root: Path, spec: str) -> str:
    settings = load_settings(root)
    backend = make_backend(root)
    token = f"build:{spec}:{tree_hash(root)[:12]}"
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
    }
    run_id = backend.submit(
        job="build_dataset",
        job_key=token,
        params=params,
        idempotency_token=token,
    )
    state[token] = run_id
    _save_state(root, f"build-{Path(spec).name}", state)
    return run_id


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
    tree = tree_hash(root)[:12]
    for rung, group in by_rung.items():
        token = f"{run_name}:{rung}:{tree}"
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
            matrix_path = settings.volume("matrices", run_name, f"{tree}-{rung}.json")
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
    return {
        "candidate_id": "matrix",
        "node_id": "matrix",
        "parent_id": "",
        "family": "matrix",
        "operator": "draft",
        "rung": rung,
        "seed": "0",
        "split": "search_val",
        "batch": run_name,
        "spec": spec_path,
        "baseline": "",
        "reference_score": "production",
        "paired_reference": "production",
        "model": "",
        "catalog": settings.catalog,
        "schema": settings.schema_name,
        "dataset_dir": "",
        "experiment_name": settings.experiment_for(brief_name),
        "git_sha": git_sha,
        "run_id": run_name,
        "baseline_scores": "",
        "write_scores": "",
        "root": "",
        "shared_cluster": "true" if settings.fixed_cluster else "",
        "matrix": matrix,
        "device": "cpu",
    }


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


def _load_state(root: Path, run_name: str) -> dict:
    path = root / "runs" / run_name / "state.json"
    if not path.is_file():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _save_state(root: Path, run_name: str, state: dict) -> None:
    path = root / "runs" / run_name / "state.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(state, indent=2) + "\n", encoding="utf-8")
