"""Pipeline A: doctor, deploy, preflight, build, modality, noise, and ladder.

``amt prep --brief briefs/X.yaml [--repair]`` writes
``gates/ready/<brief>/ready_report.{json,md}``. Noise and ladder are submitted
together, then collected. ``--repair`` may edit ``datasets/<spec>/features/*.sql``
before the full build. The brief file is not rewritten; the resolved
``min_effect`` is locked later by ``amt sign ready``.
"""

from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from amt.batch import (
    collect_ladder,
    collect_noise,
    submit_dataset,
    submit_ladder,
    submit_noise,
    wait_job,
)
from amt.config import load_settings
from amt.dataset_preflight import run_preflight
from amt.dataset_repair import DatasetRepairError, repair_features
from amt.deploy import DeployError, assert_fresh, deploy, write_deployed
from amt.doctor import cheap
from amt.doctor import report as doctor_report
from amt.gates import (
    ReadyReport,
    SignoffError,
    _spec_path,
    brief_key,
    build_ready_report,
    load_ladder_report,
    load_noise_report,
    ready_dir,
    repair_records,
    write_ready_report,
)
from amt.schemas import Brief, load_brief
from amt.volume import VolumeError
from amt.volume import get as volume_get
from harness.data.contract import load_spec


class PrepError(RuntimeError):
    """Prep stopped before a ready packet could be signed."""


@dataclass
class PrepHooks:
    """Replacements for the remote steps. Tests pass the ones they need to stub."""

    doctor: Callable[[Path], list[str]] | None = None
    deploy_if_stale: Callable[[Path], str] | None = None
    preflight: Callable[[Path, str, Brief], dict] | None = None
    repair: Callable[[Path, str, Brief], dict] | None = None
    build: Callable[[Path, str], str] | None = None
    fetch_manifest: Callable[[Path, Brief, Path, Any], Path] | None = None
    submit_noise: Callable[..., Any] | None = None
    submit_ladder: Callable[..., Any] | None = None
    collect_noise: Callable[..., tuple[str, Path]] | None = None
    collect_ladder: Callable[..., tuple[str, Path]] | None = None


def run_prep(
    root: Path,
    brief_path: Path,
    *,
    repair: bool = False,
    hooks: PrepHooks | None = None,
) -> tuple[ReadyReport, Path]:
    """Run pipeline A and return the ready report plus its JSON path."""

    hooks = hooks or PrepHooks()
    brief = load_brief(brief_path)
    spec_path = _spec_path(root, brief)
    spec = load_spec(spec_path)
    spec_text = str(spec_path)

    problems = (hooks.doctor or _doctor)(root)
    if problems:
        raise PrepError("; ".join(problems))
    (hooks.deploy_if_stale or _deploy_if_stale)(root)

    try:
        if repair:
            outcome = (hooks.repair or _repair)(root, spec_text, brief)
        else:
            outcome = (hooks.preflight or _preflight)(root, spec_text, brief)
    except DatasetRepairError as exc:
        detail = exc.failure.get("message") if isinstance(exc.failure, dict) else str(exc)
        raise PrepError(str(detail or exc)) from exc
    except SignoffError as exc:
        raise PrepError(str(exc)) from exc
    if outcome.get("status") != "pass":
        failure = outcome.get("failure") or {}
        message = failure.get("message") if isinstance(failure, dict) else ""
        raise PrepError(message or "dataset preflight failed")

    (hooks.build or _build)(root, spec_text)
    manifest_path = (hooks.fetch_manifest or _fetch_manifest)(root, brief, brief_path, spec)
    _submit_round0(root, brief, spec_text, spec, hooks)
    _collect_round0(root, brief, brief_path, spec, hooks)

    name = brief_key(brief, brief_path)
    noise, _noise_sha = load_noise_report(root, brief, brief_path)
    ladder, _ladder_sha = load_ladder_report(root, spec.name, name=name)
    report = build_ready_report(
        root,
        brief,
        brief_path,
        noise=noise,
        ladder=ladder,
        manifest_path=manifest_path,
        repairs=repair_records(root, spec.name),
    )
    path = write_ready_report(root, name, report)
    return report, path


def _doctor(root: Path) -> list[str]:
    doctor_report(root)
    return cheap()


def _deploy_if_stale(root: Path) -> str:
    try:
        assert_fresh(root)
    except DeployError:
        if load_settings(root).jobs_backend == "fake":
            write_deployed(root, git_sha="fake")
            return "recorded"
        deploy(root)
        return "deployed"
    return "fresh"


def _preflight(root: Path, spec_text: str, brief: Brief) -> dict:
    return run_preflight(root, spec_text, brief)


def _repair(root: Path, spec_text: str, brief: Brief) -> dict:
    return repair_features(root, spec_text, brief)


def _build(root: Path, spec_text: str) -> str:
    run_id = submit_dataset(root, spec_text)
    result = wait_job(root, run_id)
    if result.get("result_state") != "SUCCESS":
        raise PrepError(f"dataset build {run_id} finished {result.get('result_state')}")
    return str(run_id)


def _fetch_manifest(root: Path, brief: Brief, brief_path: Path, spec: Any) -> Path:
    local = ready_dir(root, brief_key(brief, brief_path)) / "manifest.json"
    local.parent.mkdir(parents=True, exist_ok=True)
    if load_settings(root).jobs_backend == "fake":
        staged = root / ".amt" / "fake_manifest.json"
        if not staged.is_file():
            raise PrepError("fake prep needs .amt/fake_manifest.json")
        local.write_bytes(staged.read_bytes())
        return local
    settings = load_settings(root)
    remote = settings.volume(spec.name, "manifest.json")
    try:
        volume_get(root, remote, local)
    except VolumeError as exc:
        raise PrepError(str(exc)) from exc
    if not local.is_file():
        raise PrepError(f"manifest download did not write {local}")
    return local


def _submit_round0(root: Path, brief: Brief, spec_text: str, spec: Any, hooks: PrepHooks) -> None:
    noise_fn = hooks.submit_noise or _submit_noise
    ladder_fn = hooks.submit_ladder or _submit_ladder

    def _noise() -> None:
        noise_fn(root, brief, spec_text, list(spec.reference_scores))

    def _ladder() -> None:
        ladder_fn(root, spec.name, spec_text)

    _together(_noise, _ladder)


def _collect_round0(
    root: Path, brief: Brief, brief_path: Path, spec: Any, hooks: PrepHooks
) -> None:
    name = brief_key(brief, brief_path)
    fraction = float(spec.sampling.rungs["R1"].train_fraction)
    noise_fn = hooks.collect_noise or collect_noise
    ladder_fn = hooks.collect_ladder or collect_ladder

    def _noise() -> None:
        noise_fn(root, brief, brief_path, spec.name, list(spec.reference_scores))

    def _ladder() -> None:
        ladder_fn(
            root,
            spec.name,
            skip_r1=[],
            r1_train_fraction=fraction,
            report_name=name,
        )

    _together(_noise, _ladder)


def _submit_noise(root: Path, brief: Brief, spec_text: str, reference_scores: list[str]) -> None:
    submit_noise(root, brief, spec_text, reference_scores)


def _submit_ladder(root: Path, dataset: str, spec_text: str) -> None:
    submit_ladder(root, dataset, spec_text)


def _together(left: Callable[[], None], right: Callable[[], None]) -> None:
    with ThreadPoolExecutor(max_workers=2) as pool:
        pending = (pool.submit(left), pool.submit(right))
        for future in pending:
            try:
                future.result()
            except (PrepError, AssertionError):
                raise
            except Exception as exc:
                raise PrepError(str(exc)) from exc


def manifest_document(path: Path) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise PrepError("manifest is not a JSON object")
    return payload
