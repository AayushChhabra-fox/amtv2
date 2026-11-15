"""The ready gate: one signature over the prep packet.

``amt prep`` writes ``gates/ready/<brief>/ready_report.{json,md}``.
``amt sign ready`` writes ``ready.json`` in that directory. The signature
locks the problem and task, the spec, the manifest, accepted feature-SQL
repair diffs, the resolved ``min_effect``, the budget, and the viable ladder
families. Optional ``design_space.families`` hints are not part of the hash.
Search briefs do not start until the signature still matches those inputs.
"""

from __future__ import annotations

import hashlib
import json
import re
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from amt.ladder import LadderReport
from amt.modality import ladder_blockers, paradigm_decisions, profile_from_manifest
from amt.noise import NoiseReport, resolved_min_effect
from amt.schemas import Brief, load_brief
from harness.data.contract import load_spec
from harness.ladder import ladder_families

_RESERVED_SIGNERS = frozenset({"auto", "amt", "system", "orchestrator", "agent"})
_IDENT = re.compile(r"^[A-Za-z0-9_.+-]+$")


class SignoffError(ValueError):
    """A human gate could not be signed or no longer matches its inputs."""


class RepairRecord(BaseModel):
    """One accepted ``features/*.sql`` diff locked by the ready signature."""

    model_config = ConfigDict(extra="forbid")

    attempt: int = Field(ge=1)
    spec: str
    diff_sha256: str
    path: str


class ReadyReport(BaseModel):
    """Packet a human reads before signing the ready gate."""

    model_config = ConfigDict(extra="forbid")

    brief_name: str
    dataset: str
    status: Literal["pass", "fail", "incomplete"]
    primary_metric: str = ""
    evaluation: str = ""
    min_effect_mode: Literal["auto", "explicit"] = "explicit"
    proposed_min_effect: float
    resolved_min_effect: float | None = None
    budget_total_dbu: float
    noise_status: str
    ladder_status: str
    noise_sha256: str
    ladder_sha256: str
    manifest_sha256: str
    spec_path: str
    spec_sha256: str
    problem_task_sha256: str
    repairs: list[RepairRecord] = Field(default_factory=list)
    repair_hashes: list[str] = Field(default_factory=list)
    viable_families: list[str] = Field(default_factory=list)
    paradigms: list[dict[str, Any]] = Field(default_factory=list)
    unsupported_families: dict[str, str] = Field(default_factory=dict)
    modality: dict[str, Any] = Field(default_factory=dict)
    kendall_tau: float | None = None
    skip_r1: list[str] = Field(default_factory=list)
    remediation: list[str] = Field(default_factory=list)


class ReadySignoff(BaseModel):
    """One person's acceptance of a ready packet. Tracked in git."""

    model_config = ConfigDict(extra="forbid")

    gate: Literal["ready"] = "ready"
    signer: str
    signed_at: str
    report_sha256: str
    brief_name: str
    brief_path: str
    dataset: str
    problem_task_sha256: str
    spec_path: str
    spec_sha256: str
    manifest_path: str
    manifest_sha256: str
    repair_hashes: list[str] = Field(default_factory=list)
    repairs: list[RepairRecord] = Field(default_factory=list)
    resolved_min_effect: float
    total_dbu: float
    viable_families: list[str] = Field(default_factory=list)
    primary_metric: str
    evaluation: str
    noise_sha256: str
    ladder_sha256: str


def brief_key(brief: Brief, brief_path: Path) -> str:
    raw = brief.name or brief_path.stem
    if not _IDENT.fullmatch(raw):
        raise SignoffError(f"brief name {raw!r} cannot be used as a gate directory")
    return raw


def ready_dir(root: Path, name: str) -> Path:
    return root / "gates" / "ready" / name


def problem_task_payload(brief: Brief) -> dict[str, Any]:
    """Constraint hash. ``design_space.families`` and ``min_effect`` are omitted."""

    design = None
    if brief.design_space is not None:
        extra = {
            "out_of_scope_for_mvp": list(brief.design_space.out_of_scope_for_mvp),
            "data": brief.design_space.data,
        }
        if extra["out_of_scope_for_mvp"] or extra["data"]:
            design = extra
    return {
        "problem": brief.problem.model_dump(mode="json"),
        "task": brief.task.model_dump(mode="json"),
        "constraints": brief.constraints.model_dump(mode="json"),
        "budget": brief.budget.model_dump(mode="json"),
        "efficiency": brief.efficiency.model_dump(mode="json"),
        "eval": None if brief.eval is None else brief.eval.model_dump(mode="json"),
        "reference": None if brief.reference is None else brief.reference.model_dump(mode="json"),
        "off_limits": list(brief.off_limits),
        "dataset_spec": brief.resolved_dataset_spec(),
        "design_space": design,
    }


def problem_task_sha256(brief: Brief) -> str:
    encoded = json.dumps(
        problem_task_payload(brief), sort_keys=True, separators=(",", ":"), default=str
    )
    return hashlib.sha256(encoded.encode()).hexdigest()


def repair_records(root: Path, spec_name: str) -> list[RepairRecord]:
    """Diffs recorded by the dataset repair loop, oldest attempt first."""

    base = root / "runs" / "dataset-repair" / spec_name
    if not base.is_dir():
        return []
    records: list[RepairRecord] = []
    directories = [path for path in base.iterdir() if path.is_dir() and path.name.isdigit()]
    for directory in sorted(directories, key=lambda path: int(path.name)):
        patch = directory / "diff.patch"
        if not patch.is_file():
            continue
        records.append(
            RepairRecord(
                attempt=int(directory.name),
                spec=spec_name,
                diff_sha256=hash_file(patch),
                path=_relative(root, patch),
            )
        )
    return records


def write_noise_report(root: Path, brief: Brief, brief_path: Path, report: NoiseReport) -> Path:
    directory = ready_dir(root, brief_key(brief, brief_path))
    _write_envelope(directory / "noise_report.json", report.model_dump(mode="json"))
    (directory / "noise_report.md").write_text(_noise_markdown(report))
    return directory / "noise_report.json"


def write_ladder_report(root: Path, report: LadderReport, *, name: str | None = None) -> Path:
    directory = ready_dir(root, name or report.dataset)
    _write_envelope(directory / "ladder_report.json", report.model_dump(mode="json"))
    (directory / "ladder_report.md").write_text(_ladder_markdown(report))
    return directory / "ladder_report.json"


def load_noise_report(root: Path, brief: Brief, brief_path: Path) -> tuple[NoiseReport, str]:
    path = ready_dir(root, brief_key(brief, brief_path)) / "noise_report.json"
    body, digest = _read_envelope(path)
    return NoiseReport.model_validate(body), digest


def load_ladder_report(
    root: Path, dataset: str, *, name: str | None = None
) -> tuple[LadderReport, str]:
    path = ready_dir(root, name or dataset) / "ladder_report.json"
    body, digest = _read_envelope(path)
    return LadderReport.model_validate(body), digest


def publish_ready_packet(
    root: Path,
    brief: Brief,
    brief_path: Path,
    noise: NoiseReport,
    ladder: LadderReport,
    manifest_path: Path,
    *,
    repairs: list[RepairRecord] | None = None,
) -> tuple[ReadyReport, Path]:
    """Write the noise, ladder, manifest, and ready report into one directory."""

    name = brief_key(brief, brief_path)
    directory = ready_dir(root, name)
    directory.mkdir(parents=True, exist_ok=True)
    write_noise_report(root, brief, brief_path, noise)
    write_ladder_report(root, ladder, name=name)
    destination = directory / "manifest.json"
    source = manifest_path if manifest_path.is_absolute() else root / manifest_path
    source = _existing(source)
    if source.resolve() != destination.resolve():
        shutil.copyfile(source, destination)
    spec = load_spec(_spec_path(root, brief))
    accepted = repair_records(root, spec.name) if repairs is None else list(repairs)
    report = build_ready_report(
        root,
        brief,
        brief_path,
        noise=noise,
        ladder=ladder,
        manifest_path=destination,
        repairs=accepted,
    )
    path = write_ready_report(root, name, report)
    return report, path


def build_ready_report(
    root: Path,
    brief: Brief,
    brief_path: Path,
    *,
    noise: NoiseReport,
    ladder: LadderReport,
    manifest_path: Path,
    repairs: list[RepairRecord],
) -> ReadyReport:
    """Assemble the packet. Does not write it."""

    spec_path = _spec_path(root, brief)
    spec = load_spec(spec_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(manifest, dict):
        raise SignoffError("manifest is not a JSON object")
    profile = profile_from_manifest(manifest)
    paradigms = [] if profile is None else paradigm_decisions(profile, brief)
    blocked = {} if profile is None else ladder_blockers(profile, brief)
    unsupported = dict(ladder.unsupported_families)
    for family, reason in blocked.items():
        unsupported.setdefault(family, reason)
    viable = [family for family in ladder.family_set if family not in blocked]
    for family in ladder_families():
        if family in unsupported and family in viable:
            viable = [item for item in viable if item != family]
    noise_body = noise.model_dump(mode="json")
    ladder_body = ladder.model_dump(mode="json")
    fresh = resolved_min_effect(brief.effect_floor(), noise.required_min_effect)
    resolved = noise.resolved_min_effect if noise.resolved_min_effect is not None else fresh
    if fresh is not None and resolved is not None and abs(fresh - resolved) > 1e-9:
        resolved = fresh
    remediation: list[str] = []
    if noise.dataset != spec.name or ladder.dataset != spec.name:
        remediation.append(
            f"noise dataset {noise.dataset} or ladder dataset {ladder.dataset} "
            f"does not match spec {spec.name}"
        )
    if profile is None:
        remediation.append("manifest has no modality summary")
    if noise.status != "pass":
        remediation.extend(noise.remediation or [f"noise report is {noise.status}"])
    if ladder.status != "pass":
        remediation.extend(ladder.remediation or [f"ladder report is {ladder.status}"])
    if profile is not None and not viable:
        remediation.append("No ladder family is viable for this dataset and these constraints.")
    if noise.status == "incomplete" or ladder.status == "incomplete":
        status: Literal["pass", "fail", "incomplete"] = "incomplete"
    elif remediation:
        status = "fail"
    else:
        status = "pass"
    metric = brief.problem.metric
    return ReadyReport(
        brief_name=brief_key(brief, brief_path),
        dataset=spec.name,
        status=status,
        primary_metric=noise.primary_metric or (metric.name if metric is not None else ""),
        evaluation=noise.evaluation,
        min_effect_mode="auto" if brief.min_effect == "auto" else "explicit",
        proposed_min_effect=brief.effect_floor(),
        resolved_min_effect=resolved,
        budget_total_dbu=brief.budget.total_dbu,
        noise_status=noise.status,
        ladder_status=ladder.status,
        noise_sha256=_digest(noise_body),
        ladder_sha256=_digest(ladder_body),
        manifest_sha256=hash_file(manifest_path),
        spec_path=_relative(root, spec_path),
        spec_sha256=hash_tree(spec_path),
        problem_task_sha256=problem_task_sha256(brief),
        repairs=list(repairs),
        repair_hashes=[item.diff_sha256 for item in repairs],
        viable_families=viable,
        paradigms=[item.model_dump(mode="json") for item in paradigms],
        unsupported_families=unsupported,
        modality={} if profile is None else profile.model_dump(mode="json"),
        kendall_tau=ladder.kendall_tau,
        skip_r1=list(ladder.skip_r1),
        remediation=remediation,
    )


def write_ready_report(root: Path, name: str, report: ReadyReport) -> Path:
    directory = ready_dir(root, name)
    _write_envelope(directory / "ready_report.json", report.model_dump(mode="json"))
    (directory / "ready_report.md").write_text(_ready_markdown(report), encoding="utf-8")
    return directory / "ready_report.json"


def load_ready_report(root: Path, brief: Brief, brief_path: Path) -> tuple[ReadyReport, str]:
    path = ready_dir(root, brief_key(brief, brief_path)) / "ready_report.json"
    body, digest = _read_envelope(path)
    return ReadyReport.model_validate(body), digest


def sign_ready(
    root: Path,
    brief_path: Path,
    signer: str,
    *,
    signed_at: datetime | None = None,
) -> ReadySignoff:
    """Record the ready signature. Refuses a packet that is not a pass."""

    person = _signer(signer)
    brief = load_brief(brief_path)
    report, digest = load_ready_report(root, brief, brief_path)
    if report.status != "pass":
        detail = " ".join(report.remediation) or report.status
        raise SignoffError(f"ready gate is not ready to sign ({report.status}): {detail}")
    if report.resolved_min_effect is None:
        raise SignoffError("ready report has no resolved min_effect")
    fresh = _fresh_resolved(root, brief, brief_path)
    if fresh is None or abs(fresh - report.resolved_min_effect) > 1e-9:
        raise SignoffError("resolved min_effect does not match the noise report")
    metric = brief.problem.metric
    if metric is None or not metric.name or metric.name != report.primary_metric:
        raise SignoffError("the brief metric does not match the ready report")
    if abs(brief.budget.total_dbu - report.budget_total_dbu) > 1e-9:
        raise SignoffError("the brief budget does not match the ready report")
    manifest = ready_dir(root, report.brief_name) / "manifest.json"
    if not manifest.is_file() or hash_file(manifest) != report.manifest_sha256:
        raise SignoffError("manifest does not match the ready report")
    if not _repairs_match(root, report.repairs):
        raise SignoffError("accepted repair diffs do not match the ready report")
    if not _components_match(root, brief, brief_path, report):
        raise SignoffError("noise or ladder report does not match the ready packet")
    record = ReadySignoff(
        signer=person,
        signed_at=_stamp(signed_at),
        report_sha256=digest,
        brief_name=report.brief_name,
        brief_path=_relative(root, brief_path),
        dataset=report.dataset,
        problem_task_sha256=problem_task_sha256(brief),
        spec_path=report.spec_path,
        spec_sha256=report.spec_sha256,
        manifest_path=_relative(root, manifest),
        manifest_sha256=report.manifest_sha256,
        repair_hashes=list(report.repair_hashes),
        repairs=list(report.repairs),
        resolved_min_effect=report.resolved_min_effect,
        total_dbu=brief.budget.total_dbu,
        viable_families=list(report.viable_families),
        primary_metric=report.primary_metric,
        evaluation=report.evaluation,
        noise_sha256=report.noise_sha256,
        ladder_sha256=report.ladder_sha256,
    )
    path = ready_dir(root, record.brief_name) / "ready.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(record.model_dump_json(indent=2) + "\n", encoding="utf-8")
    return record


def load_ready(root: Path, brief: Brief, brief_path: Path) -> ReadySignoff | None:
    path = ready_dir(root, brief_key(brief, brief_path)) / "ready.json"
    if not path.is_file():
        return None
    return ReadySignoff.model_validate(json.loads(path.read_text(encoding="utf-8")))


def ready_is_current(root: Path, brief_path: Path) -> bool:
    """True when ``ready.json`` still matches the brief, spec, manifest, and repairs."""

    try:
        brief = load_brief(brief_path)
        name = brief_key(brief, brief_path)
    except (OSError, ValueError):
        return False
    record = load_ready(root, brief, brief_path)
    if record is None or record.gate != "ready" or record.brief_name != name:
        return False
    try:
        report, digest = load_ready_report(root, brief, brief_path)
    except (OSError, ValueError):
        return False
    if record.report_sha256 != digest or report.status != "pass":
        return False
    if record.problem_task_sha256 != problem_task_sha256(brief):
        return False
    if record.resolved_min_effect is None:
        return False
    fresh = _fresh_resolved(root, brief, brief_path)
    if fresh is None or abs(fresh - record.resolved_min_effect) > 1e-9:
        return False
    if abs(record.total_dbu - brief.budget.total_dbu) > 1e-9:
        return False
    metric = brief.problem.metric
    if metric is None or record.primary_metric != metric.name:
        return False
    if record.viable_families != report.viable_families:
        return False
    if record.repair_hashes != report.repair_hashes:
        return False
    if not _repairs_match(root, record.repairs):
        return False
    if not _components_match(root, brief, brief_path, report):
        return False
    return _ready_files_match(root, record)


def apply_locked_min_effect(root: Path, brief: Brief, brief_path: Path) -> float:
    """Use the signed floor when the ready gate is current. Otherwise the proposal."""

    value = brief.effect_floor()
    if ready_is_current(root, brief_path):
        record = load_ready(root, brief, brief_path)
        if record is not None:
            value = record.resolved_min_effect
    brief.min_effect = value
    return value


def search_block_reason(root: Path, brief: Brief, brief_path: Path) -> str | None:
    """Why a search brief must not start. Walking-skeleton briefs return None."""

    if brief.candidate is not None:
        return None
    if not ready_is_current(root, brief_path):
        return (
            "The ready gate is not signed for this brief. Run amt prep, then "
            "amt sign ready, and re-run with --fresh."
        )
    return None


def hash_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def hash_tree(directory: Path) -> str:
    if directory.is_file():
        return hash_file(directory)
    digest = hashlib.sha256()
    files = [
        path
        for path in directory.rglob("*")
        if path.is_file() and "__pycache__" not in path.parts and path.suffix != ".pyc"
    ]
    for path in sorted(files, key=lambda item: item.relative_to(directory).as_posix()):
        rel = path.relative_to(directory).as_posix()
        digest.update(rel.encode())
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def _fresh_resolved(root: Path, brief: Brief, brief_path: Path) -> float | None:
    try:
        noise, _digest = load_noise_report(root, brief, brief_path)
    except (OSError, ValueError):
        return None
    return resolved_min_effect(brief.effect_floor(), noise.required_min_effect)


def _components_match(
    root: Path, brief: Brief, brief_path: Path, report: ReadyReport
) -> bool:
    try:
        _noise, noise_digest = load_noise_report(root, brief, brief_path)
        _ladder, ladder_digest = load_ladder_report(
            root, report.dataset, name=report.brief_name
        )
    except (OSError, ValueError):
        return False
    return noise_digest == report.noise_sha256 and ladder_digest == report.ladder_sha256


def _repairs_match(root: Path, repairs: list[RepairRecord]) -> bool:
    for item in repairs:
        path = _resolve(root, item.path)
        if not path.is_file() or hash_file(path) != item.diff_sha256:
            return False
    return True


def _ready_files_match(root: Path, record: ReadySignoff) -> bool:
    try:
        spec = _resolve(root, record.spec_path)
        manifest = _resolve(root, record.manifest_path)
        if not spec.exists() or not manifest.is_file():
            return False
        return record.spec_sha256 == hash_tree(spec) and record.manifest_sha256 == hash_file(
            manifest
        )
    except OSError:
        return False


def _spec_path(root: Path, brief: Brief) -> Path:
    text = brief.resolved_dataset_spec()
    if not text:
        raise SignoffError("the brief has no dataset_spec")
    path = _resolve(root, text)
    if not path.exists():
        raise SignoffError(f"dataset spec {text} is missing")
    return path


def _resolve(root: Path, text: str) -> Path:
    candidate = Path(text)
    if candidate.is_absolute():
        return candidate
    rooted = root / candidate
    if rooted.exists():
        return rooted
    if candidate.exists():
        return candidate
    return rooted


def _existing(path: Path) -> Path:
    if not path.is_file():
        raise SignoffError(f"manifest {path} is missing")
    return path


def _relative(root: Path, path: Path) -> str:
    resolved = path.resolve()
    try:
        return resolved.relative_to(root.resolve()).as_posix()
    except ValueError:
        return resolved.as_posix()


def _signer(signer: str) -> str:
    person = signer.strip()
    if not person or person.casefold() in _RESERVED_SIGNERS:
        raise SignoffError("a human signer is required")
    return person


def _stamp(moment: datetime | None) -> str:
    when = moment or datetime.now(timezone.utc)
    if when.tzinfo is None:
        when = when.replace(tzinfo=timezone.utc)
    return when.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _write_envelope(path: Path, body: dict[str, Any]) -> str:
    digest = _digest(body)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"report_sha256": digest, "report": body}
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return digest


def _read_envelope(path: Path) -> tuple[dict[str, Any], str]:
    if not path.is_file():
        raise SignoffError(f"missing report {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    body = payload.get("report")
    if not isinstance(body, dict) or "report_sha256" not in payload:
        raise SignoffError(f"{path} is not a gate report")
    digest = _digest(body)
    if digest != payload["report_sha256"]:
        raise SignoffError(f"{path} was edited after it was written")
    return body, digest


def _digest(body: dict[str, Any]) -> str:
    encoded = json.dumps(body, sort_keys=True, separators=(",", ":"))
    normalized = json.loads(encoded)
    encoded = json.dumps(normalized, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode()).hexdigest()


def _noise_markdown(report: NoiseReport) -> str:
    from amt.noise import render_noise_markdown

    return render_noise_markdown(report)


def _ladder_markdown(report: LadderReport) -> str:
    from amt.ladder import render_ladder_markdown

    return render_ladder_markdown(report)


def _ready_markdown(report: ReadyReport) -> str:
    lines = [
        f"# Ready report: {report.brief_name}",
        "",
        f"Status: **{report.status}**",
        "",
        (
            f"Dataset `{report.dataset}`. "
            f"Primary metric `{report.evaluation or report.primary_metric}`."
        ),
        f"Budget {report.budget_total_dbu:g} DBU.",
        (
            f"min_effect {report.min_effect_mode}. "
            f"Proposed {report.proposed_min_effect:.6f}. "
            f"Resolved {report.resolved_min_effect}."
        ),
        f"Noise {report.noise_status}. Ladder {report.ladder_status}.",
        "",
        "Viable ladder families: " + (", ".join(report.viable_families) or "(none)"),
        "",
    ]
    if report.modality:
        lines.append(
            "Modality: "
            f"{report.modality.get('numeric_count', 0)} numeric, "
            f"{len(report.modality.get('categorical_columns') or [])} categorical, "
            f"embeddings {report.modality.get('embedding_groups') or []}, "
            f"{report.modality.get('row_count', 0)} rows, "
            f"{report.modality.get('query_count', 0)} queries."
        )
        lines.append("")
    if report.unsupported_families:
        lines.append("## Unsupported families")
        lines.append("")
        for family, reason in sorted(report.unsupported_families.items()):
            lines.append(f"- {family}: {reason}")
        lines.append("")
    if report.repairs:
        lines.append("## Accepted feature SQL repairs")
        lines.append("")
        for item in report.repairs:
            lines.append(f"- attempt {item.attempt}: `{item.path}` `{item.diff_sha256[:12]}`")
        lines.append("")
    if report.remediation:
        lines.append("## Remediation")
        lines.append("")
        for item in report.remediation:
            lines.append(f"- {item}")
        lines.append("")
    lines.append(
        "amt sign ready locks this packet: the problem and task, the spec, the manifest, "
        "accepted repair diffs, the resolved min_effect, the budget, and the viable families."
    )
    lines.append("")
    return "\n".join(lines)
