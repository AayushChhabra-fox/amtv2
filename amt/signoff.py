"""Human signatures for gates G1 and G2.

G1 is once per brief: the noise report, the brief, the dataset spec, the
manifest, the evaluation definition, ``min_effect``, and the budget. G2 is
once per dataset and ladder family set, and later runs reuse it. A signature
goes stale when the report, the brief, the spec, or the manifest changes.
Search briefs (no pinned candidate) do not start until both signatures are
current.
"""

from __future__ import annotations

import hashlib
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from amt.ladder import LadderReport
from amt.noise import NoiseReport
from amt.schemas import Brief, load_brief
from harness.data.contract import load_spec
from harness.ladder import family_key, ladder_families

_RESERVED_SIGNERS = frozenset({"auto", "amt", "system", "orchestrator", "agent"})
_IDENT = re.compile(r"^[A-Za-z0-9_.+-]+$")


class SignoffError(ValueError):
    """A human gate could not be signed or no longer matches its inputs."""


class Signoff(BaseModel):
    """One person's acceptance of a gate report."""

    model_config = ConfigDict(extra="forbid")

    gate: Literal["G1", "G2"]
    signer: str
    signed_at: str
    report_sha256: str
    dataset: str
    brief_name: str | None = None
    brief_path: str | None = None
    brief_sha256: str | None = None
    spec_path: str | None = None
    spec_sha256: str | None = None
    manifest_path: str | None = None
    manifest_sha256: str | None = None
    min_effect: float | None = None
    total_dbu: float | None = None
    primary_metric: str | None = None
    evaluation: str | None = None
    family_set: list[str] = Field(default_factory=list)
    family_key: str | None = None
    kendall_tau: float | None = None
    skip_r1: list[str] = Field(default_factory=list)
    r1_train_fraction: float | None = None


def brief_key(brief: Brief, brief_path: Path) -> str:
    raw = brief.name or brief_path.stem
    if not _IDENT.fullmatch(raw):
        raise SignoffError(f"brief name {raw!r} cannot be used as a gate directory")
    return raw


def g1_dir(root: Path, name: str) -> Path:
    return root / "gates" / "g1" / name


def g2_dir(root: Path, dataset: str, key: str | None = None) -> Path:
    return root / "gates" / "g2" / dataset / (key or family_key())


def write_noise_report(root: Path, brief: Brief, brief_path: Path, report: NoiseReport) -> Path:
    directory = g1_dir(root, brief_key(brief, brief_path))
    _write_envelope(directory / "noise_report.json", report.model_dump(mode="json"))
    (directory / "noise_report.md").write_text(_noise_markdown(report))
    return directory / "noise_report.json"


def write_ladder_report(root: Path, report: LadderReport) -> Path:
    directory = g2_dir(root, report.dataset, report.family_key)
    _write_envelope(directory / "ladder_report.json", report.model_dump(mode="json"))
    (directory / "ladder_report.md").write_text(_ladder_markdown(report))
    return directory / "ladder_report.json"


def load_noise_report(root: Path, brief: Brief, brief_path: Path) -> tuple[NoiseReport, str]:
    path = g1_dir(root, brief_key(brief, brief_path)) / "noise_report.json"
    body, digest = _read_envelope(path)
    return NoiseReport.model_validate(body), digest


def load_ladder_report(root: Path, dataset: str) -> tuple[LadderReport, str]:
    path = g2_dir(root, dataset) / "ladder_report.json"
    body, digest = _read_envelope(path)
    return LadderReport.model_validate(body), digest


def sign_g1(
    root: Path,
    brief_path: Path,
    signer: str,
    manifest_path: Path,
    *,
    signed_at: datetime | None = None,
) -> Signoff:
    """Record a G1 signature. Refuses a report that is not a pass."""

    person = _signer(signer)
    brief = load_brief(brief_path)
    report, digest = load_noise_report(root, brief, brief_path)
    if report.status != "pass":
        detail = " ".join(report.remediation) or report.status
        raise SignoffError(f"G1 is not ready to sign ({report.status}): {detail}")
    if not report.evaluation or not report.primary_metric:
        raise SignoffError("G1 needs an evaluation definition on the brief")
    metric = brief.problem.metric
    if metric is None or metric.name != report.primary_metric:
        raise SignoffError("the brief metric does not match the noise report")
    if abs(brief.min_effect - report.brief_min_effect) > 1e-12:
        raise SignoffError("rebuild the noise report after changing min_effect")
    if abs(brief.budget.total_dbu - report.budget_total_dbu) > 1e-9:
        raise SignoffError("rebuild the noise report after changing the budget")
    if report.required_min_effect is None or brief.min_effect + 1e-12 < report.required_min_effect:
        raise SignoffError("min_effect is below 2× the R3 half-width")
    spec_path = _spec_path(root, brief)
    manifest = _existing(manifest_path if manifest_path.is_absolute() else root / manifest_path)
    record = Signoff(
        gate="G1",
        signer=person,
        signed_at=_stamp(signed_at),
        report_sha256=digest,
        dataset=report.dataset,
        brief_name=brief_key(brief, brief_path),
        brief_path=_relative(root, brief_path),
        brief_sha256=hash_file(brief_path),
        spec_path=_relative(root, spec_path),
        spec_sha256=hash_tree(spec_path),
        manifest_path=_relative(root, manifest),
        manifest_sha256=hash_file(manifest),
        min_effect=brief.min_effect,
        total_dbu=brief.budget.total_dbu,
        primary_metric=report.primary_metric,
        evaluation=report.evaluation,
    )
    _write_signoff(g1_dir(root, record.brief_name or "") / "g1.json", record)
    return record


def sign_g2(
    root: Path,
    dataset: str,
    signer: str,
    *,
    signed_at: datetime | None = None,
) -> Signoff:
    """Record a G2 signature for this dataset and the current family set."""

    person = _signer(signer)
    report, digest = load_ladder_report(root, dataset)
    if report.status != "pass":
        detail = " ".join(report.remediation) or report.status
        raise SignoffError(f"G2 is not ready to sign ({report.status}): {detail}")
    if report.family_key != family_key() or report.family_set != ladder_families():
        raise SignoffError("the ladder report is for a different family set")
    record = Signoff(
        gate="G2",
        signer=person,
        signed_at=_stamp(signed_at),
        report_sha256=digest,
        dataset=dataset,
        family_set=list(report.family_set),
        family_key=report.family_key,
        kendall_tau=report.kendall_tau,
        skip_r1=list(report.skip_r1),
        r1_train_fraction=report.r1_train_fraction,
    )
    _write_signoff(g2_dir(root, dataset, report.family_key) / "g2.json", record)
    return record


def load_signoff(path: Path) -> Signoff | None:
    if not path.is_file():
        return None
    return Signoff.model_validate(json.loads(path.read_text()))


def g1_is_current(root: Path, brief_path: Path) -> bool:
    """True when the stored G1 signature still matches the brief and report."""

    try:
        brief = load_brief(brief_path)
        name = brief_key(brief, brief_path)
    except (OSError, ValueError):
        return False
    record = load_signoff(g1_dir(root, name) / "g1.json")
    if record is None or record.gate != "G1":
        return False
    try:
        report, digest = load_noise_report(root, brief, brief_path)
    except (OSError, ValueError):
        return False
    if record.report_sha256 != digest or report.status != "pass":
        return False
    if record.min_effect is None or abs(record.min_effect - brief.min_effect) > 1e-12:
        return False
    if record.total_dbu is None or abs(record.total_dbu - brief.budget.total_dbu) > 1e-9:
        return False
    metric = brief.problem.metric
    if metric is None or record.primary_metric != metric.name:
        return False
    return _hashes_match(
        root,
        record,
        brief_path=brief_path,
    )


def g2_is_current(root: Path, dataset: str) -> bool:
    """True when this dataset and family set have a current G2 signature."""

    record = load_signoff(g2_dir(root, dataset) / "g2.json")
    if record is None or record.gate != "G2":
        return False
    if record.family_key != family_key() or record.family_set != ladder_families():
        return False
    try:
        report, digest = load_ladder_report(root, dataset)
    except (OSError, ValueError):
        return False
    if record.report_sha256 != digest or report.status != "pass":
        return False
    if record.skip_r1 != report.skip_r1:
        return False
    return record.kendall_tau == report.kendall_tau


def search_block_reason(root: Path, brief: Brief, brief_path: Path) -> str | None:
    """Why a search brief must not start. Walking-skeleton briefs return None."""

    if brief.candidate is not None:
        return None
    if not g1_is_current(root, brief_path):
        return (
            "G1 is not signed for this brief. Sign the round-0 noise report, "
            "min_effect, and budget, then re-run with --fresh."
        )
    spec_text = brief.resolved_dataset_spec()
    if not spec_text:
        return "G2 requires a dataset_spec on the brief."
    try:
        spec = load_spec(_spec_path(root, brief))
    except (OSError, ValueError) as exc:
        return f"G2 cannot load dataset spec {spec_text}: {exc}"
    if not g2_is_current(root, spec.name):
        families = ", ".join(ladder_families())
        return (
            f"G2 is not signed for dataset {spec.name} and ladder families {families}. "
            "Sign the ladder report, then re-run with --fresh."
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


def _hashes_match(root: Path, record: Signoff, *, brief_path: Path) -> bool:
    try:
        if record.brief_sha256 != hash_file(brief_path):
            return False
        if record.spec_path is None or record.spec_sha256 is None:
            return False
        if record.manifest_path is None or record.manifest_sha256 is None:
            return False
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
        raise SignoffError("G1 and G2 require a human signer")
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
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return digest


def _read_envelope(path: Path) -> tuple[dict[str, Any], str]:
    if not path.is_file():
        raise SignoffError(f"missing report {path}")
    payload = json.loads(path.read_text())
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


def _write_signoff(path: Path, record: Signoff) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(record.model_dump_json(indent=2) + "\n")


def _noise_markdown(report: NoiseReport) -> str:
    from amt.noise import render_noise_markdown

    return render_noise_markdown(report)


def _ladder_markdown(report: LadderReport) -> str:
    from amt.ladder import render_ladder_markdown

    return render_ladder_markdown(report)
