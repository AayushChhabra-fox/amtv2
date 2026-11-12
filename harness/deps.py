"""Pinned third-party packages for one candidate.

The candidate declares exact pins in ``requirements.txt``. The harness
installs them. Candidate code never runs pip, and it still cannot touch
the network, Spark, or subprocess.
"""

from __future__ import annotations

import hashlib
import os
import re
import subprocess
import sys
import time
from dataclasses import dataclass
from importlib.metadata import PackageNotFoundError, distributions, packages_distributions, version
from pathlib import Path

_PIN = re.compile(
    r"^([A-Za-z0-9](?:[A-Za-z0-9._-]*[A-Za-z0-9])?)==([A-Za-z0-9][A-Za-z0-9.*+!-]*)$"
)

# Distribution names already provided by the training environment. A candidate
# must not pin a different build of these.
BASE_PACKAGES = frozenset(
    {
        "numpy",
        "pandas",
        "polars",
        "lightgbm",
        "xgboost",
        "torch",
        "scikit-learn",
        "sentence-transformers",
        "mlflow",
    }
)

# Same clients the preflight check refuses to import, plus Spark.
DENIED_PACKAGES = frozenset(
    {
        "pyspark",
        "spark",
        "dbutils",
        "requests",
        "urllib3",
        "httpx",
        "aiohttp",
        "boto3",
        "socket",
        "subprocess",
    }
)

_MAX_PINS = 10


class RequirementsError(ValueError):
    """The requirements file is not a list of exact, allowed pins."""


@dataclass(frozen=True)
class Requirement:
    name: str
    version: str


def normalize_name(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name).lower()


def parse_requirements(path: Path) -> list[Requirement]:
    """Accept only ``name==version`` lines. Reject URLs, VCS, and flags."""

    if not path.is_file():
        raise RequirementsError(f"{path} is missing")
    found: list[Requirement] = []
    seen: set[str] = set()
    for line_number, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        text = raw.split("#", 1)[0].strip()
        if not text:
            continue
        lowered = text.lower()
        if (
            text.startswith("-")
            or "://" in text
            or lowered.startswith("git+")
            or "@" in text
            or "/" in text
            or "\\" in text
        ):
            raise RequirementsError(f"line {line_number} is not an exact pin: {text}")
        match = _PIN.fullmatch(text)
        if match is None:
            raise RequirementsError(f"line {line_number} is not an exact pin: {text}")
        name, pinned = match.group(1), match.group(2)
        key = normalize_name(name)
        if key in seen:
            raise RequirementsError(f"line {line_number} repeats {name}")
        if key in {normalize_name(item) for item in BASE_PACKAGES}:
            raise RequirementsError(f"{name} is already in the base environment")
        denied = {normalize_name(item) for item in DENIED_PACKAGES}
        if key in denied or key.startswith("databricks-"):
            raise RequirementsError(f"{name} is blocked")
        seen.add(key)
        found.append(Requirement(name=name, version=pinned))
    if len(found) > _MAX_PINS:
        raise RequirementsError(f"at most {_MAX_PINS} packages")
    return found


def requirements_hash(requirements: list[Requirement]) -> str:
    blob = "\n".join(
        f"{normalize_name(item.name)}=={item.version}" for item in requirements
    )
    return hashlib.sha256(blob.encode()).hexdigest()[:16]


def constraints_text() -> str:
    """Pin the base environment so a candidate install cannot replace it."""

    lines: list[str] = []
    for name in sorted(BASE_PACKAGES):
        try:
            lines.append(f"{name}=={version(name)}")
        except PackageNotFoundError:
            continue
    return "\n".join(lines) + ("\n" if lines else "")


def env_root() -> Path:
    """Directory that holds one subdirectory per requirements hash."""

    override = os.environ.get("AMT_ENV_ROOT")
    if override:
        return Path(override)
    preferred = Path("/local_disk0/amt_envs")
    if preferred.parent.is_dir():
        return preferred
    return Path.cwd() / ".amt" / "envs"


def activate_candidate_requirements(model_path: Path) -> tuple[str, float]:
    """Install pins and put them on ``sys.path``. Returns ``(sha, seconds)``."""

    req_path = model_path.parent / "requirements.txt"
    if not req_path.is_file():
        return "", 0.0
    started = time.perf_counter()
    dest = install(model_path.parent, env_root())
    location = str(dest)
    if location not in sys.path:
        sys.path.insert(0, location)
    digest = requirements_hash(parse_requirements(req_path))
    return digest, time.perf_counter() - started


def install(candidate_dir: Path, target: Path) -> Path:
    """Install pins into ``target/<hash>``. A matching directory is reused."""

    req_path = candidate_dir / "requirements.txt"
    requirements = parse_requirements(req_path)
    if not requirements:
        raise RequirementsError("requirements.txt has no pins")
    digest = requirements_hash(requirements)
    dest = target / digest
    marker = dest / ".amt_installed"
    if marker.is_file() and marker.read_text(encoding="utf-8").strip() == digest:
        return dest
    dest.mkdir(parents=True, exist_ok=True)
    constraints = target / f"{digest}.constraints.txt"
    constraints.write_text(constraints_text(), encoding="utf-8")
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            "--disable-pip-version-check",
            "--target",
            str(dest),
            "-c",
            str(constraints),
            "-r",
            str(req_path),
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        detail = (completed.stderr or completed.stdout or "pip install failed").strip()
        raise RequirementsError(detail[-500:])
    marker.write_text(digest + "\n", encoding="utf-8")
    return dest


def import_names(target: Path) -> frozenset[str]:
    """Import roots provided by distributions installed under ``target``."""

    installed: set[str] = set()
    names: set[str] = set()
    for dist in distributions(path=[str(target)]):
        raw = dist.metadata["Name"] or ""
        installed.add(normalize_name(raw))
        top = dist.read_text("top_level.txt") or ""
        for line in top.splitlines():
            root = line.strip().split(".", 1)[0]
            if root:
                names.add(root)
    for imported, dists in packages_distributions().items():
        if any(normalize_name(item) in installed for item in dists):
            names.add(imported.split(".", 1)[0])
    return frozenset(names)
