"""Single list of packages the training environment provides.

Candidates may import ``BASE_IMPORTS`` without pinning them. They may not
re-pin ``BASE_PACKAGES`` or import ``DENIED_IMPORTS``. Job clusters install
``PYPI_LIBRARIES``. HuggingFace hub downloads are refused unless
``AMT_HF_OFFLINE_CACHE`` points at a Unity Catalog volume.
"""

from __future__ import annotations

import os

# Distribution names already installed for training. A candidate must not pin
# a different build of these.
BASE_PACKAGES = frozenset(
    {
        "numpy",
        "pandas",
        "pyarrow",
        "scipy",
        "lightgbm",
        "xgboost",
        "catboost",
        "torch",
        "scikit-learn",
    }
)

# Import roots for those distributions. ``scikit-learn`` imports as ``sklearn``.
BASE_IMPORTS = frozenset(
    {
        "numpy",
        "pandas",
        "pyarrow",
        "scipy",
        "lightgbm",
        "xgboost",
        "catboost",
        "torch",
        "sklearn",
    }
)

# Import roots the static check rejects.
DENIED_IMPORTS = frozenset(
    {
        "pyspark",
        "spark",
        "dbutils",
        "subprocess",
        "socket",
        "requests",
        "urllib",
        "http",
        "ftplib",
        "smtplib",
        "webbrowser",
        "httpx",
        "aiohttp",
        "boto3",
    }
)

# Distributions a candidate requirements file may not pin. Includes clients
# that are not importable under the same name as the distribution.
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

# Calls that fetch weights from the HuggingFace hub.
HF_DOWNLOAD_CALLS = frozenset({"from_pretrained", "hf_hub_download", "snapshot_download"})

# PyPI libraries attached to train, confirm, final_holdout, and build_dataset.
# ``mlflow`` is the harness logger, not a candidate base package. Torch comes
# from the CPU index so a CPU cluster does not pull CUDA wheels.
PYPI_LIBRARIES: tuple[dict[str, dict[str, str]], ...] = (
    {"pypi": {"package": "numpy"}},
    {"pypi": {"package": "pandas"}},
    {"pypi": {"package": "pyarrow"}},
    {"pypi": {"package": "scipy"}},
    {"pypi": {"package": "scikit-learn"}},
    {"pypi": {"package": "lightgbm"}},
    {"pypi": {"package": "xgboost"}},
    {"pypi": {"package": "catboost"}},
    {"pypi": {"package": "torch", "repo": "https://download.pytorch.org/whl/cpu"}},
    {"pypi": {"package": "mlflow"}},
)


def pypi_libraries() -> list[dict[str, dict[str, str]]]:
    """Copies of ``PYPI_LIBRARIES`` safe for a job payload to mutate."""

    return [{"pypi": dict(item["pypi"])} for item in PYPI_LIBRARIES]


def offline_weights_allowed() -> bool:
    """True when hub weights are pinned to a volume path, not downloaded live."""

    cache = os.environ.get("AMT_HF_OFFLINE_CACHE", "").strip()
    return cache.startswith("/Volumes/")
