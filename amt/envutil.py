"""Load the repo `.env` without printing secrets.

Accepts both `key: value` and `key=value`. `dbx_token` and `dbx_host` are
copied onto the Databricks CLI variable names.
"""

from __future__ import annotations

import os
from pathlib import Path


def load_env(root: Path | None = None) -> None:
    path = (root or Path.cwd()) / ".env"
    if not path.is_file():
        return
    for raw in path.read_text().splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if "=" in line and ":" not in line.split("=", 1)[0]:
            key, value = line.split("=", 1)
        elif ":" in line:
            key, value = line.split(":", 1)
        else:
            continue
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if not key or key in os.environ and os.environ[key]:
            continue
        os.environ[key] = value
    token = os.environ.get("dbx_token") or os.environ.get("DBX_TOKEN")
    host = os.environ.get("dbx_host") or os.environ.get("DBX_HOST")
    if token and not os.environ.get("DATABRICKS_TOKEN"):
        os.environ["DATABRICKS_TOKEN"] = token
    if host and not os.environ.get("DATABRICKS_HOST"):
        os.environ["DATABRICKS_HOST"] = host.rstrip("/")
