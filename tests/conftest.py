"""Keep tests independent of the developer's shell and repo `.env`."""

from __future__ import annotations

import os

import pytest

_PREFIXES = ("AMT_", "DATABRICKS_", "DBX_", "dbx_")
_NAMES = ("CURSOR_API_KEY",)


@pytest.fixture(autouse=True)
def _hermetic_env(request: pytest.FixtureRequest, monkeypatch: pytest.MonkeyPatch) -> None:
    if request.node.get_closest_marker("databricks"):
        return
    for name in list(os.environ):
        if name.startswith(_PREFIXES) or name in _NAMES:
            monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr("amt.cli.load_env", lambda *_args, **_kwargs: None)
