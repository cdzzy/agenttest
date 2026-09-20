"""Shared pytest configuration.

Pytester's ``runpytest_subprocess`` launches a fresh interpreter whose
``sys.path`` does not include the project root when the package is not
installed (e.g. running the suite from a source checkout). Ensure the
project root is on ``PYTHONPATH`` so subprocesses can import
``agenttest`` the same way the parent process does. Harmless when the
package is already installed.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)


@pytest.fixture(autouse=True)
def _project_root_on_pythonpath(monkeypatch: pytest.MonkeyPatch) -> None:
    existing = os.environ.get("PYTHONPATH", "")
    parts = [PROJECT_ROOT] + [p for p in existing.split(os.pathsep) if p]
    monkeypatch.setenv("PYTHONPATH", os.pathsep.join(parts))
