"""
Agent behavior snapshots for regression testing (Issue #1).

AI agent outputs are non-deterministic, so hardcoded assertions fail
unpredictably. Snapshots capture a stable, derived representation of an agent's
behavior and compare against it on subsequent runs, surfacing behavioral drift.

Snapshots are stored as JSON in ``__snapshots__/`` (git-trackable).

Usage::

    from agenttest.snapshots import SnapshotStore, snapshot

    store = SnapshotStore()

    @snapshot(store, name="support_tone")
    def test_customer_support_tone(agent):
        run = agent("I'm very frustrated with your service!")
        return {"sentiment": analyze(run), "escalated": run.escalated}

    # First run: writes __snapshots__/support_tone.json
    # Subsequent runs: compares, raising AssertionError on drift
"""

from __future__ import annotations

import difflib
import json
import os
from pathlib import Path
from typing import Any, Callable, Dict, Optional

DEFAULT_SNAPSHOT_DIR = "__snapshots__"

_UPDATE_ENV = "AGENTTEST_UPDATE_SNAPSHOTS"


class SnapshotStore:
    """
    Persists and compares snapshot fixtures.

    Args:
        directory: Directory to store snapshots (default ``__snapshots__``).
        update: If True, always overwrite stored snapshots (or read
                ``AGENTTEST_UPDATE_SNAPSHOTS`` env var when None).
        similarity_threshold: Optional [0, 1] fuzzy-match threshold for strings.
    """

    def __init__(
        self,
        directory: str = DEFAULT_SNAPSHOT_DIR,
        update: Optional[bool] = None,
        similarity_threshold: Optional[float] = None,
    ) -> None:
        self.directory = Path(directory)
        if update is None:
            update = os.environ.get(_UPDATE_ENV, "0") in ("1", "true", "yes")
        self.update = update
        self.similarity_threshold = similarity_threshold

    def _path(self, name: str) -> Path:
        return self.directory / f"{name}.json"

    def store(self, name: str, value: Any) -> None:
        """Write a snapshot value to disk."""
        self.directory.mkdir(parents=True, exist_ok=True)
        self._path(name).write_text(
            json.dumps(value, indent=2, ensure_ascii=False, sort_keys=True),
            encoding="utf-8",
        )

    def load(self, name: str) -> Optional[Any]:
        """Load a stored snapshot, or None if absent."""
        path = self._path(name)
        if not path.exists():
            return None
        return json.loads(path.read_text(encoding="utf-8"))

    def assert_matches(self, name: str, value: Any) -> None:
        """
        Compare ``value`` against the stored snapshot.

        On first run (no snapshot) or when update is enabled, stores and
        returns without asserting. Otherwise raises AssertionError on drift.
        """
        stored = self.load(name)

        if stored is None or self.update:
            self.store(name, value)
            return

        if self._equal(stored, value):
            return

        raise AssertionError(self._diff_message(name, stored, value))

    # ── Comparison helpers ──────────────────────────────────────────────────

    def _equal(self, stored: Any, value: Any) -> bool:
        if self.similarity_threshold is not None and isinstance(stored, str) and isinstance(value, str):
            return _string_similarity(stored, value) >= self.similarity_threshold
        return stored == value

    def _diff_message(self, name: str, stored: Any, value: Any) -> str:
        old_lines = json.dumps(stored, indent=2, ensure_ascii=False, sort_keys=True).splitlines()
        new_lines = json.dumps(value, indent=2, ensure_ascii=False, sort_keys=True).splitlines()
        diff = "\n".join(difflib.unified_diff(
            old_lines, new_lines,
            fromfile=f"{name}.json (stored)",
            tofile=f"{name}.json (actual)",
            lineterm="",
        ))
        return f"Snapshot mismatch for '{name}':\n{diff}"


def _string_similarity(a: str, b: str) -> float:
    """SequenceMatcher-based similarity ratio in [0, 1]."""
    from difflib import SequenceMatcher
    return SequenceMatcher(None, a, b).ratio()


def snapshot(store: SnapshotStore, name: str, key: Optional[str] = None):
    """
    Decorator that snapshots a test function's return value.

    Args:
        store: A SnapshotStore instance.
        name: Snapshot name (defaults to the function name when None).
        key: Optional key within a returned dict to snapshot (defaults to the
             entire return value).
    """
    def decorator(fn: Callable) -> Callable:
        snapshot_name = name or fn.__name__

        def wrapper(*args, **kwargs):
            result = fn(*args, **kwargs)
            value = result.get(key) if key is not None and isinstance(result, dict) else result
            store.assert_matches(snapshot_name, value)
            return result

        wrapper.__name__ = getattr(fn, "__name__", "snapshot_test")
        wrapper._snapshot_name = snapshot_name
        return wrapper

    return decorator
