"""
SuiteRunHistory — persist run summaries and detect pass-rate regression (Issue #8).

Tracks the overall pass rate across sequential suite runs and flags sustained
directional drift (e.g. a 95%→87%→79%→71% slide) that per-run results alone
cannot reveal.

Zero new dependencies: uses ``statistics.linear_regression``, ``pathlib``, and
``json`` only.

Usage::

    from agenttest.core.history import SuiteRunHistory
    from agenttest.core.runner import AgentTestRunner

    history = SuiteRunHistory(".agenttest-history.jsonl", window=10)

    results = runner.run_suite(suite)
    history.record(results)

    trend = history.analyze()
    if trend and trend.any_regression:
        print(f"Pass-rate regression detected: slope {trend.pass_rate_slope:+.4f}")
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

from agenttest.core.case import TestStatus


@dataclass
class RunSummary:
    """Aggregated pass/fail/error counts for a single suite run."""
    timestamp: float
    passed: int
    failed: int
    errors: int
    total: int
    skipped: int = 0

    @property
    def pass_rate(self) -> float:
        non_skip = self.total - self.skipped
        return self.passed / non_skip if non_skip else 0.0

    def to_dict(self) -> dict:
        return {
            "timestamp": self.timestamp,
            "passed": self.passed,
            "failed": self.failed,
            "errors": self.errors,
            "total": self.total,
            "skipped": self.skipped,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "RunSummary":
        return cls(
            timestamp=d["timestamp"],
            passed=d["passed"],
            failed=d["failed"],
            errors=d["errors"],
            total=d["total"],
            skipped=d.get("skipped", 0),
        )


@dataclass
class RunTrendReport:
    """Trend analysis over a window of recent runs."""
    window: int
    summaries: List[RunSummary]
    pass_rate_slope: float
    direction: str
    any_regression: bool


class SuiteRunHistory:
    """
    Append-only JSONL history of suite run summaries.

    Args:
        history_path: Path to the JSONL history file.
        window: Number of most recent runs to analyze (default 10).
    """

    def __init__(self, history_path: str = ".agenttest-history.jsonl", window: int = 10):
        self.path = Path(history_path)
        self.window = window

    def record(self, results: List) -> None:
        """Append a run summary to history."""
        summary = RunSummary(
            timestamp=time.time(),
            passed=sum(1 for r in results if r.status == TestStatus.PASSED),
            failed=sum(1 for r in results if r.status == TestStatus.FAILED),
            errors=sum(1 for r in results if r.status == TestStatus.ERROR),
            skipped=sum(1 for r in results if r.status == TestStatus.SKIPPED),
            total=len(results),
        )
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(summary.to_dict()) + "\n")

    def load(self, window: Optional[int] = None) -> List[RunSummary]:
        """Load the most recent ``window`` run summaries."""
        if not self.path.exists():
            return []
        limit = window or self.window
        lines = self.path.read_text(encoding="utf-8").strip().splitlines()
        recent = lines[-limit:] if limit else lines
        return [RunSummary.from_dict(json.loads(l)) for l in recent if l.strip()]

    def analyze(self) -> Optional[RunTrendReport]:
        """
        Compute an OLS slope over the last ``window`` run summaries.

        Returns:
            RunTrendReport, or None if fewer than 3 runs are recorded.
        """
        summaries = self.load()
        if len(summaries) < 3:
            return None

        try:
            from statistics import linear_regression
        except ImportError:  # pragma: no cover - Python < 3.10
            return None

        xs = list(range(len(summaries)))
        ys = [s.pass_rate for s in summaries]
        slope, _intercept = linear_regression(xs, ys)

        if slope > 0.005:
            direction = "improving"
        elif slope < -0.005:
            direction = "degrading"
        else:
            direction = "stable"

        return RunTrendReport(
            window=len(summaries),
            summaries=summaries,
            pass_rate_slope=round(slope, 6),
            direction=direction,
            any_regression=direction == "degrading",
        )

    def clear(self) -> None:
        """Delete the history file."""
        if self.path.exists():
            self.path.unlink()
