"""
Flakiness detection for non-deterministic agent tests (Issue #3).

Repeated runs of a test can oscillate between pass/fail due to LLM
non-determinism. This module tracks per-test pass rates and flags flaky tests
(high variance) so real regressions can be distinguished from random failures.

Usage::

    from agenttest.flakiness import FlakinessDetector

    detector = FlakinessDetector(threshold=0.2)   # >20% failure rate = flaky

    for _ in range(5):
        results = runner.run_suite(suite)
        detector.record(results)

    for report in detector.report():
        if report.is_flaky:
            print(f"Flaky: {report.test_name} ({report.pass_rate:.0%} pass rate)")
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

from agenttest.core.case import TestStatus


@dataclass
class FlakyReport:
    test_name: str
    total_runs: int
    passed: int
    failed: int
    pass_rate: float
    threshold: float
    is_flaky: bool

    def __repr__(self) -> str:  # pragma: no cover
        return (
            f"FlakyReport(name={self.test_name!r}, pass_rate={self.pass_rate:.0%}, "
            f"flaky={self.is_flaky})"
        )


class FlakinessDetector:
    """
    Tracks per-test pass rates across runs and flags flaky tests.

    Args:
        threshold: Failure rate above which a test is considered flaky
                   (default 0.2 → >20% failures).
        min_runs: Minimum number of runs before a verdict is reported
                  (default 3).
    """

    def __init__(self, threshold: float = 0.2, min_runs: int = 3) -> None:
        self.threshold = threshold
        self.min_runs = min_runs
        self._history: Dict[str, Dict[str, int]] = {}  # name → {passed, total}

    def record(self, results: List) -> None:
        """Record the outcomes of a single suite run."""
        for result in results:
            name = result.test_name
            entry = self._history.setdefault(name, {"passed": 0, "total": 0})
            entry["total"] += 1
            if result.status == TestStatus.PASSED:
                entry["passed"] += 1

    def report(self) -> List[FlakyReport]:
        """Compute flakiness reports for all tracked tests."""
        reports: List[FlakyReport] = []
        for name, entry in self._history.items():
            total = entry["total"]
            if total < self.min_runs:
                continue
            passed = entry["passed"]
            failed = total - passed
            pass_rate = passed / total if total else 0.0
            is_flaky = (1.0 - pass_rate) > self.threshold
            reports.append(FlakyReport(
                test_name=name,
                total_runs=total,
                passed=passed,
                failed=failed,
                pass_rate=pass_rate,
                threshold=self.threshold,
                is_flaky=is_flaky,
            ))
        return reports

    def flaky_tests(self) -> List[FlakyReport]:
        """Return only the flaky test reports."""
        return [r for r in self.report() if r.is_flaky]

    def stable_tests(self) -> List[FlakyReport]:
        """Return only the stable (non-flaky) test reports."""
        return [r for r in self.report() if not r.is_flaky]

    def __len__(self) -> int:
        return len(self._history)
