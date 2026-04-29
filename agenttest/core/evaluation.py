"""
Evaluation harness — run multiple test suites with evaluation metrics.

Inspired by hermes-agent's self-evaluation loops and goose's extensible
testing patterns. Provides a framework for running AI agent tests,
collecting metrics, and generating evaluation reports.

Features:
- Suite-based test organization
- Pass/fail/skip with timing and token tracking
- Aggregate metrics with per-suit and per-category breakdowns
- JSON report generation for CI integration
- Trend comparison between runs
"""

from __future__ import annotations

import json
import time
import uuid
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Type


class TestStatus(str, Enum):
    PASSED = "passed"
    FAILED = "failed"
    ERROR = "error"
    SKIPPED = "skipped"


@dataclass
class TestCaseResult:
    """Result of a single test case."""
    name: str
    status: TestStatus
    duration_ms: float = 0.0
    tokens_used: int = 0
    error_message: str = ""
    assertion_detail: str = ""

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "status": self.status.value,
            "duration_ms": round(self.duration_ms, 1),
            "tokens_used": self.tokens_used,
            "error_message": self.error_message,
            "assertion_detail": self.assertion_detail,
        }


@dataclass
class SuiteResult:
    """Result of a test suite."""
    suite_name: str
    cases: List[TestCaseResult] = field(default_factory=list)
    started_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    ended_at: Optional[str] = None
    tags: List[str] = field(default_factory=list)

    @property
    def total(self) -> int:
        return len(self.cases)

    @property
    def passed(self) -> int:
        return sum(1 for c in self.cases if c.status == TestStatus.PASSED)

    @property
    def failed(self) -> int:
        return sum(1 for c in self.cases if c.status == TestStatus.FAILED)

    @property
    def errors(self) -> int:
        return sum(1 for c in self.cases if c.status == TestStatus.ERROR)

    @property
    def skipped(self) -> int:
        return sum(1 for c in self.cases if c.status == TestStatus.SKIPPED)

    @property
    def total_tokens(self) -> int:
        return sum(c.tokens_used for c in self.cases)

    @property
    def total_duration_ms(self) -> float:
        return sum(c.duration_ms for c in self.cases)

    @property
    def pass_rate(self) -> float:
        attempted = self.total - self.skipped
        return self.passed / attempted if attempted > 0 else 0.0

    def to_dict(self) -> dict:
        return {
            "suite_name": self.suite_name,
            "tags": self.tags,
            "started_at": self.started_at,
            "ended_at": self.ended_at,
            "summary": {
                "total": self.total,
                "passed": self.passed,
                "failed": self.failed,
                "errors": self.errors,
                "skipped": self.skipped,
                "pass_rate": round(self.pass_rate, 3),
                "total_tokens": self.total_tokens,
                "total_duration_ms": round(self.total_duration_ms, 1),
            },
            "cases": [c.to_dict() for c in self.cases],
        }


@dataclass
class EvalReport:
    """Complete evaluation report across all suites."""
    run_id: str = field(default_factory=lambda: str(uuid.uuid4())[:12])
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    suites: List[SuiteResult] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def total(self) -> int:
        return sum(s.total for s in self.suites)

    @property
    def passed(self) -> int:
        return sum(s.passed for s in self.suites)

    @property
    def failed(self) -> int:
        return sum(s.failed + s.errors for s in self.suites)

    @property
    def pass_rate(self) -> float:
        attempted = self.total - sum(s.skipped for s in self.suites)
        return self.passed / attempted if attempted > 0 else 0.0

    def to_dict(self) -> dict:
        return {
            "run_id": self.run_id,
            "timestamp": self.timestamp,
            "metadata": self.metadata,
            "summary": {
                "total": self.total,
                "passed": self.passed,
                "failed": self.failed,
                "pass_rate": round(self.pass_rate, 3),
                "suites": len(self.suites),
            },
            "suites": [s.to_dict() for s in self.suites],
        }

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, ensure_ascii=False)


class EvalHarness:
    """
    Test evaluation harness for AI agent testing.

    Organizes tests into suites, runs them, and produces detailed
    evaluation reports with metrics and trend data.

    Usage::

        harness = EvalHarness(metadata={"model": "gpt-4", "version": "2.0"})

        # Define a test case
        def test_basic_response():
            run = my_agent.run("Hello")
            assert_run_output_contains(run, "greeting")

        # Add to suite
        harness.add_suite("basic_suite", [
            ("test_basic_response", test_basic_response),
        ])

        # Run all suites
        report = harness.run()
        print(report.to_json())

        # Or run specific suite
        result = harness.run_suite("basic_suite")
    """

    def __init__(self, metadata: Optional[Dict[str, Any]] = None):
        self._suites: Dict[str, List[tuple]] = {}
        self._suite_tags: Dict[str, List[str]] = {}
        self.metadata = metadata or {}

    def add_suite(
        self,
        name: str,
        tests: List[tuple],
        tags: Optional[List[str]] = None,
    ) -> None:
        """
        Register a test suite.

        Args:
            name: Suite name (must be unique)
            tests: List of (test_name, test_callable) tuples
            tags: Optional tags for categorization
        """
        if name in self._suites:
            raise ValueError(f"Suite '{name}' already exists")
        self._suites[name] = tests
        self._suite_tags[name] = tags or []

    def remove_suite(self, name: str) -> None:
        """Remove a suite by name."""
        self._suites.pop(name, None)
        self._suite_tags.pop(name, None)

    def run_suite(self, name: str) -> SuiteResult:
        """
        Run a single test suite.

        Args:
            name: Suite name

        Returns:
            SuiteResult with individual test results and aggregate stats.
        """
        if name not in self._suites:
            raise ValueError(f"Suite '{name}' not found")

        result = SuiteResult(
            suite_name=name,
            tags=self._suite_tags.get(name, []),
        )

        for test_name, test_fn in self._suites[name]:
            t0 = time.time()
            try:
                test_fn()
                result.cases.append(TestCaseResult(
                    name=test_name,
                    status=TestStatus.PASSED,
                    duration_ms=(time.time() - t0) * 1000,
                ))
            except AssertionError as e:
                result.cases.append(TestCaseResult(
                    name=test_name,
                    status=TestStatus.FAILED,
                    duration_ms=(time.time() - t0) * 1000,
                    error_message=str(e),
                    assertion_detail=str(e),
                ))
            except Exception as e:
                result.cases.append(TestCaseResult(
                    name=test_name,
                    status=TestStatus.ERROR,
                    duration_ms=(time.time() - t0) * 1000,
                    error_message=f"{type(e).__name__}: {e}",
                ))

        result.ended_at = datetime.now(timezone.utc).isoformat()
        return result

    def run(self, suite_names: Optional[List[str]] = None) -> EvalReport:
        """
        Run all suites (or specified ones) and produce a report.

        Args:
            suite_names: Optional list of suite names to run. None = all.

        Returns:
            EvalReport with results from all suites.
        """
        report = EvalReport(metadata=self.metadata)

        names = suite_names or list(self._suites.keys())
        for name in names:
            try:
                result = self.run_suite(name)
                report.suites.append(result)
            except Exception as e:
                # Create a failed suite result
                report.suites.append(SuiteResult(
                    suite_name=name,
                    ended_at=datetime.now(timezone.utc).isoformat(),
                ))
                report.suites[-1].cases.append(TestCaseResult(
                    name=f"suite_error",
                    status=TestStatus.ERROR,
                    error_message=f"Suite setup failed: {e}",
                ))

        return report

    def list_suites(self) -> List[str]:
        """Return all registered suite names."""
        return list(self._suites.keys())

    def get_suite_info(self, name: str) -> Dict[str, Any]:
        """Get info about a registered suite."""
        if name not in self._suites:
            return {}
        tests = self._suites[name]
        return {
            "name": name,
            "tags": self._suite_tags.get(name, []),
            "test_count": len(tests),
            "test_names": [t[0] for t in tests],
        }

