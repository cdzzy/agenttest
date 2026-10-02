"""
Flaky test detection for CI (wires the `fail-on-flaky` composite-action input).

Reads the per-test outcome records appended by the pytest plugin
(``agenttest.pytest_plugin``) to ``.agenttest-history.jsonl``, groups them by
test name, and flags tests whose outcome varied across runs. Exits 1 (with a
report) when ``--fail-on-flaky`` is set and flaky tests exist.

Usage::

    python -m agenttest.check_flaky [--history .agenttest-history.jsonl]
                                    [--min-runs 2] [--fail-on-flaky]
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional


@dataclass
class FlakyTestReport:
    """Per-test outcome summary across history records."""
    name: str
    runs: int
    passed: int
    failed: int
    is_flaky: bool

    def __repr__(self) -> str:  # pragma: no cover
        return (
            f"FlakyTestReport(name={self.name!r}, passed={self.passed}/"
            f"{self.runs}, flaky={self.is_flaky})"
        )


def load_history(path: Path) -> List[dict]:
    """Load `kind: "test"` records, skipping corrupt lines."""
    if not path.exists():
        return []
    records: List[dict] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            continue
        if record.get("kind") == "test":
            records.append(record)
    return records


def analyze(records: List[dict], min_runs: int = 2) -> List[FlakyTestReport]:
    """Flag tests whose outcomes varied across runs (0 < passed < total)."""
    by_name: Dict[str, List[dict]] = {}
    for record in records:
        by_name.setdefault(record.get("name", ""), []).append(record)

    reports: List[FlakyTestReport] = []
    for name, runs in by_name.items():
        if not name or len(runs) < min_runs:
            continue
        passed = sum(int(r.get("passed", 0)) for r in runs)
        total = sum(int(r.get("total", 0)) for r in runs)
        reports.append(FlakyTestReport(
            name=name,
            runs=total,
            passed=passed,
            failed=total - passed,
            is_flaky=0 < passed < total,
        ))
    return reports


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        prog="agenttest.check_flaky",
        description="Detect flaky agent tests from .agenttest-history.jsonl",
    )
    parser.add_argument(
        "--history",
        default=".agenttest-history.jsonl",
        help="Path to the history JSONL file (default: .agenttest-history.jsonl)",
    )
    parser.add_argument(
        "--min-runs",
        type=int,
        default=2,
        help="Minimum records per test before a verdict is reported (default: 2)",
    )
    parser.add_argument(
        "--fail-on-flaky",
        action="store_true",
        help="Exit non-zero when flaky tests are found",
    )
    args = parser.parse_args(argv)

    reports = analyze(load_history(Path(args.history)), args.min_runs)
    flaky = [r for r in reports if r.is_flaky]

    if flaky:
        print(f"{len(flaky)} flaky test(s) detected:")
        for r in flaky:
            print(f"  - {r.name}: {r.passed}/{r.runs} passed")
    elif reports:
        print(f"No flaky tests (checked {len(reports)} test(s)).")

    if flaky and args.fail_on_flaky:
        return 1
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
