"""
JUnit XML -> Markdown summary for CI pull-request comments.

Turns a pytest ``--junitxml`` report into a compact Markdown block that the
bundled GitHub Action posts (and updates) on the pull request, so reviewers
see agent quality results without opening the Actions tab.

The block starts with :data:`REPORT_MARKER`; the Action uses that marker to
find and update its previous comment instead of stacking a new one on every
push.

Usage::

    # From a shell / CI step
    python -m agenttest.ci_summary agenttest-results.xml

    # As a library
    from agenttest.ci_summary import markdown_from_junit, parse_junit
    report = parse_junit("agenttest-results.xml")
    print(markdown_from_junit("agenttest-results.xml"))
"""

from __future__ import annotations

import sys
import xml.etree.ElementTree as ET
from typing import Any, Dict, List, Optional

#: Hidden marker that lets the Action find and update its own PR comment.
REPORT_MARKER = "<!-- agenttest-ci-report -->"


def _case_status(case: ET.Element) -> str:
    """Classify a ``<testcase>`` element."""
    if case.find("failure") is not None:
        return "failed"
    if case.find("error") is not None:
        return "error"
    if case.find("skipped") is not None:
        return "skipped"
    return "passed"


def _failure_text(case: ET.Element) -> Optional[str]:
    """Extract the failure/error message and stack trace, if any."""
    node = case.find("failure")
    if node is None:
        node = case.find("error")
    if node is None:
        return None
    message = (node.get("message") or "").strip()
    detail = (node.text or "").strip()
    if message and detail:
        return f"{message}\n{detail}"
    return message or detail or None


def parse_junit(path: str) -> Dict[str, Any]:
    """
    Parse a JUnit XML file into a summary dict.

    Supports both roots pytest can emit: ``<testsuites>`` (multiple suites) and
    ``<testsuite>`` (a single suite).

    Returns:
        ``{"tests", "failures", "errors", "skipped", "time", "cases"}`` where
        ``cases`` is a list of ``{"name", "classname", "time", "status",
        "failure"}``.

    Raises:
        FileNotFoundError: If ``path`` does not exist.
        ValueError: If the file is not valid JUnit XML.
    """
    try:
        tree = ET.parse(path)
    except FileNotFoundError:
        raise
    except ET.ParseError as exc:
        raise ValueError(f"Not valid JUnit XML: {path} ({exc})") from exc

    root = tree.getroot()
    if root.tag == "testsuites":
        cases: List[ET.Element] = root.findall(".//testcase")
    elif root.tag == "testsuite":
        cases = root.findall("testcase")
    else:
        raise ValueError(
            f"Unsupported JUnit root element <{root.tag}>; expected "
            "<testsuites> or <testsuite>"
        )

    parsed_cases: List[Dict[str, Any]] = []
    counts = {"passed": 0, "failed": 0, "error": 0, "skipped": 0}

    for case in cases:
        status = _case_status(case)
        counts[status] += 1
        parsed_cases.append(
            {
                "name": case.get("name", "<unnamed>"),
                "classname": case.get("classname", ""),
                "time": float(case.get("time", "0") or 0),
                "status": status,
                "failure": _failure_text(case),
            }
        )

    return {
        "tests": len(cases),
        "failures": counts["failed"],
        "errors": counts["error"],
        "passed": counts["passed"],
        "skipped": counts["skipped"],
        "time": round(sum(c["time"] for c in parsed_cases), 2),
        "cases": parsed_cases,
    }


def markdown_from_report(
    report: Dict[str, Any],
    *,
    title: str = "AgentTest Report",
    max_failures: int = 10,
    include_marker: bool = True,
) -> str:
    """
    Render a parsed report as GitHub-flavoured Markdown.

    Args:
        report: Output of :func:`parse_junit`.
        title: Heading shown at the top of the comment.
        max_failures: Cap on individual failures shown; the rest are counted.
        include_marker: Prefix the hidden ``REPORT_MARKER`` so the Action can
            update an existing comment instead of duplicating it.

    Returns:
        A Markdown string, safe to post as a PR comment body.
    """
    total = report["tests"]
    passed = report["passed"]
    failed = report["failures"]
    errors = report["errors"]
    skipped = report["skipped"]
    pass_rate = (passed / total * 100) if total else 0.0

    lines: List[str] = []
    if include_marker:
        lines.append(REPORT_MARKER)
    lines.append(f"## {title}")
    lines.append("")
    lines.append("| Metric | Value |")
    lines.append("|--------|-------|")
    lines.append(f"| Passed | {passed} |")
    lines.append(f"| Failed | {failed} |")
    lines.append(f"| Errors | {errors} |")
    lines.append(f"| Skipped | {skipped} |")
    lines.append(f"| **Total** | {total} |")
    lines.append(f"| Pass rate | {pass_rate:.1f}% |")
    lines.append(f"| Duration | {report['time']}s |")
    lines.append("")

    problems = [c for c in report["cases"] if c["status"] in ("failed", "error")]
    if not problems:
        lines.append("All agent tests passed.")
        return "\n".join(lines)

    shown = problems[:max_failures]
    hidden = len(problems) - len(shown)
    summary_label = f"{len(problems)} problem(s)"
    lines.append(f"<details><summary>{summary_label}</summary>")
    lines.append("")
    for case in shown:
        label = f"{case['classname']}::{case['name']}" if case["classname"] else case["name"]
        lines.append(f"- `{label}` ({case['status']})")
        if case["failure"]:
            lines.append("")
            lines.append("  ```")
            for detail_line in case["failure"].splitlines()[:40]:
                lines.append(f"  {detail_line}")
            lines.append("  ```")
    if hidden > 0:
        lines.append("")
        lines.append(f"... and {hidden} more. See the workflow log for the full output.")
    lines.append("")
    lines.append("</details>")
    return "\n".join(lines)


def markdown_from_junit(
    path: str,
    *,
    title: str = "AgentTest Report",
    max_failures: int = 10,
    include_marker: bool = True,
) -> str:
    """Convenience wrapper: parse a JUnit XML file and render it as Markdown."""
    return markdown_from_report(
        parse_junit(path),
        title=title,
        max_failures=max_failures,
        include_marker=include_marker,
    )


def main(argv: Optional[List[str]] = None) -> int:
    """
    CLI entry point: ``python -m agenttest.ci_summary <junit.xml>``.

    Prints the Markdown block to stdout. Returns exit code 1 when the report
    contains failures or errors, so the step can gate a workflow even when the
    Markdown is consumed downstream.
    """
    args = sys.argv[1:] if argv is None else argv
    if not args or args[0] in ("-h", "--help"):
        print(__doc__)
        return 0 if args else 2

    path = args[0]
    title = "AgentTest Report"
    for index, arg in enumerate(args):
        if arg in ("--title", "-t") and index + 1 < len(args):
            title = args[index + 1]

    try:
        markdown = markdown_from_junit(path, title=title)
    except (FileNotFoundError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    print(markdown)
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI
    # Exit non-zero when there were problems, so `set -e` CI steps fail fast.
    _args = sys.argv[1:]
    if not _args or _args[0] in ("-h", "--help"):
        raise SystemExit(main())
    try:
        _report = parse_junit(_args[0])
    except (FileNotFoundError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(2)
    raise SystemExit(0 if (_report["failures"] == 0 and _report["errors"] == 0) else 1)


__all__ = [
    "REPORT_MARKER",
    "parse_junit",
    "markdown_from_report",
    "markdown_from_junit",
    "main",
]
