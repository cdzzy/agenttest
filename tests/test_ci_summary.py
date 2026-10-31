"""
Tests for the JUnit XML -> Markdown CI summary used by the GitHub Action.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import pytest

from agenttest.ci_summary import (
    REPORT_MARKER,
    main,
    markdown_from_junit,
    markdown_from_report,
    parse_junit,
)

SINGLE_SUITE = """<?xml version="1.0" encoding="utf-8"?>
<testsuite name="pytest" tests="4" failures="1" errors="0" skipped="1">
  <testcase classname="tests.test_agent" name="test_greets" time="0.12"/>
  <testcase classname="tests.test_agent" name="test_refuses_pii" time="0.08">
    <failure message="AssertionError: agent leaked PII">Traceback (most recent call last):
  File "tests/test_agent.py", line 12, in test_refuses_pii
    assert result.assertion_passed
</failure>
  </testcase>
  <testcase classname="tests.test_agent" name="test_skipped" time="0.0">
    <skipped message="needs an LLM key"/>
  </testcase>
  <testcase classname="tests.test_agent" name="test_structured" time="0.30"/>
</testsuite>
"""

MULTI_SUITE = """<?xml version="1.0" encoding="utf-8"?>
<testsuites>
  <testsuite name="suite-a">
    <testcase classname="a.test_x" name="test_one" time="0.1"/>
  </testsuite>
  <testsuite name="suite-b">
    <testcase classname="b.test_y" name="test_two" time="0.2">
      <error message="RuntimeError: boom">trace</error>
    </testcase>
  </testsuite>
</testsuites>
"""


def _write(tmp_path, name, content):
    path = tmp_path / name
    path.write_text(content, encoding="utf-8")
    return str(path)


class TestParseJunit:
    def test_counts_a_single_suite(self, tmp_path):
        report = parse_junit(_write(tmp_path, "r.xml", SINGLE_SUITE))
        assert report["tests"] == 4
        assert report["passed"] == 2
        assert report["failures"] == 1
        assert report["errors"] == 0
        assert report["skipped"] == 1

    def test_sums_durations(self, tmp_path):
        report = parse_junit(_write(tmp_path, "r.xml", SINGLE_SUITE))
        assert report["time"] == pytest.approx(0.5)

    def test_captures_failure_text(self, tmp_path):
        report = parse_junit(_write(tmp_path, "r.xml", SINGLE_SUITE))
        failed = [c for c in report["cases"] if c["status"] == "failed"][0]
        assert "AssertionError: agent leaked PII" in failed["failure"]
        assert "test_agent.py" in failed["failure"]

    def test_case_metadata(self, tmp_path):
        report = parse_junit(_write(tmp_path, "r.xml", SINGLE_SUITE))
        first = report["cases"][0]
        assert first["name"] == "test_greets"
        assert first["classname"] == "tests.test_agent"
        assert first["status"] == "passed"
        assert first["failure"] is None

    def test_handles_multiple_suites(self, tmp_path):
        report = parse_junit(_write(tmp_path, "r.xml", MULTI_SUITE))
        assert report["tests"] == 2
        assert report["passed"] == 1
        assert report["errors"] == 1

    def test_missing_file_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            parse_junit(str(tmp_path / "nope.xml"))

    def test_invalid_xml_raises_value_error(self, tmp_path):
        with pytest.raises(ValueError, match="Not valid JUnit XML"):
            parse_junit(_write(tmp_path, "bad.xml", "<not-xml"))

    def test_unsupported_root_raises(self, tmp_path):
        with pytest.raises(ValueError, match="Unsupported JUnit root"):
            parse_junit(_write(tmp_path, "weird.xml", "<html></html>"))


class TestMarkdownRendering:
    def test_starts_with_marker_for_idempotent_comments(self, tmp_path):
        md = markdown_from_junit(_write(tmp_path, "r.xml", SINGLE_SUITE))
        assert md.startswith(REPORT_MARKER)

    def test_marker_can_be_omitted(self, tmp_path):
        md = markdown_from_junit(
            _write(tmp_path, "r.xml", SINGLE_SUITE), include_marker=False
        )
        assert REPORT_MARKER not in md

    def test_renders_summary_table(self, tmp_path):
        md = markdown_from_junit(_write(tmp_path, "r.xml", SINGLE_SUITE))
        assert "| Passed | 2 |" in md
        assert "| Failed | 1 |" in md
        assert "| Errors | 0 |" in md
        assert "| Skipped | 1 |" in md
        assert "| **Total** | 4 |" in md
        assert "| Pass rate | 50.0% |" in md

    def test_includes_failure_details(self, tmp_path):
        md = markdown_from_junit(_write(tmp_path, "r.xml", SINGLE_SUITE))
        assert "<details>" in md
        assert "tests.test_agent::test_refuses_pii" in md
        assert "AssertionError: agent leaked PII" in md

    def test_all_green_report(self):
        report = {
            "tests": 2, "passed": 2, "failures": 0, "errors": 0,
            "skipped": 0, "time": 0.2, "cases": [],
        }
        md = markdown_from_report(report)
        assert "All agent tests passed." in md
        assert "<details>" not in md

    def test_caps_displayed_failures(self):
        cases = [
            {"name": f"test_{i}", "classname": "c", "time": 0.1,
             "status": "failed", "failure": "boom"}
            for i in range(5)
        ]
        report = {
            "tests": 5, "passed": 0, "failures": 5, "errors": 0,
            "skipped": 0, "time": 0.5, "cases": cases,
        }
        md = markdown_from_report(report, max_failures=2)
        assert "5 problem(s)" in md
        assert "... and 3 more" in md
        assert md.count("- `c::test_") == 2

    def test_handles_empty_report(self):
        report = {
            "tests": 0, "passed": 0, "failures": 0, "errors": 0,
            "skipped": 0, "time": 0, "cases": [],
        }
        md = markdown_from_report(report)
        assert "| Pass rate | 0.0% |" in md
        assert "All agent tests passed." in md

    def test_truncates_long_tracebacks(self, tmp_path):
        long_trace = "\n".join(f"line {i}" for i in range(80))
        xml = f"""<?xml version="1.0"?>
<testsuite name="s" tests="1" failures="1">
  <testcase classname="c" name="t" time="0.1">
    <failure message="boom">{long_trace}</failure>
  </testcase>
</testsuite>
"""
        md = markdown_from_junit(_write(tmp_path, "long.xml", xml))
        assert "line 0" in md
        assert "line 79" not in md


class TestCli:
    def test_prints_markdown(self, tmp_path, capsys):
        path = _write(tmp_path, "r.xml", SINGLE_SUITE)
        assert main([path]) == 0
        out = capsys.readouterr().out
        assert REPORT_MARKER in out
        assert "AgentTest Report" in out

    def test_custom_title(self, tmp_path, capsys):
        path = _write(tmp_path, "r.xml", SINGLE_SUITE)
        assert main([path, "--title", "Nightly Agent Quality"]) == 0
        out = capsys.readouterr().out
        assert "## Nightly Agent Quality" in out

    def test_help(self, capsys):
        assert main(["--help"]) == 0

    def test_no_args_returns_usage_error(self):
        assert main([]) == 2

    def test_missing_file_returns_error_code(self, tmp_path, capsys):
        assert main([str(tmp_path / "nope.xml")]) == 2
        err = capsys.readouterr().err
        assert "error:" in err
