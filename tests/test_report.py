"""
Tests for HTML/JSON test reports (v0.3.0).
"""

import json
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agenttest.core.case import TestResult, TestStatus
from agenttest.report import (
    build_report,
    report_to_json,
    report_to_html,
    save_report,
)


def _results():
    return [
        TestResult(test_name="test_ok", status=TestStatus.PASSED, duration_ms=10.0),
        TestResult(test_name="test_bad", status=TestStatus.FAILED, duration_ms=5.0, error_message="assert failed"),
        TestResult(test_name="test_err", status=TestStatus.ERROR, duration_ms=3.0, error_message="boom"),
        TestResult(test_name="test_skip", status=TestStatus.SKIPPED, duration_ms=0.0),
    ]


class TestBuildReport:
    def test_summary_counts(self):
        report = build_report(_results())
        s = report["summary"]
        assert s["passed"] == 1
        assert s["failed"] == 1
        assert s["errors"] == 1
        assert s["skipped"] == 1
        assert s["total"] == 4
        assert s["pass_rate"] == 25.0

    def test_results_include_errors(self):
        report = build_report(_results())
        errors = [r for r in report["results"] if r["status"] == "error"]
        assert errors[0]["error"] == "boom"


class TestReportToJson:
    def test_valid_json(self):
        data = json.loads(report_to_json(_results()))
        assert data["summary"]["total"] == 4
        assert len(data["results"]) == 4


class TestReportToHtml:
    def test_contains_summary(self):
        html_text = report_to_html(_results())
        assert "<html" in html_text
        assert "passed" in html_text
        assert "test_bad" in html_text
        assert "assert failed" in html_text

    def test_escapes_error_html(self):
        results = [TestResult(test_name="x", status=TestStatus.ERROR, error_message="<script>alert(1)</script>")]
        html_text = report_to_html(results)
        assert "<script>" not in html_text
        assert "&lt;script&gt;" in html_text


class TestSaveReport:
    def test_save_html_and_json(self, tmp_path):
        html_path = save_report(_results(), str(tmp_path / "r.html"))
        assert os.path.isfile(html_path)
        assert html_path.endswith(".html")
        with open(html_path, encoding="utf-8") as f:
            assert "<html" in f.read()

        json_path = save_report(_results(), str(tmp_path / "r.json"))
        with open(json_path, encoding="utf-8") as f:
            assert json.loads(f.read())["summary"]["total"] == 4

    def test_infer_format_from_extension(self, tmp_path):
        p = save_report(_results(), str(tmp_path / "auto.json"))
        assert os.path.isfile(p)
        assert p.endswith(".json")

    def test_unsupported_format_raises(self, tmp_path):
        import pytest
        with pytest.raises(ValueError):
            save_report(_results(), str(tmp_path / "r.txt"))


class TestRunnerIntegration:
    def test_run_suite_with_report(self, tmp_path):
        from agenttest.core.runner import AgentTestRunner
        from agenttest.core.suite import AgentTestSuite

        suite = AgentTestSuite("demo")
        suite.add_function(lambda agent: None, agent=lambda m: {"output": "ok"}, name="passing_test")
        suite.add_function(lambda agent: None, agent=lambda m: {"output": "ok"}, name="other_test")

        runner = AgentTestRunner(verbose=False)
        results = runner.run_suite_with_report(suite, str(tmp_path / "report.html"))
        assert len(results) == 2
        out = tmp_path / "report.html"
        assert out.exists()
        with open(out, encoding="utf-8") as f:
            assert "passing_test" in f.read()