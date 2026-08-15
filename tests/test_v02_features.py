"""
Tests for v0.2.0 features:
- SuiteRunHistory (#8)
- Async support (#2)
- Snapshots (#1)
- Benchmarking (#5)
- Chaos engineering (#9)
- Flakiness detection (#3)
- Reasoning assertions (#6)
- MCP testing toolkit (#7)
"""

import json
import os
import sys

import pytest

from agenttest.core.case import TestResult as _TestResult, TestStatus as _TestStatus
from agenttest.core.case import agent_test as _agent_test, test_async as _test_async
from agenttest.core.runner import AgentTestRunner
from agenttest.core.history import SuiteRunHistory, RunSummary, RunTrendReport
from agenttest.snapshots import SnapshotStore, snapshot
from agenttest.benchmark import BenchmarkSuite, Scenario, BenchmarkResults
from agenttest.chaos import ChaosScenario, ChaosAgent, run_chaos, measure_resilience, scenario_error
from agenttest.flakiness import FlakinessDetector
from agenttest.assertions.trace import (
    assert_reasoning_steps,
    assert_reasoning_covers,
    assert_reasoning_order,
    assert_reasoning_time,
)
from agenttest.mcp_tools import MCPTool, assert_tool_exists, assert_tool_response_valid


def _result(status, name="t"):
    return _TestResult(test_name=name, status=status)


# ── SuiteRunHistory (#8) ───────────────────────────────────────────────

class TestSuiteRunHistory:

    def test_record_and_load(self, tmp_path):
        history = SuiteRunHistory(str(tmp_path / "hist.jsonl"), window=10)
        history.record([_result(_TestStatus.PASSED), _result(_TestStatus.PASSED), _result(_TestStatus.FAILED)])
        history.record([_result(_TestStatus.PASSED), _result(_TestStatus.PASSED), _result(_TestStatus.PASSED)])
        summaries = history.load()
        assert len(summaries) == 2
        assert summaries[0].pass_rate == pytest.approx(2 / 3)
        assert summaries[1].pass_rate == 1.0

    def test_analyze_requires_three_runs(self, tmp_path):
        history = SuiteRunHistory(str(tmp_path / "hist.jsonl"))
        history.record([_result(_TestStatus.PASSED)])
        assert history.analyze() is None

    def test_analyze_detects_degradation(self, tmp_path):
        history = SuiteRunHistory(str(tmp_path / "hist.jsonl"))
        # pass rates: 1.0, 0.5, 0.0 �?degrading slope
        history.record([_result(_TestStatus.PASSED)])
        history.record([_result(_TestStatus.PASSED), _result(_TestStatus.FAILED)])
        history.record([_result(_TestStatus.FAILED)])
        trend = history.analyze()
        assert trend is not None
        assert trend.direction == "degrading"
        assert trend.any_regression is True

    def test_analyze_detects_improvement(self, tmp_path):
        history = SuiteRunHistory(str(tmp_path / "hist.jsonl"))
        history.record([_result(_TestStatus.FAILED)])
        history.record([_result(_TestStatus.PASSED), _result(_TestStatus.FAILED)])
        history.record([_result(_TestStatus.PASSED)])
        trend = history.analyze()
        assert trend.direction == "improving"
        assert trend.any_regression is False

    def test_run_summary_pass_rate_skips_skipped(self):
        s = RunSummary(timestamp=0, passed=2, failed=0, errors=0, total=3, skipped=1)
        assert s.pass_rate == 1.0


# ── Async support (#2) ────────────────────────────────────────────────

class TestAsyncSupport:

    def test_async_function_no_agent(self):
        async def test_fn():
            assert 1 + 1 == 2
        runner = AgentTestRunner(verbose=False)
        result = runner.run_function(test_fn, agent=lambda m: m)
        assert result.status == _TestStatus.PASSED

    def test_async_function_with_sync_agent(self):
        async def test_fn(agent):
            run = agent("hello")
            assert "HELLO" in run.output
        runner = AgentTestRunner(verbose=False)
        result = runner.run_function(test_fn, agent=lambda m: m.upper())
        assert result.status == _TestStatus.PASSED

    def test_async_function_with_async_agent(self):
        async def async_agent(text):
            return {"output": f"echo:{text}"}

        async def test_fn(agent):
            run = await agent("hi")
            assert run.output == "echo:hi"

        runner = AgentTestRunner(verbose=False)
        result = runner.run_function(test_fn, agent=async_agent)
        assert result.status == _TestStatus.PASSED

    def test_sync_function_with_async_agent(self):
        async def async_agent(text):
            return {"output": text.upper()}

        def test_fn(agent):
            run = agent("hello")
            assert run.output == "HELLO"

        runner = AgentTestRunner(verbose=False)
        result = runner.run_function(test_fn, agent=async_agent)
        assert result.status == _TestStatus.PASSED

    def test_test_async_decorator(self):
        @_test_async
        async def t(agent):
            run = await agent("x")
            assert run.output == "x"
        assert t._is_async is True


# ── Snapshots (#1) ─────────────────────────────────────────────────────

class TestSnapshots:

    def test_first_run_stores(self, tmp_path):
        store = SnapshotStore(str(tmp_path / "snaps"))
        store.assert_matches("foo", {"a": 1})
        assert (tmp_path / "snaps" / "foo.json").exists()

    def test_match_passes(self, tmp_path):
        store = SnapshotStore(str(tmp_path / "snaps"))
        store.assert_matches("foo", {"a": 1})
        store.assert_matches("foo", {"a": 1})  # no error

    def test_mismatch_raises(self, tmp_path):
        store = SnapshotStore(str(tmp_path / "snaps"))
        store.assert_matches("foo", {"a": 1})
        with pytest.raises(AssertionError):
            store.assert_matches("foo", {"a": 2})

    def test_update_mode(self, tmp_path):
        store = SnapshotStore(str(tmp_path / "snaps"), update=True)
        store.assert_matches("foo", {"a": 1})
        store.assert_matches("foo", {"a": 2})  # overwrites, no error
        assert store.load("foo") == {"a": 2}

    def test_fuzzy_similarity(self, tmp_path):
        store = SnapshotStore(str(tmp_path / "snaps"), similarity_threshold=0.8)
        store.assert_matches("s", "The quick brown fox jumps over the lazy dog")
        store.assert_matches("s", "The quick brown fox jumps over the lazy cat")  # similar


# ── Benchmarking (#5) ──────────────────────────────────────────────────

class TestBenchmark:

    def _suite(self):
        return BenchmarkSuite(name="b", scenarios=[
            Scenario("greet", "hello", evaluate=lambda r: "hi" in r.output.lower()),
            Scenario("math", "2+2", evaluate=lambda r: "4" in r.output),
        ])

    def test_run_and_report(self):
        suite = self._suite()
        agents = {
            "good": lambda m: {"output": "hi there, 2+2=4"},
            "bad": lambda m: {"output": "nope"},
        }
        results = suite.run(agents, runs_per_scenario=2)
        assert results.averages["good"] == 1.0
        assert results.averages["bad"] == 0.0

    def test_markdown_report(self):
        suite = self._suite()
        agents = {"good": lambda m: {"output": "hi, 4"}}
        results = suite.run(agents)
        report = suite.generate_report(results, format="markdown")
        assert "good" in report
        assert "100%" in report

    def test_text_report(self):
        suite = self._suite()
        results = suite.run({"a": lambda m: {"output": "x"}})
        report = suite.generate_report(results, format="text")
        assert "Benchmark" in report


# ── Chaos engineering (#9) ─────────────────────────────────────────────

class TestChaos:

    def test_scenario_error(self):
        assert isinstance(scenario_error(ChaosScenario.LLM_TIMEOUT), TimeoutError)

    def test_chaos_agent_injects_once(self):
        def agent(text, _chaos=None):
            if _chaos is not None:
                raise RuntimeError(f"chaos {_chaos.value}")
            return {"output": "ok"}

        chaos = ChaosAgent(agent)
        chaos.inject(ChaosScenario.RATE_LIMIT)
        with pytest.raises(RuntimeError):
            chaos("hi")
        # second call is not injected
        assert chaos("hi") == {"output": "ok"}

    def test_run_chaos_reports_failure(self):
        result = run_chaos(lambda m: {"output": "ok"}, ChaosScenario.TOOL_ERROR, "hi")
        assert result.graceful is False
        assert result.scenario == ChaosScenario.TOOL_ERROR

    def test_measure_resilience(self):
        # Chaos-aware agent that degrades gracefully
        def resilient_agent(text, _chaos=None):
            if _chaos is not None:
                return {"output": f"degraded due to {_chaos.value}"}
            return {"output": "ok"}

        metrics = measure_resilience(
            resilient_agent,
            [ChaosScenario.TOOL_ERROR, ChaosScenario.LLM_TIMEOUT],
        )
        assert metrics.total == 2
        assert metrics.graceful == 2
        assert metrics.graceful_rate == 1.0

    def test_measure_resilience_non_resilient(self):
        # Agent that doesn't accept _chaos → fails
        def brittle_agent(text):
            return {"output": "ok"}

        metrics = measure_resilience(brittle_agent, [ChaosScenario.TOOL_ERROR])
        assert metrics.graceful == 0


# ── Flakiness detection (#3) ───────────────────────────────────────────

class TestFlakiness:

    def test_flaky_detection(self):
        detector = FlakinessDetector(threshold=0.2, min_runs=3)
        for _ in range(5):
            detector.record([
                _result(_TestStatus.PASSED, "stable"),
                _result(_TestStatus.PASSED if _ % 2 == 0 else _TestStatus.FAILED, "flaky"),
            ])
        flaky = detector.flaky_tests()
        assert any(r.test_name == "flaky" for r in flaky)
        assert all(r.test_name != "stable" for r in flaky)

    def test_min_runs_gate(self):
        detector = FlakinessDetector(min_runs=5)
        detector.record([_result(_TestStatus.FAILED, "x")])
        assert detector.report() == []


# ── Reasoning assertions (#6) ──────────────────────────────────────────

class TestReasoningAssertions:

    def _run(self, steps=None, output="ok"):
        from agenttest.core.case import AgentRun
        return AgentRun(input="x", output=output, reasoning_steps=steps or [])

    def test_assert_reasoning_steps(self):
        with pytest.raises(AssertionError):
            assert_reasoning_steps(self._run([]), min_steps=2)
        assert_reasoning_steps(self._run(["a", "b"]), min_steps=2)

    def test_assert_reasoning_covers(self):
        run = self._run(["analyze risks", "plan rollback"])
        assert_reasoning_covers(run, ["risks", "rollback"])
        with pytest.raises(AssertionError):
            assert_reasoning_covers(run, ["testing"])

    def test_assert_reasoning_order(self):
        run = self._run(["first analyze", "then plan", "finally validate"])
        assert_reasoning_order(run, ["analyze", "plan", "validate"])
        with pytest.raises(AssertionError):
            assert_reasoning_order(run, ["plan", "analyze"])

    def test_assert_reasoning_time(self):
        run = self._run(["step"])
        run.duration_ms = 500
        assert_reasoning_time(run, max_seconds=1.0)
        with pytest.raises(AssertionError):
            assert_reasoning_time(run, max_seconds=0.1)


# ── MCP testing toolkit (#7) ───────────────────────────────────────────

class TestMCPTools:

    def test_mcp_tool_from_dict(self):
        tool = MCPTool.from_dict({"name": "brave_web_search", "description": "d", "inputSchema": {"type": "object"}})
        assert tool.name == "brave_web_search"
        assert tool.input_schema == {"type": "object"}

    def test_assert_tool_exists(self):
        tools = [MCPTool(name="a"), MCPTool(name="b")]
        assert_tool_exists(tools, "a")
        with pytest.raises(AssertionError):
            assert_tool_exists(tools, "zzz")

    def test_assert_tool_response_valid(self):
        assert_tool_response_valid({"content": [{"type": "text", "text": "hi"}]})
        with pytest.raises(AssertionError):
            assert_tool_response_valid({"isError": True, "content": []})
