"""
Self-tests for agenttest — verifying the framework itself works.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
from agenttest.core.case import AgentRun, AgentTestCase
from agenttest.assertions.behavior import (
    assert_tool_called, assert_no_tool_called,
    assert_tool_sequence, assert_tool_called_before, assert_max_tool_calls,
)
from agenttest.assertions.output import (
    assert_output_contains, assert_output_not_contains,
    assert_output_matches, assert_output_length,
)
from agenttest.assertions.trace import assert_step_count, assert_reasoning_step
from agenttest.fixtures.mock_tools import MockTool, MockToolkit


def make_run(**kwargs) -> AgentRun:
    defaults = dict(input="test", output="hello world")
    defaults.update(kwargs)
    return AgentRun(**defaults)


# ── Behavior assertions ────────────────────────────────────────────────

def test_assert_tool_called_passes():
    run = make_run(tool_calls=[{"name": "search"}])
    assert_tool_called(run, "search")  # should not raise


def test_assert_tool_called_fails():
    run = make_run(tool_calls=[])
    with pytest.raises(AssertionError, match="search"):
        assert_tool_called(run, "search")


def test_assert_tool_called_exact_count():
    run = make_run(tool_calls=[{"name": "search"}, {"name": "search"}])
    assert_tool_called(run, "search", times=2)
    with pytest.raises(AssertionError):
        assert_tool_called(run, "search", times=1)


def test_assert_no_tool_called_passes():
    run = make_run(tool_calls=[])
    assert_no_tool_called(run, "dangerous_tool")


def test_assert_no_tool_called_fails():
    run = make_run(tool_calls=[{"name": "dangerous_tool"}])
    with pytest.raises(AssertionError, match="dangerous_tool"):
        assert_no_tool_called(run, "dangerous_tool")


def test_assert_tool_sequence_passes():
    run = make_run(tool_calls=[{"name": "a"}, {"name": "b"}, {"name": "c"}])
    assert_tool_sequence(run, ["a", "b"])
    assert_tool_sequence(run, ["a", "c"])
    assert_tool_sequence(run, ["a", "b", "c"])


def test_assert_tool_sequence_fails():
    run = make_run(tool_calls=[{"name": "b"}, {"name": "a"}])
    with pytest.raises(AssertionError):
        assert_tool_sequence(run, ["a", "b"])  # a must come before b


def test_assert_tool_sequence_strict():
    run = make_run(tool_calls=[{"name": "a"}, {"name": "b"}])
    assert_tool_sequence(run, ["a", "b"], strict=True)
    with pytest.raises(AssertionError):
        assert_tool_sequence(run, ["a"], strict=True)  # extra tool b


def test_assert_tool_called_before():
    run = make_run(tool_calls=[{"name": "search"}, {"name": "summarize"}])
    assert_tool_called_before(run, "search", "summarize")
    with pytest.raises(AssertionError):
        assert_tool_called_before(run, "summarize", "search")


def test_assert_max_tool_calls():
    run = make_run(tool_calls=[{"name": "a"}, {"name": "b"}])
    assert_max_tool_calls(run, 5)
    with pytest.raises(AssertionError):
        assert_max_tool_calls(run, 1)


# ── Output assertions ─────────────────────────────────────────────────

def test_assert_output_contains():
    run = make_run(output="The capital of France is Paris.")
    assert_output_contains(run, "Paris")
    assert_output_contains(run, "paris")  # case-insensitive by default
    with pytest.raises(AssertionError):
        assert_output_contains(run, "Berlin")


def test_assert_output_not_contains():
    run = make_run(output="I am happy to help.")
    assert_output_not_contains(run, "Berlin")
    with pytest.raises(AssertionError):
        assert_output_not_contains(run, "happy")


def test_assert_output_matches():
    run = make_run(output="The date is 2026-03-14.")
    assert_output_matches(run, r"\d{4}-\d{2}-\d{2}")
    with pytest.raises(AssertionError):
        assert_output_matches(run, r"^\d+$")


def test_assert_output_length():
    run = make_run(output="Hello world!")
    assert_output_length(run, min_chars=5)
    assert_output_length(run, min_chars=5, max_chars=100)
    with pytest.raises(AssertionError):
        assert_output_length(run, min_chars=100)
    with pytest.raises(AssertionError):
        assert_output_length(run, max_chars=3)


# ── Trace assertions ──────────────────────────────────────────────────

def test_assert_step_count():
    run = make_run(reasoning_steps=["think", "act", "done"])
    assert_step_count(run, min_steps=1, max_steps=5)
    with pytest.raises(AssertionError):
        assert_step_count(run, min_steps=10)
    with pytest.raises(AssertionError):
        assert_step_count(run, max_steps=1)


def test_assert_reasoning_step():
    run = make_run(reasoning_steps=["I need to search for this."])
    assert_reasoning_step(run, "search")
    with pytest.raises(AssertionError):
        assert_reasoning_step(run, "calculate")


# ── MockTool ──────────────────────────────────────────────────────────

def test_mock_tool_records_calls():
    tool = MockTool("search", returns="result")
    assert tool.call_count == 0
    result = tool("query")
    assert result == "result"
    assert tool.call_count == 1
    assert tool.last_call["args"] == ("query",)


def test_mock_tool_assert_called():
    tool = MockTool("search", returns="ok")
    with pytest.raises(AssertionError):
        tool.assert_called()
    tool("x")
    tool.assert_called()  # now passes


def test_mock_toolkit():
    tk = MockToolkit()
    tk.add("search", returns="found it")
    tk.add("calc", returns="42")

    tk["search"]("query")
    assert tk.call_summary == {"search": 1, "calc": 0}

    tk.assert_tool_not_called("calc")
    with pytest.raises(AssertionError):
        tk.assert_tool_not_called("search")

