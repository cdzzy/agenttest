"""
Behavior assertions — verify tool call patterns.

These assertions answer: "Did the agent use the right tools in the right order?"
"""
from __future__ import annotations

from typing import List, Optional

from agenttest.core.case import AgentRun


def assert_tool_called(run: AgentRun, tool_name: str, times: Optional[int] = None):
    """
    Assert that a specific tool was called during the agent run.

    Args:
        run: The AgentRun to inspect.
        tool_name: Name of the expected tool.
        times: If provided, assert the tool was called exactly this many times.

    Raises:
        AssertionError: If the assertion fails.

    Example:
        assert_tool_called(run, "web_search")
        assert_tool_called(run, "calculator", times=2)
    """
    actual_calls = run.tool_names
    call_count = actual_calls.count(tool_name)

    if times is not None:
        if call_count != times:
            raise AssertionError(
                f"Expected tool '{tool_name}' to be called {times} time(s), "
                f"but it was called {call_count} time(s).\n"
                f"All tools called: {actual_calls}"
            )
    else:
        if call_count == 0:
            raise AssertionError(
                f"Expected tool '{tool_name}' to be called, but it was not.\n"
                f"All tools called: {actual_calls}"
            )


def assert_no_tool_called(run: AgentRun, tool_name: str):
    """
    Assert that a specific tool was NOT called during the agent run.

    Example:
        assert_no_tool_called(run, "dangerous_delete_tool")
    """
    actual_calls = run.tool_names
    if tool_name in actual_calls:
        raise AssertionError(
            f"Expected tool '{tool_name}' to NOT be called, "
            f"but it was called {actual_calls.count(tool_name)} time(s).\n"
            f"All tools called: {actual_calls}"
        )


def assert_tool_sequence(run: AgentRun, sequence: List[str], strict: bool = False):
    """
    Assert that tools were called in a specific sequence (order matters).

    Args:
        run: The AgentRun to inspect.
        sequence: Expected ordered list of tool names.
        strict: If True, the sequence must be an exact match (no extra tools allowed).

    Example:
        # The agent must search, then summarize (may call other tools in between)
        assert_tool_sequence(run, ["web_search", "summarize"])

        # Strict: exactly these tools, exactly this order, nothing else
        assert_tool_sequence(run, ["search", "format"], strict=True)
    """
    actual = run.tool_names

    if strict:
        if actual != sequence:
            raise AssertionError(
                f"Tool sequence mismatch (strict mode).\n"
                f"Expected: {sequence}\n"
                f"Actual:   {actual}"
            )
        return

    # Non-strict: check subsequence
    seq_idx = 0
    for tool in actual:
        if seq_idx < len(sequence) and tool == sequence[seq_idx]:
            seq_idx += 1

    if seq_idx < len(sequence):
        missing = sequence[seq_idx]
        raise AssertionError(
            f"Tool sequence not satisfied. Expected subsequence {sequence}.\n"
            f"Got stuck at '{missing}' — not found after index {seq_idx}.\n"
            f"Actual call order: {actual}"
        )


def assert_tool_called_before(run: AgentRun, tool_a: str, tool_b: str):
    """
    Assert that tool_a was called before tool_b.

    Example:
        assert_tool_called_before(run, "search", "summarize")
    """
    actual = run.tool_names

    try:
        idx_a = actual.index(tool_a)
    except ValueError:
        raise AssertionError(
            f"Tool '{tool_a}' was never called. "
            f"Cannot assert it was called before '{tool_b}'.\n"
            f"Actual calls: {actual}"
        )

    try:
        idx_b = actual.index(tool_b)
    except ValueError:
        raise AssertionError(
            f"Tool '{tool_b}' was never called.\n"
            f"Actual calls: {actual}"
        )

    if idx_a >= idx_b:
        raise AssertionError(
            f"Expected '{tool_a}' (at index {idx_a}) to be called before "
            f"'{tool_b}' (at index {idx_b}), but it wasn't.\n"
            f"Actual call order: {actual}"
        )


def assert_max_tool_calls(run: AgentRun, max_calls: int):
    """
    Assert the agent didn't make an excessive number of tool calls.
    Useful for catching runaway agents or infinite loops.

    Example:
        assert_max_tool_calls(run, max_calls=10)
    """
    if run.tool_call_count > max_calls:
        raise AssertionError(
            f"Agent made {run.tool_call_count} tool calls, "
            f"exceeding the maximum of {max_calls}.\n"
            f"Tools called: {run.tool_names}"
        )

