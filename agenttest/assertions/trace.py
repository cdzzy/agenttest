"""
Trace assertions — verify the agent's reasoning process.

These assertions answer: "Did the agent think through the problem correctly?"
"""
from __future__ import annotations

from typing import List, Optional

from agenttest.core.case import AgentRun


def assert_reasoning_step(run: AgentRun, keyword: str, case_sensitive: bool = False):
    """
    Assert that a specific keyword appears in the agent's reasoning/thought steps.

    Example:
        assert_reasoning_step(run, "search for")
        assert_reasoning_step(run, "calculate")
    """
    if not run.reasoning_steps:
        raise AssertionError(
            f"No reasoning steps were captured. "
            f"Make sure your agent returns 'reasoning_steps' or 'thoughts' in its output dict."
        )

    for step in run.reasoning_steps:
        haystack = step if case_sensitive else step.lower()
        needle = keyword if case_sensitive else keyword.lower()
        if needle in haystack:
            return

    raise AssertionError(
        f"Expected to find '{keyword}' in agent reasoning steps, but didn't.\n"
        f"Reasoning steps:\n" + "\n".join(f"  [{i}] {s}" for i, s in enumerate(run.reasoning_steps))
    )


def assert_step_count(run: AgentRun, min_steps: int = 0, max_steps: Optional[int] = None):
    """
    Assert the number of reasoning steps is within bounds.
    Useful for detecting agents that over-think or under-think.

    Example:
        assert_step_count(run, min_steps=1, max_steps=5)
    """
    count = len(run.reasoning_steps)

    if count < min_steps:
        raise AssertionError(
            f"Expected at least {min_steps} reasoning step(s), "
            f"but only got {count}.\n"
            f"Steps: {run.reasoning_steps}"
        )

    if max_steps is not None and count > max_steps:
        raise AssertionError(
            f"Expected at most {max_steps} reasoning step(s), "
            f"but got {count} (possible overthinking).\n"
            f"Steps: {run.reasoning_steps}"
        )


def assert_no_reasoning_loops(run: AgentRun, max_repeated: int = 2):
    """
    Assert the agent didn't get stuck in a reasoning loop
    (same thought repeated too many times).

    Example:
        assert_no_reasoning_loops(run)
        assert_no_reasoning_loops(run, max_repeated=3)
    """
    if not run.reasoning_steps:
        return

    from collections import Counter
    counts = Counter(run.reasoning_steps)
    repeated = {step: cnt for step, cnt in counts.items() if cnt > max_repeated}

    if repeated:
        raise AssertionError(
            f"Agent appears to be looping in reasoning. "
            f"Steps repeated more than {max_repeated} times:\n"
            + "\n".join(f"  (x{cnt}) {step!r}" for step, cnt in repeated.items())
        )


def assert_tool_reasoning_alignment(run: AgentRun):
    """
    Assert that every tool call has a corresponding reasoning step that mentions it.
    Helps catch cases where the agent calls tools "silently" without explaining why.

    Example:
        assert_tool_reasoning_alignment(run)
    """
    if not run.tool_calls or not run.reasoning_steps:
        return

    reasoning_text = " ".join(run.reasoning_steps).lower()

    unmentioned = []
    for tool_name in run.tool_names:
        if tool_name.lower() not in reasoning_text:
            unmentioned.append(tool_name)

    if unmentioned:
        raise AssertionError(
            f"The following tools were called but not mentioned in reasoning steps: {unmentioned}\n"
            f"This may indicate the agent is calling tools without proper justification."
        )


def assert_reasoning_steps(run: AgentRun, min_steps: int = 1):
    """Assert the agent produced at least ``min_steps`` reasoning steps."""
    assert_step_count(run, min_steps=min_steps)


def assert_reasoning_covers(run: AgentRun, topics: List[str], case_sensitive: bool = False):
    """
    Assert that every topic keyword appears somewhere in the reasoning trace.

    Example:
        assert_reasoning_covers(run, ["risk_analysis", "rollback_plan"])
    """
    reasoning_text = "\n".join(run.reasoning_steps)
    if not case_sensitive:
        reasoning_text = reasoning_text.lower()

    missing = []
    for topic in topics:
        needle = topic if case_sensitive else topic.lower()
        if needle not in reasoning_text:
            missing.append(topic)

    if missing:
        raise AssertionError(
            f"Reasoning trace does not cover topics: {missing}.\n"
            f"Reasoning steps:\n" + "\n".join(f"  [{i}] {s}" for i, s in enumerate(run.reasoning_steps))
        )


def assert_reasoning_order(run: AgentRun, expected_order: List[str], case_sensitive: bool = False):
    """
    Assert topics appear in the reasoning trace in the given order.

    Example:
        assert_reasoning_order(run, ["analyze", "plan", "validate"])
    """
    reasoning_text = "\n".join(run.reasoning_steps)
    if not case_sensitive:
        reasoning_text = reasoning_text.lower()

    positions = []
    for topic in expected_order:
        needle = topic if case_sensitive else topic.lower()
        pos = reasoning_text.find(needle)
        if pos == -1:
            raise AssertionError(
                f"Topic '{topic}' not found in reasoning trace (expected order {expected_order})."
            )
        positions.append(pos)

    if positions != sorted(positions):
        raise AssertionError(
            f"Reasoning topics out of order. Expected {expected_order}, "
            f"but found them in a different sequence in the trace."
        )


def assert_reasoning_time(run: AgentRun, max_seconds: float):
    """
    Assert the agent completed reasoning within ``max_seconds``.

    Example:
        assert_reasoning_time(run, max_seconds=30)
    """
    max_ms = max_seconds * 1000
    if run.duration_ms > max_ms:
        raise AssertionError(
            f"Agent reasoning took {run.duration_ms/1000:.2f}s, "
            f"exceeding max allowed {max_seconds}s"
        )

