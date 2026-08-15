"""
Core test case definition for AgentTest.
"""
from __future__ import annotations

import functools
import inspect
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Union


class TestStatus(Enum):
    PENDING = "pending"
    RUNNING = "running"
    PASSED = "passed"
    FAILED = "failed"
    SKIPPED = "skipped"
    ERROR = "error"


@dataclass
class AgentRun:
    """Captures a single agent execution result."""
    input: str
    output: str
    tool_calls: List[Dict[str, Any]] = field(default_factory=list)
    reasoning_steps: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    duration_ms: float = 0.0
    tokens_used: int = 0
    error: Optional[Exception] = None

    @property
    def tool_names(self) -> List[str]:
        """Return ordered list of tool names that were called."""
        return [call.get("name", "") for call in self.tool_calls]

    @property
    def tool_call_count(self) -> int:
        return len(self.tool_calls)

    @property
    def succeeded(self) -> bool:
        return self.error is None


@dataclass
class AssertionResult:
    """Result of a single assertion."""
    assertion_name: str
    passed: bool
    message: str
    details: Dict[str, Any] = field(default_factory=dict)


@dataclass
class TestResult:
    """Full result of a test case execution."""
    test_name: str
    status: TestStatus
    agent_run: Optional[AgentRun] = None
    assertion_results: List[AssertionResult] = field(default_factory=list)
    duration_ms: float = 0.0
    error_message: Optional[str] = None

    @property
    def passed(self) -> bool:
        return self.status == TestStatus.PASSED

    @property
    def failed_assertions(self) -> List[AssertionResult]:
        return [a for a in self.assertion_results if not a.passed]


class AgentTestCase:
    """
    Base class for Agent test cases.

    Usage:
        class MyAgentTest(AgentTestCase):
            def test_basic_query(self):
                run = self.invoke("What is the capital of France?")
                self.assert_output_contains(run, "Paris")
                self.assert_tool_called(run, "search")
    """

    # Subclasses should set this to the agent callable
    agent: Optional[Callable] = None

    def setUp(self):
        """Override to add setup logic before each test."""
        pass

    def tearDown(self):
        """Override to add teardown logic after each test."""
        pass

    def invoke(self, input_text: str, **kwargs) -> AgentRun:
        """
        Invoke the agent with the given input and capture the result.

        Args:
            input_text: The user input to send to the agent.
            **kwargs: Additional kwargs passed to the agent.

        Returns:
            AgentRun capturing all execution details.
        """
        # Retrieve the raw callable from the class hierarchy to avoid Python's
        # descriptor protocol binding the function as an instance method.
        # When `agent = some_function` is set as a class attribute, accessing
        # `self.agent` would bind `self` as the first argument, turning
        # `agent(input_text)` into `agent(self, input_text)` — which is wrong.
        raw_agent = None
        for cls in type(self).__mro__:
            if "agent" in cls.__dict__:
                raw_agent = cls.__dict__["agent"]
                break
        if raw_agent is None:
            raw_agent = self.__dict__.get("agent")  # handles dynamic assignment via self.agent = ...
        if raw_agent is None:
            raise ValueError(
                "AgentTestCase.agent must be set before calling invoke(). "
                "Either set it as a class attribute or pass it in setUp()."
            )

        start = time.perf_counter()
        run = AgentRun(input=input_text, output="")

        try:
            result = raw_agent(input_text, **kwargs)

            # Handle different agent return formats
            if isinstance(result, dict):
                run.output = result.get("output", result.get("response", str(result)))
                run.tool_calls = result.get("tool_calls", result.get("intermediate_steps", []))
                run.reasoning_steps = result.get("reasoning_steps", result.get("thoughts", []))
                run.tokens_used = result.get("usage", {}).get("total_tokens", 0)
                run.metadata = {k: v for k, v in result.items()
                                if k not in ("output", "tool_calls", "reasoning_steps")}
            elif isinstance(result, str):
                run.output = result
            else:
                run.output = str(result)

        except Exception as e:
            run.error = e

        run.duration_ms = (time.perf_counter() - start) * 1000
        return run

    # ── Assertion helpers ──────────────────────────────────────────────────
    def assert_tool_called(self, run: AgentRun, tool_name: str, times: Optional[int] = None):
        from agenttest.assertions.behavior import assert_tool_called as _assert
        _assert(run, tool_name, times)

    def assert_tool_sequence(self, run: AgentRun, sequence: List[str]):
        from agenttest.assertions.behavior import assert_tool_sequence as _assert
        _assert(run, sequence)

    def assert_no_tool_called(self, run: AgentRun, tool_name: str):
        from agenttest.assertions.behavior import assert_no_tool_called as _assert
        _assert(run, tool_name)

    def assert_max_tool_calls(self, run: AgentRun, max_calls: int):
        from agenttest.assertions.behavior import assert_max_tool_calls as _assert
        _assert(run, max_calls)

    def assert_output_contains(self, run: AgentRun, text: str, case_sensitive: bool = False):
        from agenttest.assertions.output import assert_output_contains as _assert
        _assert(run, text, case_sensitive)

    def assert_output_matches(self, run: AgentRun, pattern: str):
        from agenttest.assertions.output import assert_output_matches as _assert
        _assert(run, pattern)

    def assert_output_length(self, run: AgentRun, min_chars: int = 0, max_chars: Optional[int] = None):
        from agenttest.assertions.output import assert_output_length as _assert
        _assert(run, min_chars, max_chars)

    def assert_output_json(self, run: AgentRun):
        from agenttest.assertions.output import assert_output_json as _assert
        _assert(run)

    def assert_step_count(self, run: AgentRun, min_steps: int = 0, max_steps: Optional[int] = None):
        from agenttest.assertions.trace import assert_step_count as _assert
        _assert(run, min_steps, max_steps)

    def assert_no_error(self, run: AgentRun):
        if run.error is not None:
            raise AssertionError(
                f"Agent raised an unexpected error: {type(run.error).__name__}: {run.error}"
            )

    def assert_response_time(self, run: AgentRun, max_ms: float):
        if run.duration_ms > max_ms:
            raise AssertionError(
                f"Agent response took {run.duration_ms:.1f}ms, "
                f"exceeding max allowed {max_ms}ms"
            )


def agent_test(
    name: Optional[str] = None,
    tags: Optional[List[str]] = None,
    skip: bool = False,
    skip_reason: str = "",
    repeat: int = 1,
):
    """
    Decorator to mark a function as an agent test.

    Usage:
        @agent_test(tags=["smoke", "critical"])
        def test_my_agent(agent):
            run = agent("Hello!")
            assert "hi" in run.output.lower()

        @agent_test(repeat=5)  # run 5 times for stability testing
        def test_stability(agent):
            run = agent("What is 2+2?")
            assert "4" in run.output
    """
    def decorator(fn: Callable) -> Callable:
        fn._is_agent_test = True
        fn._test_name = name or fn.__name__
        fn._test_tags = tags or []
        fn._skip = skip
        fn._skip_reason = skip_reason
        fn._repeat = repeat

        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            return fn(*args, **kwargs)

        wrapper._is_agent_test = True
        wrapper._test_name = fn._test_name
        wrapper._test_tags = fn._test_tags
        wrapper._skip = fn._skip
        wrapper._skip_reason = fn._skip_reason
        wrapper._repeat = fn._repeat
        return wrapper

    # Allow use as @agent_test without parentheses
    if callable(name):
        fn = name
        name = None
        return decorator(fn)

    return decorator


def test_async(
    name: Optional[str] = None,
    tags: Optional[List[str]] = None,
    skip: bool = False,
    skip_reason: str = "",
    repeat: int = 1,
):
    """
    Decorator marking an async agent test function.

    The runner auto-detects ``async def`` functions, so this is mostly a
    semantic alias of :func:`agent_test` for async tests::

        @test_async
        async def test_async_agent(agent):
            run = await agent("What is 2+2?")
            assert "4" in run.output
    """
    def decorator(fn: Callable) -> Callable:
        fn = agent_test(name=name, tags=tags, skip=skip,
                        skip_reason=skip_reason, repeat=repeat)(fn)
        fn._is_async = True
        return fn

    if callable(name):
        fn = name
        name = None
        return decorator(fn)

    return decorator

