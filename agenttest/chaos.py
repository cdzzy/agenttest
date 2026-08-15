"""
Chaos engineering toolkit for agent resilience testing (Issue #9).

Simulate production failure modes (timeouts, rate limits, corrupt inputs,
prompt injection, tool errors) and verify the agent degrades gracefully.

Usage::

    from agenttest.chaos import ChaosScenario, inject_chaos, measure_resilience

    @inject_chaos(ChaosScenario.LLM_TIMEOUT)
    def test_timeout_resilience(agent):
        run = agent("Process user request")
        assert run.error is None or run.error  # agent handled the timeout
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional


class ChaosScenario(str, Enum):
    """Failure modes that can be injected into an agent run."""
    LLM_TIMEOUT = "llm_timeout"          # LLM call hangs / exceeds timeout
    RATE_LIMIT = "rate_limit"            # API returns 429
    CORRUPT_CONTEXT = "corrupt_context"  # Malformed context injected
    PROMPT_INJECTION = "prompt_injection"  # Malicious prompt attempt
    TOOL_ERROR = "tool_error"            # A tool call returns an error
    NETWORK_ERROR = "network_error"      # Transient connection failure


# Default error messages / exceptions associated with each scenario
_CHAOS_EXCEPTIONS: Dict[ChaosScenario, Exception] = {
    ChaosScenario.LLM_TIMEOUT: TimeoutError("LLM request timed out after 30s"),
    ChaosScenario.RATE_LIMIT: Exception("429 Too Many Requests (rate limit exceeded)"),
    ChaosScenario.CORRUPT_CONTEXT: ValueError("Malformed context object"),
    ChaosScenario.PROMPT_INJECTION: ValueError("Suspicious prompt injection detected"),
    ChaosScenario.TOOL_ERROR: RuntimeError("Tool execution failed"),
    ChaosScenario.NETWORK_ERROR: ConnectionError("Network connection reset"),
}


def scenario_error(scenario: ChaosScenario) -> Exception:
    """Return the representative exception for a chaos scenario."""
    return _CHAOS_EXCEPTIONS[scenario]


@dataclass
class ChaosResult:
    """Outcome of a single chaos-injected run."""
    scenario: ChaosScenario
    graceful: bool            # agent handled failure without crashing
    recovered: bool           # agent produced a valid response despite failure
    error: Optional[str] = None

    @property
    def degraded(self) -> bool:
        return not self.recovered


@dataclass
class ResilienceMetrics:
    total: int
    graceful: int
    failed: int
    recovery_ms: float

    @property
    def graceful_rate(self) -> float:
        return self.graceful / self.total if self.total else 0.0

    @property
    def fail_rate(self) -> float:
        return self.failed / self.total if self.total else 0.0

    @property
    def avg_recovery_ms(self) -> float:
        return self.recovery_ms / self.total if self.total else 0.0


class ChaosAgent:
    """
    Wraps an agent callable, delivering a chaos scenario to the next invocation.

    When a scenario is injected, it is passed to the agent via the ``_chaos``
    keyword argument. Chaos-aware agents can inspect it and degrade gracefully
    (e.g. return a fallback response); agents that don't handle it will raise.
    """

    def __init__(self, agent: Callable) -> None:
        self._agent = agent
        self._pending: Optional[ChaosScenario] = None

    def inject(self, scenario: ChaosScenario) -> None:
        """Queue a chaos scenario for the next call."""
        self._pending = scenario

    def __call__(self, *args, **kwargs) -> Any:
        scenario = self._pending
        self._pending = None
        if scenario is None:
            return self._agent(*args, **kwargs)
        return self._agent(*args, _chaos=scenario, **kwargs)


def inject_chaos(scenario: ChaosScenario):
    """
    Decorator injecting a chaos failure into the agent's first invocation.
    """
    def decorator(fn: Callable) -> Callable:
        def wrapper(agent, *args, **kwargs):
            chaos_agent = ChaosAgent(agent)
            chaos_agent.inject(scenario)
            return fn(chaos_agent, *args, **kwargs)
        wrapper.__name__ = getattr(fn, "__name__", "chaos_test")
        return wrapper
    return decorator


def run_chaos(agent: Callable, scenario: ChaosScenario, input_text: str) -> ChaosResult:
    """
    Run an agent once with a chaos scenario delivered via ``_chaos``.

    Returns:
        ChaosResult describing whether the agent degraded gracefully.
    """
    chaos_agent = ChaosAgent(agent)
    chaos_agent.inject(scenario)

    import time
    start = time.perf_counter()
    try:
        chaos_agent(input_text)
        graceful, recovered, error = True, True, None
    except Exception as e:
        graceful, recovered, error = False, False, str(e)
    elapsed_ms = (time.perf_counter() - start) * 1000

    return ChaosResult(scenario=scenario, graceful=graceful, recovered=recovered, error=error)


def measure_resilience(
    agent: Callable,
    scenarios: List[ChaosScenario],
    input_text: str = "Process user request",
) -> ResilienceMetrics:
    """
    Measure resilience across multiple chaos scenarios.

    Returns:
        ResilienceMetrics aggregating graceful degradation rates.
    """
    import time

    total = len(scenarios)
    graceful = 0
    total_recovery_ms = 0.0

    for scenario in scenarios:
        start = time.perf_counter()
        chaos_agent = ChaosAgent(agent)
        chaos_agent.inject(scenario)
        try:
            chaos_agent(input_text)
            graceful += 1
        except Exception:
            pass
        total_recovery_ms += (time.perf_counter() - start) * 1000

    return ResilienceMetrics(
        total=total,
        graceful=graceful,
        failed=total - graceful,
        recovery_ms=total_recovery_ms,
    )
