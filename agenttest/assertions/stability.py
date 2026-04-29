"""
Stability assertions — verify the agent behaves consistently across multiple runs.

These assertions answer: "Is the agent reliable, or does it randomly fail?"
"""
from __future__ import annotations

import time
from typing import Callable, List, Optional

from agenttest.core.case import AgentRun


class StabilityAssertion:
    """
    Run an agent multiple times and check how consistently it behaves.

    Usage:
        stability = StabilityAssertion(agent, runs=10, pass_rate=0.9)
        stability.assert_consistent_output(
            input_text="What is 2 + 2?",
            check_fn=lambda run: "4" in run.output
        )
    """

    def __init__(
        self,
        agent: Callable,
        runs: int = 10,
        pass_rate: float = 1.0,
        delay_between_runs_ms: float = 0,
    ):
        """
        Args:
            agent: The agent callable to test.
            runs: Number of times to invoke the agent.
            pass_rate: Required fraction of passing runs (0.0–1.0).
                       1.0 means all runs must pass.
                       0.9 means at least 90% must pass.
            delay_between_runs_ms: Optional delay between runs (avoids rate limiting).
        """
        self.agent = agent
        self.runs = runs
        self.pass_rate = pass_rate
        self.delay_between_runs_ms = delay_between_runs_ms

    def assert_consistent_output(
        self,
        input_text: str,
        check_fn: Callable[[AgentRun], bool],
        **agent_kwargs,
    ):
        """
        Run the agent N times and assert a check function passes on at least pass_rate% of runs.

        Args:
            input_text: The input to send to the agent each time.
            check_fn: A function that takes an AgentRun and returns True/False.
            **agent_kwargs: Extra kwargs forwarded to the agent.

        Raises:
            AssertionError: If the pass rate is too low.

        Example:
            stability.assert_consistent_output(
                "What year was the Eiffel Tower built?",
                lambda run: "1889" in run.output
            )
        """
        results = self._run_all(input_text, **agent_kwargs)

        passed = sum(1 for run in results if run.succeeded and check_fn(run))
        total = len(results)
        actual_rate = passed / total

        if actual_rate < self.pass_rate:
            failed_outputs = [
                run.output for run in results if not (run.succeeded and check_fn(run))
            ]
            raise AssertionError(
                f"Stability check failed: {passed}/{total} runs passed "
                f"({actual_rate*100:.0f}%), required {self.pass_rate*100:.0f}%.\n"
                f"Failed outputs (first 3):\n"
                + "\n".join(f"  [{i}] {o!r}" for i, o in enumerate(failed_outputs[:3]))
            )

    def assert_output_deterministic(self, input_text: str, **agent_kwargs):
        """
        Assert the agent always returns the exact same output for the same input.
        Useful for testing deterministic (temperature=0) agents.

        Raises:
            AssertionError: If outputs differ across runs.
        """
        results = self._run_all(input_text, **agent_kwargs)
        outputs = [run.output for run in results if run.succeeded]

        if len(set(outputs)) > 1:
            raise AssertionError(
                f"Agent output is not deterministic across {self.runs} runs.\n"
                f"Distinct outputs seen ({len(set(outputs))}):\n"
                + "\n".join(f"  [{i}] {o!r}" for i, o in enumerate(set(outputs)))
            )

    def assert_tool_usage_consistent(self, input_text: str, **agent_kwargs):
        """
        Assert the agent always calls the same set of tools (regardless of order).

        Raises:
            AssertionError: If tool usage varies across runs.
        """
        results = self._run_all(input_text, **agent_kwargs)
        tool_sets = [frozenset(run.tool_names) for run in results if run.succeeded]

        if len(set(tool_sets)) > 1:
            raise AssertionError(
                f"Agent tool usage is inconsistent across {self.runs} runs.\n"
                f"Tool sets seen:\n"
                + "\n".join(f"  {set(ts)}" for ts in set(tool_sets))
            )

    def get_pass_rate(
        self,
        input_text: str,
        check_fn: Callable[[AgentRun], bool],
        **agent_kwargs,
    ) -> float:
        """Run the agent N times and return the actual pass rate (0.0–1.0)."""
        results = self._run_all(input_text, **agent_kwargs)
        passed = sum(1 for run in results if run.succeeded and check_fn(run))
        return passed / len(results)

    def _run_all(self, input_text: str, **kwargs) -> List[AgentRun]:
        results = []
        for i in range(self.runs):
            if i > 0 and self.delay_between_runs_ms > 0:
                time.sleep(self.delay_between_runs_ms / 1000)

            run = AgentRun(input=input_text, output="")
            start = time.perf_counter()
            try:
                result = self.agent(input_text, **kwargs)
                if isinstance(result, dict):
                    run.output = result.get("output", str(result))
                    run.tool_calls = result.get("tool_calls", [])
                    run.reasoning_steps = result.get("reasoning_steps", [])
                else:
                    run.output = str(result)
            except Exception as e:
                run.error = e
            run.duration_ms = (time.perf_counter() - start) * 1000
            results.append(run)

        return results

