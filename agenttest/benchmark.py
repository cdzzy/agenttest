"""
Agent benchmarking and performance comparison (Issue #5).

Compare how different models (or agents) perform on a common scenario suite,
producing a comparison report for model selection and regression tracking.

Usage::

    from agenttest.benchmark import BenchmarkSuite, Scenario

    suite = BenchmarkSuite(name="support-v1", scenarios=[
        Scenario("refund", "I want a refund", evaluate=lambda r: "refund" in r.output.lower()),
    ])

    results = suite.run(
        agents={"gpt-4o": agent_factory("gpt-4o"), "claude": agent_factory("claude")},
        runs_per_scenario=3,
    )

    print(suite.generate_report(results, format="markdown"))
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

from agenttest.core.case import AgentRun
from agenttest.core.runner import _wrap_agent_as_invoke


@dataclass
class Scenario:
    """A single benchmark scenario."""
    name: str
    input: str
    evaluate: Callable[[AgentRun], bool]
    description: str = ""

    def score(self, run: AgentRun) -> bool:
        try:
            return bool(self.evaluate(run))
        except Exception:
            return False


@dataclass
class ScenarioResult:
    scenario: str
    passed: int
    total: int

    @property
    def pass_rate(self) -> float:
        return self.passed / self.total if self.total else 0.0


@dataclass
class BenchmarkResults:
    name: str
    scenarios: List[str]
    rates: Dict[str, Dict[str, float]] = field(default_factory=dict)
    averages: Dict[str, float] = field(default_factory=dict)

    def average_for(self, agent: str) -> float:
        rates = self.rates.get(agent, {})
        if not rates:
            return 0.0
        return sum(rates.values()) / len(rates)


class BenchmarkSuite:
    """
    Runs a set of scenarios against multiple agents and reports pass rates.
    """

    def __init__(self, name: str, scenarios: List[Scenario]) -> None:
        self.name = name
        self.scenarios = scenarios

    def run(
        self,
        agents: Dict[str, Callable],
        runs_per_scenario: int = 1,
    ) -> BenchmarkResults:
        """
        Run every scenario against every agent.

        Args:
            agents: Map of agent name → agent callable.
            runs_per_scenario: Number of repeated runs (for statistical stability).

        Returns:
            BenchmarkResults with per-scenario pass rates.
        """
        results = BenchmarkResults(name=self.name, scenarios=[s.name for s in self.scenarios])

        for agent_name, agent in agents.items():
            invoke = _wrap_agent_as_invoke(agent)
            rates: Dict[str, float] = {}
            for scenario in self.scenarios:
                passed = 0
                for _ in range(runs_per_scenario):
                    run = invoke(scenario.input)
                    if scenario.score(run):
                        passed += 1
                rates[scenario.name] = passed / runs_per_scenario if runs_per_scenario else 0.0
            results.rates[agent_name] = rates
            results.averages[agent_name] = sum(rates.values()) / len(rates) if rates else 0.0

        return results

    def generate_report(self, results: BenchmarkResults, format: str = "markdown") -> str:
        """
        Render a comparison report.

        Args:
            results: BenchmarkResults from :meth:`run`.
            format: "markdown" or "text".

        Returns:
            The rendered report string.
        """
        agents = list(results.rates.keys())
        if format == "markdown":
            return self._markdown_report(results, agents)
        return self._text_report(results, agents)

    def _markdown_report(self, results: BenchmarkResults, agents: List[str]) -> str:
        header = "| Scenario | " + " | ".join(agents) + " |"
        sep = "|" + "---|" * (len(agents) + 1)
        lines = [f"# Benchmark: {results.name}", "", header, sep]

        for scenario in results.scenarios:
            cells = [scenario]
            for agent in agents:
                rate = results.rates.get(agent, {}).get(scenario, 0.0)
                cells.append(f"{rate*100:.0f}%")
            lines.append("| " + " | ".join(cells) + " |")

        avg_cells = ["**Average**"]
        for agent in agents:
            avg_cells.append(f"**{results.averages.get(agent, 0.0)*100:.1f}%**")
        lines.append("| " + " | ".join(avg_cells) + " |")
        return "\n".join(lines)

    def _text_report(self, results: BenchmarkResults, agents: List[str]) -> str:
        width = max(len(s) for s in ["Scenario"] + agents + results.scenarios) + 2
        lines = [f"Benchmark: {results.name}", ""]
        lines.append("".ljust(width) + "".join(a.ljust(width) for a in agents))
        for scenario in results.scenarios:
            row = scenario.ljust(width)
            for agent in agents:
                rate = results.rates.get(agent, {}).get(scenario, 0.0)
                row += f"{rate*100:.0f}%".ljust(width)
            lines.append(row)
        avg = "Average".ljust(width)
        for agent in agents:
            avg += f"{results.averages.get(agent, 0.0)*100:.1f}%".ljust(width)
        lines.append(avg)
        return "\n".join(lines)
