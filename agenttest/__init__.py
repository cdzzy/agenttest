"""
AgentTest - The Testing Framework for AI Agents

Think pytest, but for agents.
"""

from agenttest.assertions.behavior import (
    assert_no_tool_called,
    assert_tool_called,
    assert_tool_sequence,
)
from agenttest.assertions.output import (
    assert_output_contains,
    assert_output_matches,
    assert_output_sentiment,
)
from agenttest.assertions.stability import StabilityAssertion
from agenttest.assertions.trace import (
    assert_reasoning_covers,
    assert_reasoning_order,
    assert_reasoning_step,
    assert_reasoning_steps,
    assert_reasoning_time,
    assert_step_count,
)
from agenttest.benchmark import BenchmarkResults, BenchmarkSuite, Scenario
from agenttest.chaos import ChaosAgent, ChaosScenario, inject_chaos, measure_resilience, run_chaos
from agenttest.core.case import AgentTestCase, agent_test, test_async
from agenttest.core.history import RunSummary, RunTrendReport, SuiteRunHistory
from agenttest.core.runner import AgentTestRunner
from agenttest.core.suite import AgentTestSuite
from agenttest.fixtures.mock_tools import MockTool, MockToolkit
from agenttest.fixtures.scenarios import ScenarioBuilder
from agenttest.flakiness import FlakinessDetector, FlakyReport
from agenttest.mcp_tools import (
    MCPServerClient,
    MCPTool,
    assert_tool_exists,
    assert_tool_response_valid,
)
from agenttest.report import build_report, report_to_html, report_to_json, save_report
from agenttest.snapshots import SnapshotStore, snapshot

__version__ = "0.4.2"
__all__ = [
    "AgentTestCase",
    "AgentTestSuite",
    "AgentTestRunner",
    "agent_test",
    "test_async",
    "SuiteRunHistory",
    "RunSummary",
    "RunTrendReport",
    "assert_tool_called",
    "assert_tool_sequence",
    "assert_no_tool_called",
    "assert_output_contains",
    "assert_output_matches",
    "assert_output_sentiment",
    "assert_reasoning_step",
    "assert_step_count",
    "assert_reasoning_steps",
    "assert_reasoning_covers",
    "assert_reasoning_order",
    "assert_reasoning_time",
    "StabilityAssertion",
    "MockTool",
    "MockToolkit",
    "ScenarioBuilder",
    "SnapshotStore",
    "snapshot",
    "BenchmarkSuite",
    "Scenario",
    "BenchmarkResults",
    "ChaosScenario",
    "ChaosAgent",
    "inject_chaos",
    "run_chaos",
    "measure_resilience",
    "FlakinessDetector",
    "FlakyReport",
    "MCPServerClient",
    "MCPTool",
    "assert_tool_exists",
    "assert_tool_response_valid",
    "save_report",
    "build_report",
    "report_to_json",
    "report_to_html",
]

