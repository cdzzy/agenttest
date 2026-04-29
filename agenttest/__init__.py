"""
AgentTest - The Testing Framework for AI Agents

Think pytest, but for agents.
"""

from agenttest.core.case import AgentTestCase, agent_test
from agenttest.core.suite import AgentTestSuite
from agenttest.core.runner import AgentTestRunner
from agenttest.assertions.behavior import assert_tool_called, assert_tool_sequence, assert_no_tool_called
from agenttest.assertions.output import assert_output_contains, assert_output_matches, assert_output_sentiment
from agenttest.assertions.trace import assert_reasoning_step, assert_step_count
from agenttest.assertions.stability import StabilityAssertion
from agenttest.fixtures.mock_tools import MockTool, MockToolkit
from agenttest.fixtures.scenarios import ScenarioBuilder

__version__ = "0.1.0"
__all__ = [
    "AgentTestCase",
    "AgentTestSuite",
    "AgentTestRunner",
    "agent_test",
    "assert_tool_called",
    "assert_tool_sequence",
    "assert_no_tool_called",
    "assert_output_contains",
    "assert_output_matches",
    "assert_output_sentiment",
    "assert_reasoning_step",
    "assert_step_count",
    "StabilityAssertion",
    "MockTool",
    "MockToolkit",
    "ScenarioBuilder",
]

