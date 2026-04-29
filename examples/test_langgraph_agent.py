"""
Example: Testing a LangGraph agent with agenttest.

Shows how to wrap a LangGraph StateGraph agent for testing.
Run with: python examples/test_langgraph_agent.py
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agenttest import (
    AgentTestCase,
    AgentTestSuite,
    AgentTestRunner,
    MockToolkit,
)
from agenttest.assertions.behavior import (
    assert_tool_called,
    assert_tool_sequence,
    assert_tool_called_before,
)
from agenttest.assertions.output import assert_output_contains, assert_output_json


# ─────────────────────────────────────────────────────────────────
# Simulated LangGraph agent wrapper
# In real usage, replace this with your actual LangGraph graph.invoke()
# ─────────────────────────────────────────────────────────────────

class FakeLangGraphAgent:
    """Demonstrates how to wrap a LangGraph agent for agenttest."""

    def __init__(self, tools: dict = None):
        self.tools = tools or {}

    def __call__(self, input_text: str, **kwargs) -> dict:
        """
        Wrap your LangGraph graph like this:

            result = self.graph.invoke({"messages": [HumanMessage(content=input_text)]})
            return {
                "output": result["messages"][-1].content,
                "tool_calls": self._extract_tool_calls(result),
                "reasoning_steps": self._extract_thoughts(result),
            }
        """
        # Simulate tool execution using injected mock tools
        tool_calls = []
        if "search" in input_text.lower() or "find" in input_text.lower():
            tool_name = "web_search"
            tool_result = self.tools.get(tool_name, lambda x: "search results")(input_text)
            tool_calls.append({"name": tool_name, "input": input_text, "output": tool_result})

            # Then summarize
            summary_tool = self.tools.get("summarize", lambda x: "summary")
            summary = summary_tool(tool_result)
            tool_calls.append({"name": "summarize", "input": tool_result, "output": summary})

            return {
                "output": f"Here's what I found: {summary}",
                "tool_calls": tool_calls,
                "reasoning_steps": [
                    "I need to search for this information.",
                    "Using web_search tool.",
                    "Summarizing the results.",
                ],
            }

        return {
            "output": f"I can answer that directly: {input_text}",
            "tool_calls": [],
            "reasoning_steps": ["This can be answered from my training data."],
        }


# ─────────────────────────────────────────────────────────────────
# Tests
# ─────────────────────────────────────────────────────────────────

class LangGraphAgentTests(AgentTestCase):

    def setUp(self):
        """Set up mock tools for each test."""
        self.toolkit = MockToolkit()
        self.toolkit.add("web_search", returns="AI news: GPT-5 released...")
        self.toolkit.add("summarize", returns="Summary: Major AI developments in 2026.")
        self.agent = FakeLangGraphAgent(tools=self.toolkit.as_dict())

    def test_search_flow(self):
        """Agent should search then summarize for research tasks."""
        run = self.invoke("Search for AI news")

        self.assert_no_error(run)
        self.assert_tool_called(run, "web_search")
        self.assert_tool_called(run, "summarize")
        self.assert_tool_sequence(run, ["web_search", "summarize"])
        self.assert_output_contains(run, "Summary")

    def test_direct_answer_no_search(self):
        """Agent should NOT call search for simple direct questions."""
        run = self.invoke("What is 2+2?")

        self.assert_no_error(run)
        self.assert_no_tool_called(run, "web_search")

    def test_mock_tools_were_called(self):
        """Verify that our mock tools were actually invoked."""
        run = self.invoke("Search for recent Python news")

        # Verify through MockTool's own assertions
        self.toolkit["web_search"].assert_called()
        self.toolkit["summarize"].assert_called()

        # Check call count
        assert self.toolkit["web_search"].call_count == 1, \
            f"Expected 1 search call, got {self.toolkit['web_search'].call_count}"

    def test_search_precedes_summarize(self):
        """Search must happen before summarization (logical order)."""
        run = self.invoke("Find and summarize the latest AI papers")
        assert_tool_called_before(run, "web_search", "summarize")

    def test_reasoning_captured(self):
        """Reasoning steps should be present for complex tasks."""
        run = self.invoke("Search for quantum computing advances")
        self.assert_step_count(run, min_steps=1)


if __name__ == "__main__":
    suite = AgentTestSuite("LangGraph Agent Tests")
    suite.add_case(LangGraphAgentTests)

    runner = AgentTestRunner(verbose=True)
    results = runner.run_suite(suite)
    sys.exit(sum(1 for r in results if not r.passed))

