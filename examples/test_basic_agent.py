"""
Example: Testing a simple LangChain-style agent with agenttest.

This example shows the most common patterns you'll use day-to-day.
Run with: python examples/test_basic_agent.py
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agenttest import (
    AgentTestCase,
    AgentTestSuite,
    AgentTestRunner,
    agent_test,
    MockTool,
    MockToolkit,
)
from agenttest.assertions.behavior import assert_tool_called, assert_tool_sequence, assert_max_tool_calls
from agenttest.assertions.output import assert_output_contains, assert_output_length
from agenttest.assertions.trace import assert_step_count


# ─────────────────────────────────────────────────────────────────
# Fake agent for demo purposes
# In your real tests, replace this with your actual agent
# ─────────────────────────────────────────────────────────────────

def fake_search_agent(input_text: str, **kwargs) -> dict:
    """
    Simulates an agent that always searches and returns a structured response.
    Replace this with your real agent in actual tests.
    """
    return {
        "output": f"Based on my research, here is the answer about '{input_text}': "
                  f"The answer is 42. Paris is the capital of France. The year was 1889.",
        "tool_calls": [
            {"name": "web_search", "input": input_text, "output": "search results..."},
            {"name": "summarize", "input": "...", "output": "summary..."},
        ],
        "reasoning_steps": [
            f"I need to search for information about: {input_text}",
            "I will use the web_search tool to find relevant information.",
            "Now I will summarize the findings.",
        ],
        "usage": {"total_tokens": 350},
    }


def fake_calculator_agent(input_text: str, **kwargs) -> dict:
    """Simulates an agent that uses a calculator."""
    return {
        "output": "The result is 42.",
        "tool_calls": [
            {"name": "calculator", "input": "6 * 7", "output": "42"},
        ],
        "reasoning_steps": [
            "I need to calculate this expression.",
            "I will use the calculator tool.",
        ],
    }


def fake_simple_agent(input_text: str, **kwargs) -> str:
    """Simulates the simplest possible agent — just returns a string."""
    return f"Hello! You said: {input_text}"


# ─────────────────────────────────────────────────────────────────
# Test using AgentTestCase (class-based style, like unittest)
# ─────────────────────────────────────────────────────────────────

class SearchAgentTests(AgentTestCase):
    """Tests for an agent that uses web search."""

    agent = fake_search_agent  # Set the agent to test

    def test_uses_search_tool(self):
        """Agent should always call web_search for knowledge questions."""
        run = self.invoke("What is the capital of France?")
        self.assert_no_error(run)
        self.assert_tool_called(run, "web_search")

    def test_search_then_summarize(self):
        """Agent should search before summarizing."""
        run = self.invoke("Summarize the latest AI news")
        self.assert_tool_sequence(run, ["web_search", "summarize"])

    def test_output_is_meaningful(self):
        """Output should be non-trivial."""
        run = self.invoke("Explain quantum computing")
        self.assert_output_length(run, min_chars=20)

    def test_does_not_hallucinate_tool(self):
        """Agent should not call a dangerous tool it has no reason to use."""
        run = self.invoke("What time is it?")
        self.assert_no_tool_called(run, "send_email")  # shouldn't email anyone

    def test_response_is_fast(self):
        """Agent should respond within a reasonable time."""
        run = self.invoke("Quick question: what is 2+2?")
        self.assert_response_time(run, max_ms=5000)

    def test_reasoning_mentions_search(self):
        """Agent's reasoning should mention searching."""
        run = self.invoke("Who invented the telephone?")
        # Only checked if reasoning steps are captured
        if run.reasoning_steps:
            from agenttest.assertions.trace import assert_reasoning_step
            assert_reasoning_step(run, "search")


# ─────────────────────────────────────────────────────────────────
# Test using @agent_test decorator (functional style, like pytest)
# ─────────────────────────────────────────────────────────────────

@agent_test(tags=["smoke"])
def test_basic_response(agent):
    run = agent("Hello, how are you?")
    assert run.output, "Output should not be empty"
    assert len(run.output) > 5, f"Output too short: {run.output!r}"


@agent_test(tags=["smoke", "tools"])
def test_tool_called(agent):
    run = agent("Search for information about Python")
    assert_tool_called(run, "web_search")


@agent_test(tags=["tools"])
def test_tool_sequence_order(agent):
    run = agent("Find and summarize news about AI")
    assert_tool_sequence(run, ["web_search", "summarize"])


@agent_test(tags=["safety"])
def test_tool_count_reasonable(agent):
    """Agent shouldn't call 20 tools for a simple question."""
    run = agent("What is 1 + 1?")
    assert_max_tool_calls(run, max_calls=3)


@agent_test(repeat=3, tags=["stability"])
def test_consistent_for_factual_question(agent):
    """Run 3 times — agent should consistently mention the answer."""
    run = agent("What is the capital of France?")
    assert_output_contains(run, "Paris")


# ─────────────────────────────────────────────────────────────────
# Assemble and run the suite
# ─────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    suite = AgentTestSuite("Basic Agent Tests", description="Smoke + behavior tests")

    # Add class-based tests
    suite.add_case(SearchAgentTests)

    # Add function-based tests (need to pass the agent explicitly)
    suite.add_function(test_basic_response,   agent=fake_simple_agent)
    suite.add_function(test_tool_called,      agent=fake_search_agent)
    suite.add_function(test_tool_sequence_order, agent=fake_search_agent)
    suite.add_function(test_tool_count_reasonable, agent=fake_calculator_agent)
    suite.add_function(test_consistent_for_factual_question, agent=fake_search_agent)

    runner = AgentTestRunner(verbose=True)
    results = runner.run_suite(suite)

    # Exit with non-zero code if any tests failed (useful for CI)
    failed = sum(1 for r in results if not r.passed)
    sys.exit(failed)

