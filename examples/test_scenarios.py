"""
Example: Scenario-driven testing — test multiple inputs systematically.

Useful for regression testing with a dataset of known inputs/outputs.
Run with: python examples/test_scenarios.py
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agenttest import AgentTestCase, AgentTestSuite, AgentTestRunner, ScenarioBuilder
from agenttest.assertions.behavior import assert_tool_called
from agenttest.assertions.output import assert_output_contains


def smart_agent(input_text: str) -> dict:
    """Smart demo agent that handles different input types."""
    text = input_text.lower()

    if any(w in text for w in ["search", "find", "latest", "news", "who", "what"]):
        return {
            "output": f"Based on my search: The answer to '{input_text}' is well-documented. "
                      f"Paris is the capital of France. Einstein discovered relativity.",
            "tool_calls": [{"name": "web_search", "input": input_text, "output": "results..."}],
            "reasoning_steps": ["I need to search for this information.", "Found relevant results."],
        }
    elif any(w in text for w in ["calculate", "+", "-", "*", "/", "math"]):
        return {
            "output": "The result is 42.",
            "tool_calls": [{"name": "calculator", "input": input_text, "output": "42"}],
            "reasoning_steps": ["This is a math problem.", "Using calculator."],
        }
    else:
        return {
            "output": f"I understand you're asking about: {input_text}. Here's my response.",
            "tool_calls": [],
            "reasoning_steps": ["I can answer this from knowledge."],
        }


class ScenarioTests(AgentTestCase):
    agent = smart_agent

    def test_all_scenarios(self):
        """Run through a batch of scenarios programmatically."""
        scenarios = (
            ScenarioBuilder()
            .add("Capital city query")
                .input("What is the capital of France?")
                .expect_tools("web_search")
                .expect_output_contains("Paris")
                .done()
            .add("Math calculation")
                .input("Calculate 6 * 7")
                .expect_tools("calculator")
                .expect_output_contains("42")
                .done()
            .add("General greeting")
                .input("Hello there!")
                .expect_output_contains("I understand")
                .done()
            .add("Safety check - no dangerous output")
                .input("Search for Python tutorials")
                .expect_output_not_contains("I cannot help")
                .done()
            .build()
        )

        print(f"\n    Running {len(scenarios)} scenarios...")
        passed = 0
        failed_scenarios = []

        for scenario in scenarios:
            run = self.invoke(scenario.input)

            try:
                # Check expected tools
                for tool in scenario.expected_tools:
                    assert_tool_called(run, tool)

                # Check expected output contents
                for text in scenario.expected_output_contains:
                    assert_output_contains(run, text)

                # Check output doesn't contain bad strings
                for bad_text in scenario.expected_output_not_contains:
                    from agenttest.assertions.output import assert_output_not_contains
                    assert_output_not_contains(run, bad_text)

                # Run custom check if provided
                if scenario.check_fn:
                    assert scenario.check_fn(run), f"Custom check failed for scenario: {scenario.name}"

                passed += 1
                print(f"    ✅  {scenario.name}")

            except AssertionError as e:
                failed_scenarios.append((scenario.name, str(e)))
                print(f"    ❌  {scenario.name}: {e}")

        if failed_scenarios:
            details = "\n".join(f"  - {name}: {msg}" for name, msg in failed_scenarios)
            raise AssertionError(
                f"{len(failed_scenarios)}/{len(scenarios)} scenarios failed:\n{details}"
            )


if __name__ == "__main__":
    suite = AgentTestSuite("Scenario Tests")
    suite.add_case(ScenarioTests)

    runner = AgentTestRunner(verbose=True)
    results = runner.run_suite(suite)
    sys.exit(sum(1 for r in results if not r.passed))

