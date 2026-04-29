"""
Example: Stability testing — is your agent reliable across multiple runs?

Run with: python examples/test_stability.py
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import random
from agenttest import AgentTestSuite, AgentTestRunner
from agenttest.assertions.stability import StabilityAssertion
from agenttest.assertions.output import assert_output_contains


# ─────────────────────────────────────────────────────────────────
# Fake agents with different stability characteristics
# ─────────────────────────────────────────────────────────────────

def stable_agent(input_text: str, **kwargs) -> dict:
    """Always returns the correct answer."""
    return {
        "output": "The capital of France is Paris.",
        "tool_calls": [{"name": "knowledge_base", "input": input_text, "output": "Paris"}],
    }


def flaky_agent(input_text: str, **kwargs) -> dict:
    """Returns wrong answer 20% of the time (simulates a flaky LLM)."""
    if random.random() < 0.2:
        return {"output": "I'm not sure about that.", "tool_calls": []}
    return {"output": "The answer is Paris.", "tool_calls": [{"name": "search", "input": input_text}]}


def deterministic_agent(input_text: str, **kwargs) -> dict:
    """Always returns the exact same response (temperature=0 equivalent)."""
    return {
        "output": "42",
        "tool_calls": [{"name": "calculator", "input": "6*7", "output": "42"}],
    }


# ─────────────────────────────────────────────────────────────────
# Stability tests
# ─────────────────────────────────────────────────────────────────

def test_agent_stable_10_runs():
    """Run agent 10 times and verify it always mentions Paris."""
    stability = StabilityAssertion(stable_agent, runs=10, pass_rate=1.0)
    stability.assert_consistent_output(
        input_text="What is the capital of France?",
        check_fn=lambda run: "Paris" in run.output,
    )
    print("  ✓ Agent is perfectly stable (10/10 runs correct)")


def test_flaky_agent_acceptable_rate():
    """
    This flaky agent is allowed to fail up to 30% of the time.
    (In real life, you'd want 100%, but this demonstrates partial pass rates.)
    """
    stability = StabilityAssertion(flaky_agent, runs=20, pass_rate=0.7)
    actual_rate = stability.get_pass_rate(
        input_text="What is the capital of France?",
        check_fn=lambda run: "Paris" in run.output or "paris" in run.output.lower(),
    )
    print(f"  ✓ Flaky agent pass rate: {actual_rate*100:.0f}% (threshold: 70%)")


def test_deterministic_output():
    """Agent should produce identical outputs every time (temperature=0)."""
    stability = StabilityAssertion(deterministic_agent, runs=5)
    stability.assert_output_deterministic("What is 6 times 7?")
    print("  ✓ Agent output is deterministic across 5 runs")


def test_consistent_tool_usage():
    """Agent should always use the same tools for the same input."""
    stability = StabilityAssertion(stable_agent, runs=5)
    stability.assert_tool_usage_consistent("What is the capital of France?")
    print("  ✓ Agent tool usage is consistent across 5 runs")


if __name__ == "__main__":
    print("\n" + "="*60)
    print("  AgentTest — Stability Tests")
    print("="*60 + "\n")

    tests = [
        test_agent_stable_10_runs,
        test_flaky_agent_acceptable_rate,
        test_deterministic_output,
        test_consistent_tool_usage,
    ]

    passed = 0
    for test in tests:
        try:
            test()
            passed += 1
        except AssertionError as e:
            print(f"  ✗ {test.__name__}: {e}")

    print(f"\n{'='*60}")
    print(f"  {passed}/{len(tests)} stability tests passed")
    print(f"{'='*60}\n")
    sys.exit(len(tests) - passed)

