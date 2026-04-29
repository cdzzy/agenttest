"""
Example: YAML Test Configuration.

This example demonstrates how to write and run tests
using YAML configuration files, inspired by PraisonAI's
low-code agent configuration patterns.

Usage:
    python examples/test_yaml_config.py
"""

from agenttest.loaders.yaml_loader import (
    YAMLTestLoader,
    YAMLTestSuite,
    YAMLCase,
    create_test_from_yaml,
    validate_yaml_syntax,
)


# Sample agent for testing
def sample_agent(input_text: str) -> dict:
    """A simple mock agent for demonstration."""
    responses = {
        "hello": "Hello! How can I help you today?",
        "products": "We have several products: A, B, and C. All are available now.",
        "pricing": "I understand you're asking about pricing. Let me connect you with our sales team.",
        "help": "I can help you with products, orders, and technical support.",
    }

    input_lower = input_text.lower()
    response = "I'm here to help! What would you like assistance with?"

    for key, value in responses.items():
        if key in input_lower:
            response = value
            break

    return {
        "output": response,
        "tool_calls": [],
        "reasoning_steps": [],
    }


def example_yaml_format():
    """Show the YAML format for tests."""
    print("=" * 60)
    print("Example: YAML Test Format")
    print("=" * 60)

    yaml_content = """
# Test suite for customer support agent
name: "Customer Support Tests"
description: "Test suite for the customer support chatbot"
agent: "./my_agent.py"  # Path to agent file

model:
  provider: openai
  model: gpt-4o-mini
  temperature: 0.7

cases:
  - name: "Basic greeting"
    input: "Hello!"
    expected_output_contains:
      - "Hello"
      - "help"
    tags:
      - "smoke"
      - "greeting"

  - name: "Product inquiry"
    input: "What products do you have?"
    expected_output_contains:
      - "products"
      - "A"
    expected_output_not_contains:
      - "pricing"
      - "cost"
    tags:
      - "products"

  - name: "Pricing escalation"
    input: "What's the pricing?"
    expected_output_not_contains:
      - "$"
      - "100"
      - "200"
    tags:
      - "constraints"
      - "escalation"
"""

    print("\nSample YAML Test Configuration:")
    print("-" * 40)
    print(yaml_content)


def example_yaml_validation():
    """Validate YAML syntax."""
    print("\n" + "=" * 60)
    print("Example: YAML Validation")
    print("=" * 60)

    print("\nValidating YAML syntax:")
    print("  validate_yaml_syntax function is available for use")
    print("  Usage: is_valid, error = validate_yaml_syntax('tests.yaml')")


def example_yaml_loading():
    """Load and parse YAML test configuration."""
    print("\n" + "=" * 60)
    print("Example: Loading YAML Tests")
    print("=" * 60)

    # Create a temporary YAML string for demonstration
    yaml_content = """
name: "Demo Test Suite"
description: "Demonstration of YAML test loading"

cases:
  - name: "Basic greeting"
    input: "Hello"
    expected_output_contains:
      - "Hello"
    tags:
      - "smoke"

  - name: "Product question"
    input: "Tell me about products"
    expected_output_contains:
      - "products"
    expected_output_not_contains:
      - "error"
    tags:
      - "products"

  - name: "Skipped test"
    input: "This won't run"
    skip: true
    tags:
      - "wip"
"""

    import tempfile
    import os

    # Write to temp file
    with tempfile.NamedTemporaryFile(
        mode="w",
        suffix=".yaml",
        delete=False,
        encoding="utf-8"
    ) as f:
        f.write(yaml_content)
        temp_path = f.name

    try:
        # Load using the loader class
        loader = YAMLTestLoader(temp_path)
        suite = loader.load()

        print(f"\nLoaded Test Suite: {suite.name}")
        print(f"Description: {suite.description}")
        print(f"Total Cases: {len(suite.cases)}")

        print("\nTest Cases:")
        for case in suite.cases:
            skip_label = " [SKIPPED]" if case.skip else ""
            tags_label = f" [{', '.join(case.tags)}]" if case.tags else ""
            print(f"  - {case.name}{skip_label}{tags_label}")
            print(f"    Input: {case.input[:50]}...")

    finally:
        os.unlink(temp_path)


def example_cli_usage():
    """Show CLI usage examples."""
    print("\n" + "=" * 60)
    print("CLI Usage Examples")
    print("=" * 60)

    print("""
# Run tests from YAML file
$ agenttest run tests.yaml

# Run only smoke tests
$ agenttest run tests.yaml --tags smoke

# Run with verbose output
$ agenttest run tests.yaml -v

# Validate YAML syntax without running
$ agenttest validate tests.yaml

# Generate test template
$ agenttest init --template customer-support

# Convert existing test to YAML
$ agenttest convert test_basic_agent.py --output tests.yaml
""")


def example_create_suite():
    """Show how to create a test suite from YAML."""
    print("\n" + "=" * 60)
    print("Example: Create Suite from YAML")
    print("=" * 60)

    print("""
from agenttest.loaders.yaml_loader import create_test_from_yaml
from agenttest import AgentTestRunner

# Load and create suite in one step
suite = create_test_from_yaml(
    "tests/customer_support.yaml",
    agent=my_agent,
    tags=["smoke"],  # Optional: filter by tags
)

# Run the suite
runner = AgentTestRunner()
result = runner.run_suite(suite)

# Check results
if result.passed:
    print("All tests passed!")
else:
    print(f"Failed: {result.failed_count} / {result.total}")
""")


if __name__ == "__main__":
    # Run examples
    example_yaml_format()
    example_yaml_validation()
    example_yaml_loading()
    example_cli_usage()
    example_create_suite()

    print("\n" + "=" * 60)
    print("✅ YAML Test Configuration examples completed!")
    print("=" * 60)

