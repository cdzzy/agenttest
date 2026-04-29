"""
YAML Test Loader - Load and run tests from YAML configuration files.

This module provides a YAML-based test configuration format,
inspired by PraisonAI's low-code/no-code agent configuration patterns.

Usage:
    # CLI
    agenttest run tests.yaml

    # Python API
    loader = YAMLTestLoader("tests.yaml")
    suite = loader.load()
    runner.run_suite(suite)
"""

from __future__ import annotations

import yaml
from pathlib import Path
from typing import Any, Dict, List, Optional, Union, Callable
from dataclasses import dataclass, field


@dataclass
class YAMLCase:
    """A single test case from YAML configuration."""
    name: str
    input: str
    expected_tools: Optional[List[str]] = None
    expected_output_contains: Optional[List[str]] = None
    expected_output_not_contains: Optional[List[str]] = None
    expected_output_matches: Optional[str] = None
    max_duration_ms: Optional[int] = None
    repeats: int = 1
    tags: List[str] = field(default_factory=list)
    skip: bool = False
    mock_tools: Optional[Dict[str, Any]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class YAMLTestSuite:
    """A complete test suite from YAML configuration."""
    name: str
    description: Optional[str] = None
    agent: Optional[str] = None  # Agent reference or path
    model: Optional[Dict[str, Any]] = None
    cases: List[YAMLCase] = field(default_factory=list)
    global_tags: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


class YAMLTestLoader:
    """
    Load and parse YAML test configuration files.

    Example YAML format:

    ```yaml
    name: "My Agent Tests"
    description: "Test suite for the customer support agent"
    agent: "./my_agent.py"

    model:
      provider: openai
      model: gpt-4o-mini
      temperature: 0.7

    cases:
      - name: "Basic greeting"
        input: "Hello!"
        expected_output_contains:
          - "Hello"
          - "How can I help"

      - name: "Product inquiry"
        input: "What products do you have?"
        expected_tools:
          - "search_products"
        expected_output_contains:
          - "products"
        tags:
          - "smoke"
          - "products"

      - name: "Sensitive topic blocked"
        input: "What's your pricing?"
        expected_output_not_contains:
          - "$"
          - "price"
        expected_tools:
          - "escalate"
        tags:
          - "constraints"
    ```

    Multi-agent format:

    ```yaml
    name: "Multi-Agent Research Team"
    description: "Test multi-agent research workflow"

    agents:
      researcher:
        type: langgraph
        config: "./agents/researcher.py"
      writer:
        type: autogen
        config: "./agents/writer.py"

    cases:
      - name: "Research and write"
        input: "Research AI trends and write a summary"
        agents:
          - researcher
          - writer
        expected_tools:
          - "web_search"
          - "write_file"
        tags:
          - "integration"
    ```
    """

    def __init__(self, path: Union[str, Path]):
        """
        Initialize loader with YAML file path.

        Args:
            path: Path to YAML test file
        """
        self.path = Path(path)
        if not self.path.exists():
            raise FileNotFoundError(f"Test file not found: {path}")

    def load(self) -> YAMLTestSuite:
        """
        Load and parse the YAML file.

        Returns:
            YAMLTestSuite with parsed configuration
        """
        with open(self.path, encoding="utf-8") as f:
            data = yaml.safe_load(f)

        if not data:
            raise ValueError("Empty YAML file")

        return self._parse_suite(data)

    def _parse_suite(self, data: dict) -> YAMLTestSuite:
        """Parse top-level suite configuration."""
        suite = YAMLTestSuite(
            name=data.get("name", self.path.stem),
            description=data.get("description"),
            agent=data.get("agent"),
            model=data.get("model"),
            global_tags=data.get("tags", []),
            metadata=data.get("metadata", {}),
        )

        # Parse test cases
        cases_data = data.get("cases", [])
        for case_data in cases_data:
            case = self._parse_case(case_data)
            suite.cases.append(case)

        # Parse multi-agent cases if present
        agents_data = data.get("agents", {})
        if agents_data:
            suite.metadata["agents"] = agents_data

            multi_cases = data.get("multi_cases", [])
            for case_data in multi_cases:
                case = self._parse_case(case_data)
                suite.cases.append(case)

        return suite

    def _parse_case(self, data: dict) -> YAMLCase:
        """Parse a single test case."""
        return YAMLCase(
            name=data.get("name", "Unnamed test"),
            input=data.get("input", ""),
            expected_tools=data.get("expected_tools"),
            expected_output_contains=data.get("expected_output_contains"),
            expected_output_not_contains=data.get("expected_output_not_contains"),
            expected_output_matches=data.get("expected_output_matches"),
            max_duration_ms=data.get("max_duration_ms"),
            repeats=data.get("repeats", 1),
            tags=data.get("tags", []),
            skip=data.get("skip", False),
            mock_tools=data.get("mock_tools"),
            metadata=data.get("metadata", {}),
        )


def create_test_from_yaml(
    yaml_path: Union[str, Path],
    agent: Callable,
    tags: Optional[List[str]] = None,
) -> "AgentTestSuite":
    """
    Create an AgentTestSuite from a YAML file.

    This is a convenience function that:
    1. Loads the YAML configuration
    2. Converts cases to AgentTestCase objects
    3. Returns a runnable test suite

    Args:
        yaml_path: Path to YAML test file
        agent: The agent callable to test
        tags: Optional tags to filter cases

    Returns:
        AgentTestSuite ready to run

    Example:
        suite = create_test_from_yaml("tests/customer_support.yaml", my_agent)
        runner = AgentTestRunner()
        result = runner.run_suite(suite)
    """
    from agenttest import AgentTestSuite, AgentTestCase

    loader = YAMLTestLoader(yaml_path)
    yaml_suite = loader.load()

    # Create test suite
    suite = AgentTestSuite(yaml_suite.name)

    # Filter by tags if specified
    cases = yaml_suite.cases
    if tags:
        cases = [
            c for c in cases
            if any(t in c.tags for t in tags) or not c.tags
        ]

    # Create test case for each YAML case
    for yaml_case in cases:
        if yaml_case.skip:
            continue

        # Create test method dynamically
        def make_test(case: YAMLCase):
            def test_method(self):
                run = self.invoke(case.input)

                # Check tools
                if case.expected_tools:
                    for tool in case.expected_tools:
                        self.assert_tool_called(run, tool)

                # Check output contains
                if case.expected_output_contains:
                    for expected in case.expected_output_contains:
                        self.assert_output_contains(run, expected)

                # Check output not contains
                if case.expected_output_not_contains:
                    for not_expected in case.expected_output_not_contains:
                        self.assert_output_not_contains(run, not_expected)

                # Check output matches regex
                if case.expected_output_matches:
                    self.assert_output_matches(run, case.expected_output_matches)

                # Check duration
                if case.max_duration_ms:
                    self.assert_duration(run, max_ms=case.max_duration_ms)

            test_method.__name__ = f"test_{yaml_case.name.lower().replace(' ', '_')}"
            test_method.__doc__ = yaml_case.name

            return test_method

        # Create test case class
        TestClass = type(
            f"Test{yaml_case.name.replace(' ', '')}",
            (AgentTestCase,),
            {
                "agent": agent,
                "test_" + yaml_case.name.lower().replace(" ", "_"): make_test(yaml_case),
            },
        )

        suite.add_case(TestClass)

    return suite


def yaml_to_test_cases(yaml_path: Union[str, Path]) -> List[YAMLCase]:
    """
    Load test cases from YAML without creating a suite.

    Useful for custom test runner implementations.

    Args:
        yaml_path: Path to YAML test file

    Returns:
        List of YAMLCase objects
    """
    loader = YAMLTestLoader(yaml_path)
    suite = loader.load()
    return suite.cases


def validate_yaml_syntax(path: Union[str, Path]) -> tuple[bool, Optional[str]]:
    """
    Validate YAML file syntax.

    Args:
        path: Path to YAML file

    Returns:
        Tuple of (is_valid, error_message)
    """
    try:
        with open(path, encoding="utf-8") as f:
            yaml.safe_load(f)
        return True, None
    except yaml.YAMLError as e:
        return False, str(e)

