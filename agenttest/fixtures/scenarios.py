"""
Scenario builder — fluent API for constructing test inputs and expected behaviors.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional


@dataclass
class Scenario:
    """A single test scenario with input, expected behavior, and metadata."""
    name: str
    input: str
    expected_tools: List[str] = field(default_factory=list)
    expected_tool_sequence: List[str] = field(default_factory=list)
    expected_output_contains: List[str] = field(default_factory=list)
    expected_output_not_contains: List[str] = field(default_factory=list)
    tags: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    check_fn: Optional[Callable] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "input": self.input,
            "expected_tools": self.expected_tools,
            "expected_tool_sequence": self.expected_tool_sequence,
            "expected_output_contains": self.expected_output_contains,
            "expected_output_not_contains": self.expected_output_not_contains,
            "tags": self.tags,
            "metadata": self.metadata,
        }


class ScenarioBuilder:
    """
    Fluent builder for creating test scenarios.

    Usage:
        scenarios = (
            ScenarioBuilder()
            .add("Basic greeting")
                .input("Hello!")
                .expect_output_contains("Hello", "hi", "hey")
                .done()
            .add("Web search task")
                .input("What's the latest news about AI?")
                .expect_tools("web_search")
                .expect_tool_sequence(["web_search", "summarize"])
                .done()
            .build()
        )
    """

    def __init__(self):
        self._scenarios: List[Scenario] = []
        self._current: Optional[_ScenarioEntry] = None

    def add(self, name: str) -> "_ScenarioEntry":
        """Start building a new scenario with the given name."""
        entry = _ScenarioEntry(self, name)
        self._current = entry
        return entry

    def _commit(self, scenario: Scenario):
        self._scenarios.append(scenario)

    def build(self) -> List[Scenario]:
        """Finalize and return all scenarios."""
        return self._scenarios.copy()

    @staticmethod
    def from_jsonl(path: str) -> List[Scenario]:
        """Load scenarios from a JSONL file."""
        import json
        scenarios = []
        with open(path, "r", encoding="utf-8") as f:
            for i, line in enumerate(f):
                line = line.strip()
                if not line:
                    continue
                data = json.loads(line)
                scenarios.append(Scenario(
                    name=data.get("name", f"scenario_{i}"),
                    input=data["input"],
                    expected_tools=data.get("expected_tools", []),
                    expected_tool_sequence=data.get("expected_tool_sequence", []),
                    expected_output_contains=data.get("expected_output_contains", []),
                    expected_output_not_contains=data.get("expected_output_not_contains", []),
                    tags=data.get("tags", []),
                    metadata=data.get("metadata", {}),
                ))
        return scenarios

    @staticmethod
    def from_csv(path: str) -> List[Scenario]:
        """Load scenarios from a CSV file (columns: name, input, expected_output_contains)."""
        import csv
        scenarios = []
        with open(path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for i, row in enumerate(reader):
                scenarios.append(Scenario(
                    name=row.get("name", f"row_{i}"),
                    input=row.get("input", ""),
                    expected_output_contains=[
                        x.strip() for x in row.get("expected_output_contains", "").split("|") if x.strip()
                    ],
                ))
        return scenarios


class _ScenarioEntry:
    """Internal fluent builder for a single scenario."""

    def __init__(self, parent: ScenarioBuilder, name: str):
        self._parent = parent
        self._scenario = Scenario(name=name, input="")

    def input(self, text: str) -> "_ScenarioEntry":
        self._scenario.input = text
        return self

    def expect_tools(self, *tool_names: str) -> "_ScenarioEntry":
        self._scenario.expected_tools.extend(tool_names)
        return self

    def expect_tool_sequence(self, sequence: List[str]) -> "_ScenarioEntry":
        self._scenario.expected_tool_sequence = sequence
        return self

    def expect_output_contains(self, *texts: str) -> "_ScenarioEntry":
        self._scenario.expected_output_contains.extend(texts)
        return self

    def expect_output_not_contains(self, *texts: str) -> "_ScenarioEntry":
        self._scenario.expected_output_not_contains.extend(texts)
        return self

    def tag(self, *tags: str) -> "_ScenarioEntry":
        self._scenario.tags.extend(tags)
        return self

    def with_check(self, fn: Callable) -> "_ScenarioEntry":
        self._scenario.check_fn = fn
        return self

    def done(self) -> ScenarioBuilder:
        """Commit this scenario and return the parent builder."""
        self._parent._commit(self._scenario)
        return self._parent

