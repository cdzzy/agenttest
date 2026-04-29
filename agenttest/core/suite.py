"""
Test suite for grouping and organizing agent tests.
"""
from __future__ import annotations

import inspect
from typing import Any, Callable, Dict, List, Optional, Type

from agenttest.core.case import AgentTestCase, agent_test


class AgentTestSuite:
    """
    Groups multiple agent test cases together.

    Usage:
        suite = AgentTestSuite("My Agent Tests")
        suite.add_case(MyTestCase)
        suite.add_function_test(my_test_fn, agent=my_agent)

        runner = AgentTestRunner()
        runner.run_suite(suite)
    """

    def __init__(self, name: str = "AgentTestSuite", description: str = ""):
        self.name = name
        self.description = description
        self._cases: List[Dict[str, Any]] = []

    def add_case(self, test_class: Type[AgentTestCase], agent: Optional[Callable] = None) -> "AgentTestSuite":
        """Add an AgentTestCase subclass to this suite."""
        self._cases.append({
            "type": "class",
            "test_class": test_class,
            "agent": agent,
        })
        return self

    def add_function(self, fn: Callable, agent: Callable, name: Optional[str] = None) -> "AgentTestSuite":
        """Add a standalone test function to this suite."""
        self._cases.append({
            "type": "function",
            "fn": fn,
            "agent": agent,
            "name": name or getattr(fn, "_test_name", fn.__name__),
        })
        return self

    def collect(self) -> List[Dict[str, Any]]:
        """Collect all test cases for execution."""
        collected = []

        for entry in self._cases:
            if entry["type"] == "class":
                cls = entry["test_class"]
                agent = entry.get("agent")

                instance = cls()
                if agent:
                    instance.agent = agent

                # Find all test methods
                for attr_name in dir(instance):
                    if not attr_name.startswith("test"):
                        continue
                    method = getattr(instance, attr_name)
                    if callable(method):
                        collected.append({
                            "name": f"{cls.__name__}.{attr_name}",
                            "fn": method,
                            "instance": instance,
                            "tags": getattr(method, "_test_tags", []),
                            "skip": getattr(method, "_skip", False),
                            "skip_reason": getattr(method, "_skip_reason", ""),
                            "repeat": getattr(method, "_repeat", 1),
                        })

            elif entry["type"] == "function":
                fn = entry["fn"]
                collected.append({
                    "name": entry["name"],
                    "fn": fn,
                    "agent": entry["agent"],
                    "tags": getattr(fn, "_test_tags", []),
                    "skip": getattr(fn, "_skip", False),
                    "skip_reason": getattr(fn, "_skip_reason", ""),
                    "repeat": getattr(fn, "_repeat", 1),
                })

        return collected

    def filter_by_tags(self, tags: List[str]) -> "AgentTestSuite":
        """Return a new suite containing only tests matching the given tags."""
        filtered = AgentTestSuite(f"{self.name} [filtered]")
        for entry in self._cases:
            # For simplicity, re-add all and let runner filter
            filtered._cases.append(entry)
        filtered._tag_filter = tags
        return filtered

    def __repr__(self) -> str:
        return f"AgentTestSuite(name={self.name!r}, cases={len(self._cases)})"

