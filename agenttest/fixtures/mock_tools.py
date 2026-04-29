"""
Mock tools and toolkits for testing agents without real side effects.

Use these to isolate your agent from external dependencies during testing.
"""
from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional
from unittest.mock import MagicMock


class MockTool:
    """
    A mock tool that records calls and returns a configurable response.

    Usage:
        search = MockTool("web_search", returns="Paris is the capital of France.")
        # ... inject into your agent ...
        assert search.call_count == 1
        assert search.last_call["query"] == "capital of France"
    """

    def __init__(
        self,
        name: str,
        returns: Any = "",
        side_effect: Optional[Callable] = None,
        description: str = "",
    ):
        self.name = name
        self.description = description or f"Mock tool: {name}"
        self._returns = returns
        self._side_effect = side_effect
        self._calls: List[Dict[str, Any]] = []

    def __call__(self, *args, **kwargs) -> Any:
        call_record = {"args": args, "kwargs": kwargs}
        if args:
            call_record["input"] = args[0] if len(args) == 1 else args
        call_record.update(kwargs)
        self._calls.append(call_record)

        if self._side_effect:
            return self._side_effect(*args, **kwargs)
        if callable(self._returns):
            return self._returns(*args, **kwargs)
        return self._returns

    @property
    def call_count(self) -> int:
        return len(self._calls)

    @property
    def calls(self) -> List[Dict[str, Any]]:
        return self._calls.copy()

    @property
    def last_call(self) -> Optional[Dict[str, Any]]:
        return self._calls[-1] if self._calls else None

    @property
    def was_called(self) -> bool:
        return len(self._calls) > 0

    def reset(self):
        self._calls.clear()

    def set_returns(self, value: Any):
        self._returns = value

    def assert_called(self):
        if not self.was_called:
            raise AssertionError(f"MockTool '{self.name}' was never called.")

    def assert_called_once(self):
        if self.call_count != 1:
            raise AssertionError(
                f"MockTool '{self.name}' expected 1 call, got {self.call_count}."
            )

    def assert_called_with(self, **expected_kwargs):
        if not self.was_called:
            raise AssertionError(f"MockTool '{self.name}' was never called.")
        last = self.last_call
        for key, expected_val in expected_kwargs.items():
            actual_val = last.get(key)
            if actual_val != expected_val:
                raise AssertionError(
                    f"MockTool '{self.name}': expected kwarg '{key}' = {expected_val!r}, "
                    f"got {actual_val!r}."
                )

    def __repr__(self) -> str:
        return f"MockTool(name={self.name!r}, calls={self.call_count})"


class MockToolkit:
    """
    A collection of MockTools that can be injected into an agent together.

    Usage:
        toolkit = MockToolkit()
        toolkit.add("web_search", returns="Latest news: ...")
        toolkit.add("calculator", returns="42")
        toolkit.add("send_email", returns="Email sent.")

        # Access tools
        agent = MyAgent(tools=toolkit.as_dict())

        # After running:
        toolkit["web_search"].assert_called()
        toolkit["calculator"].assert_called_once()
        toolkit["send_email"].assert_not_called()
    """

    def __init__(self):
        self._tools: Dict[str, MockTool] = {}

    def add(
        self,
        name: str,
        returns: Any = "",
        side_effect: Optional[Callable] = None,
        description: str = "",
    ) -> MockTool:
        tool = MockTool(name, returns=returns, side_effect=side_effect, description=description)
        self._tools[name] = tool
        return tool

    def __getitem__(self, name: str) -> MockTool:
        if name not in self._tools:
            raise KeyError(f"MockTool '{name}' not found in toolkit. Available: {list(self._tools.keys())}")
        return self._tools[name]

    def as_dict(self) -> Dict[str, MockTool]:
        """Return tools as a dict {name: MockTool}."""
        return self._tools.copy()

    def as_list(self) -> List[MockTool]:
        """Return tools as a list."""
        return list(self._tools.values())

    def reset_all(self):
        """Reset call history for all mock tools."""
        for tool in self._tools.values():
            tool.reset()

    @property
    def call_summary(self) -> Dict[str, int]:
        """Return a dict of {tool_name: call_count} for all tools."""
        return {name: tool.call_count for name, tool in self._tools.items()}

    def assert_tool_not_called(self, name: str):
        tool = self[name]
        if tool.was_called:
            raise AssertionError(
                f"MockTool '{name}' was expected to NOT be called, "
                f"but was called {tool.call_count} time(s)."
            )

    def __repr__(self) -> str:
        return f"MockToolkit({list(self._tools.keys())})"

