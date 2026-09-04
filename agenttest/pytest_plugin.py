"""
Pytest plugin for AgentTest (Roadmap: pytest-agenttest integration).

Runs ``@agent_test``-decorated functions under vanilla pytest with native
fixture support, stability (repeat) semantics, and skip handling.

Activation
----------
The plugin registers automatically via the ``pytest11`` entry point when
agenttest is installed. You can also load it explicitly::

    pytest -p agenttest.pytest_plugin tests/

Providing the agent
-------------------
Tests receive the agent through the ``agent`` fixture, which returns a
callable ``agent(input_text) -> AgentRun``. Configure the underlying agent
callable via ini/CLI::

    # pyproject.toml
    [tool.pytest.ini_options]
    agenttest_agent = "myapp.agents:support_agent"

or override the ``agent`` fixture in your conftest.py::

    @pytest.fixture
    def agent():
        return wrap_agent(my_agent)   # or a raw callable

Stability semantics
-------------------
``@agent_test(repeat=N)`` runs the test N times under pytest and requires
all runs to pass — mirroring the standalone runner's stability aggregation.
"""

from __future__ import annotations

import inspect
from typing import Any, Callable, Optional

import pytest

from agenttest.core.runner import _wrap_agent_as_sync_invoke


def pytest_addoption(parser: Any) -> None:
    group = parser.getgroup("agenttest")
    group.addoption(
        "--agenttest-agent",
        action="store",
        dest="agenttest_agent",
        default=None,
        help="Agent callable as 'module:attribute' (e.g. 'myapp.agents:support_agent')",
    )
    parser.addini(
        "agenttest_agent",
        help="Agent callable as 'module:attribute' for the `agent` fixture",
        default=None,
    )


def pytest_configure(config: Any) -> None:
    config.addinivalue_line(
        "markers",
        "agent_test: mark a test as an AgentTest case (applied automatically by @agent_test)",
    )


def _resolve_raw_agent(config: Any) -> Optional[Callable]:
    """Resolve the raw agent callable from CLI option or ini setting."""
    spec = config.getoption("agenttest_agent") or config.getini("agenttest_agent")
    if not spec:
        return None
    module_name, _, attr = str(spec).partition(":")
    import importlib

    obj = importlib.import_module(module_name.strip())
    if attr:
        return getattr(obj, attr.strip())
    return obj


@pytest.fixture
def agent(request: Any) -> Callable:
    """
    Provide the agent under test as ``agent(input_text) -> AgentRun``.

    Resolution order:
    1. A user-overridden ``agent`` fixture (conftest.py).
    2. The ``agenttest_agent`` ini setting / ``--agenttest-agent`` CLI option.
    """
    raw = _resolve_raw_agent(request.config)
    if raw is None:
        pytest.fail(
            "No agent configured for agenttest. "
            "Set agenttest_agent = 'yourmodule:agent_fn' in pytest config, "
            "or override the 'agent' fixture in conftest.py."
        )
    return _wrap_agent_as_sync_invoke(raw)


def pytest_collection_modifyitems(config: Any, items: list) -> None:
    """Honor @agent_test(skip=True) as a pytest skip marker."""
    for item in items:
        fn = getattr(item, "obj", None)
        if fn is not None and getattr(fn, "_skip", False):
            reason = getattr(fn, "_skip_reason", "") or "agent_test skip"
            item.add_marker(pytest.mark.skip(reason=reason))


def pytest_pyfunc_call(pyfuncitem: Any) -> Optional[bool]:
    """
    Run @agent_test functions with stability semantics.

    Returns True when the test was handled here (agent test), None to let
    pytest's default machinery run everything else.
    """
    fn = pyfuncitem.obj
    if not getattr(fn, "_is_agent_test", False):
        return None

    sig = inspect.signature(fn)
    kwargs = {
        name: value
        for name, value in pyfuncitem.funcargs.items()
        if name in sig.parameters
    }

    repeat = int(getattr(fn, "_repeat", 1) or 1)
    if repeat <= 1:
        fn(**kwargs)
        return True

    # Stability aggregation: every run must pass (mirrors AgentTestRunner)
    passed = 0
    first_failure = ""
    for _ in range(repeat):
        try:
            fn(**kwargs)
            passed += 1
        except Exception as e:  # noqa: BLE001 - aggregate below
            if not first_failure:
                first_failure = str(e)
    if passed < repeat:
        raise AssertionError(
            f"Stability: {passed}/{repeat} runs passed (required 100%). "
            f"First failure: {first_failure}"
        )
    return True
