"""
Test runner - executes tests and collects results.
"""
from __future__ import annotations

import inspect
import sys
import time
import traceback
from typing import Callable, List, Optional

from agenttest.core.case import AgentRun, TestResult, TestStatus
from agenttest.core.suite import AgentTestSuite


def _normalize_result(run: AgentRun, result) -> None:
    """Normalize an agent's raw return value into the AgentRun fields."""
    if isinstance(result, dict):
        run.output = result.get("output", result.get("response", str(result)))
        run.tool_calls = result.get("tool_calls", result.get("intermediate_steps", []))
        run.reasoning_steps = result.get("reasoning_steps", result.get("thoughts", []))
        run.tokens_used = result.get("usage", {}).get("total_tokens", 0)
        run.metadata = {k: v for k, v in result.items()
                        if k not in ("output", "tool_calls", "reasoning_steps")}
    elif isinstance(result, str):
        run.output = result
    else:
        run.output = str(result)


def _wrap_agent_as_invoke(raw_agent: Callable) -> Callable:
    """
    Wrap a raw agent callable so it returns AgentRun instead of raw output.
    This ensures @agent_test functions receive proper AgentRun objects.

    If the agent is async, an async wrapper is returned; otherwise a sync one.
    """
    import time as _time

    if inspect.iscoroutinefunction(raw_agent):
        async def wrapped_invoke(input_text: str, **kwargs) -> AgentRun:
            start = _time.perf_counter()
            run = AgentRun(input=input_text, output="")
            try:
                _normalize_result(run, await raw_agent(input_text, **kwargs))
            except Exception as e:
                run.error = e
            run.duration_ms = (_time.perf_counter() - start) * 1000
            return run
        return wrapped_invoke

    def wrapped_invoke(input_text: str, **kwargs) -> AgentRun:
        start = _time.perf_counter()
        run = AgentRun(input=input_text, output="")
        try:
            _normalize_result(run, raw_agent(input_text, **kwargs))
        except Exception as e:
            run.error = e
        run.duration_ms = (_time.perf_counter() - start) * 1000
        return run

    return wrapped_invoke


def _wrap_agent_as_sync_invoke(raw_agent: Callable) -> Callable:
    """
    Wrap an (optionally async) agent into a *sync* AgentRun-returning callable.

    For async agents, this runs the coroutine via ``asyncio.run`` internally,
    so sync test functions can call it without awaiting.
    """
    import asyncio

    async_wrapper = _wrap_agent_as_invoke(raw_agent)
    if not inspect.iscoroutinefunction(async_wrapper):
        return async_wrapper

    def wrapped_invoke(input_text: str, **kwargs) -> AgentRun:
        return asyncio.run(async_wrapper(input_text, **kwargs))

    return wrapped_invoke


class AgentTestRunner:
    """
    Executes agent tests and aggregates results.

    Usage:
        runner = AgentTestRunner(verbose=True)
        results = runner.run_suite(suite)
        runner.print_summary(results)
    """

    def __init__(self, verbose: bool = True, fail_fast: bool = False):
        self.verbose = verbose
        self.fail_fast = fail_fast

    def run_suite(self, suite: AgentTestSuite) -> List[TestResult]:
        """Run all tests in a suite and return results."""
        collected = suite.collect()
        results = []

        if self.verbose:
            print(f"\n{'='*60}")
            print(f"  AgentTest -- {suite.name}")
            print(f"  {len(collected)} test(s) collected")
            print(f"{'='*60}\n")

        for test_info in collected:
            result = self._run_one(test_info)
            results.append(result)

            if self.verbose:
                self._print_test_result(result)

            if self.fail_fast and result.status == TestStatus.FAILED:
                if self.verbose:
                    print("\n[STOP] Stopping early (--fail-fast)")
                break

        if self.verbose:
            self.print_summary(results)

        return results

    def run_suite_with_report(
        self,
        suite: AgentTestSuite,
        report_path: str,
        verbose: Optional[bool] = None,
    ) -> List[TestResult]:
        """
        Run a suite and write an HTML/JSON report in one call.

        Args:
            suite: The suite to run.
            report_path: Output path (``.html`` or ``.json``).
            verbose: Override the runner's verbose flag for this run.

        Returns:
            The list of TestResults (and writes the report file).

        Example::

            runner = AgentTestRunner(verbose=True)
            runner.run_suite_with_report(suite, "report.html")
        """
        from agenttest.report import save_report

        prev = self.verbose
        if verbose is not None:
            self.verbose = verbose
        try:
            results = self.run_suite(suite)
        finally:
            self.verbose = prev
        save_report(results, report_path)
        return results

    def run_function(self, fn: Callable, agent: Callable, name: Optional[str] = None) -> TestResult:
        """Run a single test function directly."""
        test_info = {
            "name": name or getattr(fn, "_test_name", fn.__name__),
            "fn": fn,
            "agent": agent,
            "tags": getattr(fn, "_test_tags", []),
            "skip": getattr(fn, "_skip", False),
            "skip_reason": getattr(fn, "_skip_reason", ""),
            "repeat": getattr(fn, "_repeat", 1),
        }
        return self._run_one(test_info)

    def _run_one(self, test_info: dict) -> TestResult:
        name = test_info["name"]
        fn = test_info["fn"]
        repeat = test_info.get("repeat", 1)

        # Handle skip
        if test_info.get("skip"):
            return TestResult(
                test_name=name,
                status=TestStatus.SKIPPED,
                error_message=test_info.get("skip_reason", "Skipped"),
            )

        all_results = []
        start = time.perf_counter()

        for i in range(repeat):
            result = self._execute(name, fn, test_info, run_index=i)
            all_results.append(result)

            # For repeat runs, stop on first failure
            if result.status == TestStatus.FAILED and repeat > 1:
                break

        total_ms = (time.perf_counter() - start) * 1000

        # For stability tests (repeat > 1), aggregate results
        if repeat > 1:
            return self._aggregate_stability(name, all_results, total_ms)

        return all_results[0]

    def _execute(self, name: str, fn: Callable, test_info: dict, run_index: int = 0) -> TestResult:
        import asyncio

        start = time.perf_counter()
        result = TestResult(test_name=name, status=TestStatus.RUNNING)

        try:
            sig = inspect.signature(fn)
            params = list(sig.parameters.keys())
            fn_is_async = inspect.iscoroutinefunction(fn)

            if "agent" in params and "agent" in test_info:
                raw_agent = test_info["agent"]
                if fn_is_async:
                    # async test fn: use an async wrapper (await inside fn)
                    wrapped = _wrap_agent_as_invoke(raw_agent)
                    asyncio.run(fn(wrapped))
                else:
                    # sync test fn: use a sync wrapper (handles async agents)
                    wrapped = _wrap_agent_as_sync_invoke(raw_agent)
                    fn(wrapped)
            else:
                if fn_is_async:
                    asyncio.run(fn())
                else:
                    fn()

            result.status = TestStatus.PASSED

        except AssertionError as e:
            result.status = TestStatus.FAILED
            result.error_message = str(e)

        except Exception as e:
            result.status = TestStatus.ERROR
            result.error_message = f"{type(e).__name__}: {e}\n{traceback.format_exc()}"

        result.duration_ms = (time.perf_counter() - start) * 1000
        return result

    def _aggregate_stability(self, name: str, results: List[TestResult], total_ms: float) -> TestResult:
        """Aggregate multiple runs for stability testing."""
        passed = sum(1 for r in results if r.status == TestStatus.PASSED)
        total = len(results)
        pass_rate = passed / total

        status = TestStatus.PASSED if pass_rate == 1.0 else TestStatus.FAILED
        message = None if status == TestStatus.PASSED else (
            f"Stability: {passed}/{total} runs passed ({pass_rate*100:.0f}%). "
            f"Required: 100%. First failure: {next(r.error_message for r in results if r.status != TestStatus.PASSED)}"
        )

        return TestResult(
            test_name=f"{name} [stability x{total}]",
            status=status,
            duration_ms=total_ms,
            error_message=message,
        )

    def _print_test_result(self, result: TestResult):
        icon = {
            TestStatus.PASSED:  "PASS",
            TestStatus.FAILED:  "FAIL",
            TestStatus.ERROR:   "ERROR",
            TestStatus.SKIPPED: "SKIP",
        }.get(result.status, "?")

        duration = f"({result.duration_ms:.0f}ms)"
        print(f"  {icon}  {result.test_name}  {duration}")

        if result.error_message and result.status != TestStatus.SKIPPED:
            for line in result.error_message.strip().split("\n")[:5]:
                print(f"       {line}")

    def print_summary(self, results: List[TestResult]):
        passed  = sum(1 for r in results if r.status == TestStatus.PASSED)
        failed  = sum(1 for r in results if r.status == TestStatus.FAILED)
        errors  = sum(1 for r in results if r.status == TestStatus.ERROR)
        skipped = sum(1 for r in results if r.status == TestStatus.SKIPPED)
        total_ms = sum(r.duration_ms for r in results)

        print(f"\n{'='*60}")
        summary_parts = []
        if passed:  summary_parts.append(f"[PASS] {passed} passed")
        if failed:  summary_parts.append(f"[FAIL] {failed} failed")
        if errors:  summary_parts.append(f"[ERROR] {errors} errors")
        if skipped: summary_parts.append(f"[SKIP] {skipped} skipped")
        print("  " + "  |  ".join(summary_parts))
        print(f"  Total time: {total_ms:.0f}ms")
        print(f"{'='*60}\n")

        return failed + errors == 0

