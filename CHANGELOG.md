# Changelog

All notable changes to AgentTest are documented in this file.

## [0.3.0] - 2026-08-19

### Added

- **HTML/JSON test reports**: `report.py` renders `List[TestResult]` into a self-contained HTML report (summary cards + per-test table + failure details) or a JSON report for CI tooling. `save_report()` infers format from the extension; `AgentTestRunner.run_suite_with_report(suite, path)` runs and writes in one call.

## [0.2.0] - 2026-08-15

### Added

- **Behavior snapshots** (`#1`): `SnapshotStore` + `snapshot` decorator for git-trackable regression testing of non-deterministic outputs.
- **Async test support** (`#2`): the runner auto-detects `async def` test functions and async agents; added `test_async` decorator.
- **Flakiness detection** (`#3`): `FlakinessDetector` tracks per-test pass rates across runs.
- **GitHub Action** (`#4`): `action.yml` composite action for zero-config CI integration.
- **Benchmarking** (`#5`): `BenchmarkSuite` + `Scenario` with markdown/text comparison reports.
- **Reasoning model assertions** (`#6`): `assert_reasoning_steps`, `assert_reasoning_covers`, `assert_reasoning_order`, `assert_reasoning_time`.
- **MCP server testing** (`#7`): `MCPServerClient` (stdio) + `MCPTool`, `assert_tool_exists`, `assert_tool_response_valid`.
- **SuiteRunHistory** (`#8`): `SuiteRunHistory` / `RunSummary` / `RunTrendReport` with OLS pass-rate slope and regression detection.
- **Chaos engineering** (`#9`): `ChaosScenario`, `ChaosAgent`, `inject_chaos`, `run_chaos`, `measure_resilience`.
- **Visual test editor** (`#10`): browser-based editor (`agenttest edit --ui`) that exports to Python.

### Changed

- Added `agenttest` CLI (`agenttest edit --ui`, `agenttest --version`).

## [0.1.0]

- Initial release: test cases, suites, runner, behavior/output/trace/stability assertions, mock tools, scenarios.
