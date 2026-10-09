# Changelog

All notable changes to AgentTest are documented in this file.

## [Unreleased]

### Changed

- **Python 3.13 classifier**: the declared supported-Python list in
  `pyproject.toml` now includes 3.13, matching the already-green CI matrix
  (tests were passing on 3.13 but the classifier advertised only up to 3.12).
- **`__version__` drift-proofing** (`agenttest/__init__.py`): the source-tree
  fallback now reads `version` from `pyproject.toml` instead of a hard-coded
  literal, so a fresh checkout can never report a version that lags the
  declared package version. Installed environments keep reading the real
  distribution metadata.

## [0.6.0] - 2026-10-02

### Added

- **Flaky test detection** — a complete quarantine workflow for agent tests whose
  outcomes vary across runs:
  - The pytest plugin records every `@agent_test` outcome (pass/fail per run) to
    `.agenttest-history.jsonl` and gains two CLI options: `--agenttest-repeat N`
    (override the per-test repeat count) and `--agenttest-tolerate-flaky` (defer
    the verdict when repeats give mixed results, instead of failing the run).
  - New `python -m agenttest.check_flaky` command analyzes the history file,
    aggregates per-test results across runs, and exits non-zero under
    `--fail-on-flaky` when flaky tests are found. Requires Python 3.9+.
  - The GitHub Action gains a `fail-on-flaky` input: when `true`, CI re-runs the
    suite with `--agenttest-repeat 3 --agenttest-tolerate-flaky` and gates the
    build on `check_flaky` — flaky tests surface in the report instead of
    flapping the whole pipeline.
  - `.agenttest-history.jsonl` added to `.gitignore`.

## [0.5.0] - 2026-10-02

### Added

- **DeepEval-style metric presets library** (`agenttest/metrics.py`): runs are scored 0.0–1.0 and compared against per-metric thresholds, instead of binary assertions.
  - `MetricResult` — score, pass/fail against the metric's threshold, a human-readable reason, and an optional details dict.
  - 8 deterministic metrics: `ErrorFreeMetric`, `LatencyMetric`, `TokenUsageMetric`, `ToolCountMetric`, `ExactMatchMetric`, `ContainsMetric`, `JSONValidityMetric`, `LengthMetric`.
  - 6 LLM-as-judge metrics with an optional `llm_fn` and deterministic rule-based fallbacks: `AnswerRelevancyMetric`, `FaithfulnessMetric`, `HallucinationMetric`, `SummarizationQualityMetric`, `ToxicityMetric`, `BiasMetric`. Judge invocation or parse failures fail closed (score 0). Toxicity/Bias score the *absence* of harm, so "higher is better" holds for every metric.
  - 6 named presets (`METRIC_PRESETS`): `smoke`, `performance`, `rag`, `safety`, `quality`, `summarization` — factory callables returning fresh metric instances.
  - Batch API: `evaluate_metrics()`, `assert_metrics()`, `assert_preset()`; plus `MetricAssertionMixin` and `AgentTestCase.assert_metrics_pass()` / `AgentTestCase.assert_meets_preset()` for test-case style.
  - Custom metrics: subclass `BaseMetric` and implement `_evaluate(run, context)`; threshold comparison and score clamping are handled by the base class.
  - 43 new tests in `tests/test_metrics.py` (111 total), plus `examples/test_metrics.py`.

## [0.4.2] - 2026-09-25

### Changed

- Ruff rule selection declared in `pyproject.toml` (`E4/E7/E9/F/I`) and pinned `ruff==0.16.8` in CI; the lint job now runs `ruff check .` over the whole repo. Typing-modernization rules are deliberately excluded to stay Python 3.9 compatible.
- `__version__` fixed — it had lagged at 0.4.0 while the package was already at 0.4.1.

### Fixed

- 85 ruff findings fixed repo-wide: unused imports (F401), placeholder-less f-strings (F541), unsorted/multiple imports (I001/E401), dead variables (F841), and an ambiguous loop variable (E741).
- Assertion submodules now declare explicit `__all__`, documenting the public API surface.
- `TestStatus` / `TestResult` set `__test__ = False`, so pytest no longer emits collection warnings when test modules import them.

## [0.4.1] - 2026-09-20

### Changed

- README repositioned around chaos engineering and deterministic testing (English + Chinese).
- CI: test matrix 3.9–3.13 with fail-fast off, coverage on 3.12, and a dedicated ruff lint job scoped to error-class checks.

### Fixed

- Missing `TYPE_CHECKING` import in `agenttest/loaders/yaml_loader.py` (ruff F821); no runtime change.

## [0.4.0] - 2026-08-27

### Added

- **Pytest plugin** (`pytest11` entry point): run `@agent_test` functions under vanilla pytest. Provides the `agent` fixture (resolvable via the `agenttest_agent` ini key / `--agenttest-agent` CLI option, or a conftest override), honors `repeat=N` as stability aggregation (all runs must pass), and maps `skip=True` to pytest skip markers. Non-agent tests are untouched.

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
