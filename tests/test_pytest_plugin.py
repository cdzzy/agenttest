"""
Tests for the pytest plugin (v0.4.0).

Uses pytest's ``pytester`` fixture to run real pytest sessions against
inline test files with the plugin force-loaded.
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest

pytest_plugins = ["pytester"]

PLUGIN_ARGS = ("-p", "agenttest.pytest_plugin")

AGENT_MODULE = '''
def my_agent(text):
    return {"output": f"echo: {text}"}
'''


class TestPytestPlugin:
    def test_plain_tests_unaffected(self, pytester):
        pytester.makepyfile("""
            def test_normal():
                assert 1 + 1 == 2
        """)
        result = pytester.runpytest(*PLUGIN_ARGS)
        result.assert_outcomes(passed=1)

    def test_agent_test_with_ini_agent(self, pytester):
        pytester.makepyfile(agents_mod=AGENT_MODULE)
        pytester.makepyprojecttoml(
            '[tool.pytest.ini_options]\nagenttest_agent = "agents_mod:my_agent"\n'
        )
        pytester.makepyfile("""
            import agenttest as at

            @at.agent_test
            def test_echo(agent):
                run = agent("hello")
                assert run.output == "echo: hello"
        """)
        result = pytester.runpytest(*PLUGIN_ARGS)
        result.assert_outcomes(passed=1)

    def test_agent_test_via_cli_option(self, pytester):
        pytester.makepyfile(agents_mod=AGENT_MODULE)
        pytester.makepyfile("""
            import agenttest as at

            @at.agent_test
            def test_echo(agent):
                run = agent("hi")
                assert "echo: hi" in run.output
        """)
        result = pytester.runpytest(
            *PLUGIN_ARGS, "--agenttest-agent", "agents_mod:my_agent"
        )
        result.assert_outcomes(passed=1)

    def test_repeat_stability_failure(self, pytester):
        pytester.makepyfile(agents_mod=AGENT_MODULE)
        pytester.makepyprojecttoml(
            '[tool.pytest.ini_options]\nagenttest_agent = "agents_mod:my_agent"\n'
        )
        pytester.makepyfile("""
            import agenttest as at

            counter = {"n": 0}

            @at.agent_test(repeat=3)
            def test_flaky(agent):
                counter["n"] += 1
                if counter["n"] == 2:
                    raise AssertionError("second run failed")
        """)
        result = pytester.runpytest(*PLUGIN_ARGS)
        result.assert_outcomes(failed=1)
        result.stdout.fnmatch_lines(["*Stability: 2/3 runs passed*"])

    def test_repeat_stability_all_pass(self, pytester):
        pytester.makepyfile(agents_mod=AGENT_MODULE)
        pytester.makepyprojecttoml(
            '[tool.pytest.ini_options]\nagenttest_agent = "agents_mod:my_agent"\n'
        )
        pytester.makepyfile("""
            import agenttest as at

            @at.agent_test(repeat=3)
            def test_stable(agent):
                run = agent("x")
                assert run.output == "echo: x"
        """)
        result = pytester.runpytest(*PLUGIN_ARGS)
        result.assert_outcomes(passed=1)

    def test_skip_marker(self, pytester):
        pytester.makepyfile(agents_mod=AGENT_MODULE)
        pytester.makepyprojecttoml(
            '[tool.pytest.ini_options]\nagenttest_agent = "agents_mod:my_agent"\n'
        )
        pytester.makepyfile("""
            import agenttest as at

            @at.agent_test(skip=True, skip_reason="not ready")
            def test_skipped(agent):
                assert False
        """)
        result = pytester.runpytest(*PLUGIN_ARGS)
        result.assert_outcomes(skipped=1)

    def test_missing_agent_config_fails_with_hint(self, pytester):
        pytester.makepyfile("""
            import agenttest as at

            @at.agent_test
            def test_needs_agent(agent):
                assert True
        """)
        result = pytester.runpytest(*PLUGIN_ARGS)
        result.assert_outcomes(errors=1)
        result.stdout.fnmatch_lines(["*No agent configured for agenttest*"])

    def test_agent_test_without_agent_param(self, pytester):
        """Tests that don't take `agent` run without any configuration."""
        pytester.makepyfile("""
            import agenttest as at

            @at.agent_test
            def test_no_agent_needed():
                assert True
        """)
        result = pytester.runpytest(*PLUGIN_ARGS)
        result.assert_outcomes(passed=1)
