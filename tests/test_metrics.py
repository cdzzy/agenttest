"""Tests for the DeepEval-style metric presets library (agenttest.metrics)."""
from __future__ import annotations

import json

import pytest

from agenttest.core.case import AgentRun, AgentTestCase
from agenttest.metrics import (
    METRIC_PRESETS,
    AnswerRelevancyMetric,
    BaseMetric,
    BiasMetric,
    ContainsMetric,
    ErrorFreeMetric,
    ExactMatchMetric,
    FaithfulnessMetric,
    HallucinationMetric,
    JSONValidityMetric,
    LatencyMetric,
    LengthMetric,
    MetricAssertionMixin,
    MetricResult,
    SummarizationQualityMetric,
    TokenUsageMetric,
    ToolCountMetric,
    ToxicityMetric,
    assert_metrics,
    assert_preset,
    evaluate_metrics,
)

# ── Helpers ──────────────────────────────────────────────────────────────


def make_run(
    output="Hello, how can I help you today?",
    input="hi",
    duration_ms=100.0,
    tokens_used=0,
    tool_calls=None,
    error=None,
):
    return AgentRun(
        input=input,
        output=output,
        tool_calls=tool_calls or [],
        duration_ms=duration_ms,
        tokens_used=tokens_used,
        error=error,
    )


def judge_llm(score, reasoning="ok"):
    """Stub llm_fn returning a well-formed judge JSON payload."""
    return lambda prompt: json.dumps({"score": score, "reasoning": reasoning})


# ── Deterministic metrics ─────────────────────────────────────────────────


class TestDeterministicMetrics:
    def test_error_free_passes_and_fails(self):
        assert ErrorFreeMetric().measure(make_run()).passed
        failed = ErrorFreeMetric().measure(make_run(error=RuntimeError("boom")))
        assert not failed.passed
        assert failed.score == 0.0
        assert "RuntimeError" in failed.reason

    def test_latency_within_budget_scores_high(self):
        run = make_run(duration_ms=100.0)
        result = LatencyMetric(max_ms=1000).measure(run)
        assert result.passed
        assert result.score > 0.5
        assert result.score <= 1.0

    def test_latency_over_budget_fails(self):
        run = make_run(duration_ms=2000.0)
        result = LatencyMetric(max_ms=1000).measure(run)
        assert not result.passed
        assert result.score < 0.5
        assert "over budget" in result.reason

    def test_latency_score_is_monotonic(self):
        fast = LatencyMetric(max_ms=1000).measure(make_run(duration_ms=50)).score
        slow = LatencyMetric(max_ms=1000).measure(make_run(duration_ms=500)).score
        assert fast > slow

    def test_token_usage_within_budget(self):
        result = TokenUsageMetric(max_tokens=1000).measure(make_run(tokens_used=250))
        assert result.passed
        assert result.score == pytest.approx(0.75)

    def test_token_usage_over_budget(self):
        result = TokenUsageMetric(max_tokens=100).measure(make_run(tokens_used=400))
        assert not result.passed

    def test_token_usage_zero_recorded(self):
        result = TokenUsageMetric(max_tokens=100).measure(make_run(tokens_used=0))
        assert result.passed

    def test_tool_count_within_and_over(self):
        calls = [{"name": f"tool_{i}"} for i in range(3)]
        assert ToolCountMetric(max_calls=5).measure(make_run(tool_calls=calls)).passed
        assert not ToolCountMetric(max_calls=2).measure(make_run(tool_calls=calls)).passed

    def test_exact_match(self):
        run = make_run(output="Paris")
        assert ExactMatchMetric("paris").measure(run).passed
        assert not ExactMatchMetric("London").measure(run).passed
        # case_sensitive=True rejects differing case
        assert not ExactMatchMetric("paris", case_sensitive=True).measure(run).passed

    def test_contains(self):
        run = make_run(output="The capital of France is Paris.")
        assert ContainsMetric("paris").measure(run).passed
        miss = ContainsMetric("Berlin").measure(run)
        assert not miss.passed
        assert "Berlin" in miss.reason

    def test_json_validity(self):
        assert JSONValidityMetric().measure(make_run(output='{"a": 1}')).passed
        assert JSONValidityMetric().measure(
            make_run(output='```json\n{"a": 1}\n```')
        ).passed
        bad = JSONValidityMetric().measure(make_run(output="not json {"))
        assert not bad.passed

    def test_length_bounds(self):
        assert LengthMetric(min_chars=5, max_chars=10).measure(make_run(output="12345")).passed
        assert not LengthMetric(min_chars=10).measure(make_run(output="short")).passed
        assert not LengthMetric(max_chars=3).measure(make_run(output="too long here")).passed


# ── LLM metrics: judge path ──────────────────────────────────────────────


class TestLLMMetricJudgePath:
    def test_answer_relevancy_with_llm_pass(self):
        metric = AnswerRelevancyMetric(threshold=0.7, llm_fn=judge_llm(0.9, "on topic"))
        result = metric.measure(make_run())
        assert result.passed
        assert result.score == pytest.approx(0.9)
        assert result.reason == "on topic"

    def test_answer_relevancy_with_llm_below_threshold(self):
        metric = AnswerRelevancyMetric(threshold=0.7, llm_fn=judge_llm(0.4, "off topic"))
        result = metric.measure(make_run())
        assert not result.passed
        assert result.score == pytest.approx(0.4)

    def test_threshold_override_flips_verdict(self):
        m1 = AnswerRelevancyMetric(threshold=0.5, llm_fn=judge_llm(0.6))
        m2 = AnswerRelevancyMetric(threshold=0.8, llm_fn=judge_llm(0.6))
        assert m1.measure(make_run()).passed
        assert not m2.measure(make_run()).passed

    def test_unparseable_judge_response_fails_closed(self):
        metric = AnswerRelevancyMetric(llm_fn=lambda prompt: "I refuse to answer in JSON")
        result = metric.measure(make_run())
        assert not result.passed
        assert result.score == 0.0
        assert "Could not parse" in result.reason

    def test_judge_exception_fails_closed(self):
        def exploding(prompt):
            raise RuntimeError("API down")

        metric = FaithfulnessMetric(llm_fn=exploding)
        result = metric.measure(make_run(), context="source text")
        assert not result.passed
        assert "API down" in result.reason

    def test_score_clamped_to_unit_interval(self):
        metric = ToxicityMetric(llm_fn=judge_llm(7.5))  # out-of-range score
        result = metric.measure(make_run())
        assert result.score == 1.0

    def test_judge_response_with_markdown_fence(self):
        payload = '```json\n{"score": 0.9, "reasoning": "clean"}\n```'
        metric = ToxicityMetric(threshold=0.7, llm_fn=lambda p: payload)
        result = metric.measure(make_run())
        assert result.passed

    def test_prompt_includes_input_output_and_context(self):
        seen = {}

        def spy(prompt):
            seen["prompt"] = prompt
            return json.dumps({"score": 1.0, "reasoning": "ok"})

        metric = FaithfulnessMetric(llm_fn=spy)
        metric.measure(make_run(input="What is the refund policy?", output="Refunds within 30 days."),
                       context=["Refunds allowed within 30 days of purchase."])
        prompt = seen["prompt"]
        assert "refund policy" in prompt.lower()
        assert "30 days" in prompt
        assert "[1]" in prompt  # numbered context chunk

    def test_all_llm_metrics_produce_named_results(self):
        run = make_run(input="Summarize this", output="A short summary.")
        metrics = [
            AnswerRelevancyMetric(llm_fn=judge_llm(1.0)),
            FaithfulnessMetric(llm_fn=judge_llm(1.0)),
            HallucinationMetric(llm_fn=judge_llm(1.0)),
            SummarizationQualityMetric(llm_fn=judge_llm(1.0)),
            ToxicityMetric(llm_fn=judge_llm(1.0)),
            BiasMetric(llm_fn=judge_llm(1.0)),
        ]
        for metric in metrics:
            result = metric.measure(run, context="Some source content about the topic.")
            assert isinstance(result, MetricResult)
            assert result.passed
            assert result.metric_name


# ── LLM metrics: rule-based fallbacks ─────────────────────────────────────


class TestLLMMetricFallbacks:
    def test_answer_relevancy_fallback_rewards_overlap(self):
        run = make_run(
            input="What is the capital of France?",
            output="The capital of France is Paris.",
        )
        result = AnswerRelevancyMetric().measure(run)
        assert result.passed
        assert "Fallback" in result.reason

    def test_answer_relevancy_fallback_penalizes_disjoint_output(self):
        run = make_run(
            input="What is the capital of France?",
            output="The mitochondria is the powerhouse of the cell.",
        )
        result = AnswerRelevancyMetric().measure(run)
        assert not result.passed

    def test_faithfulness_fallback_without_context_is_neutral(self):
        result = FaithfulnessMetric().measure(make_run())
        assert result.score == 0.5
        assert not result.passed  # default threshold 0.7

    def test_faithfulness_fallback_checks_support(self):
        run = make_run(
            output="The refund window is 30 days. The warranty covers two years."
        )
        supported = FaithfulnessMetric().measure(
            run, context="Our refund window is 30 days. Our warranty covers two years."
        )
        assert supported.passed

        mixed = FaithfulnessMetric().measure(
            run, context="Our refund window is 30 days."
        )
        assert not mixed.passed

    def test_hallucination_fallback_requires_context(self):
        result = HallucinationMetric().measure(make_run())
        assert result.score == 0.5
        assert "no context" in result.reason.lower()

    def test_summarization_fallback_rewards_compression(self):
        source = (
            "Q3 revenue was $4.2M, up 12% quarter over quarter. "
            "Costs fell 3% due to vendor consolidation. "
            "We hired 18 engineers, ending headcount at 240. "
        ) * 3
        good = SummarizationQualityMetric().measure(
            make_run(
                output="Q3 revenue grew 12% to $4.2M, costs fell 3%, and "
                       "headcount rose to 240 after hiring 18 engineers."
            ),
            context=source,
        )
        assert good.passed

        # Copying the source wholesale is a bad summary.
        bad = SummarizationQualityMetric().measure(
            make_run(output=source), context=source
        )
        assert not bad.passed

    def test_toxicity_fallback_detects_terms(self):
        toxic = ToxicityMetric().measure(make_run(output="You are an idiot, this is stupid."))
        assert not toxic.passed
        clean = ToxicityMetric().measure(make_run(output="Here is a clear, neutral answer."))
        assert clean.passed

    def test_bias_fallback_detects_patterns(self):
        biased = BiasMetric().measure(
            make_run(output="Women are bad at math, obviously.")
        )
        assert not biased.passed
        fair = BiasMetric().measure(
            make_run(output="Anyone can learn math with practice.")
        )
        assert fair.passed


# ── Presets & batch evaluation ───────────────────────────────────────────


class TestPresetsAndBatch:
    def test_presets_exist_and_return_fresh_instances(self):
        expected = {"smoke", "performance", "rag", "safety", "quality", "summarization"}
        assert expected <= set(METRIC_PRESETS)

        first = METRIC_PRESETS["smoke"]()
        second = METRIC_PRESETS["smoke"]()
        assert first is not second
        assert first[0] is not second[0]

        # tweaking one instance must not leak into the preset
        first[0].threshold = 0.99
        assert METRIC_PRESETS["smoke"]()[0].threshold != 0.99

    def test_preset_contents(self):
        assert {m.name for m in METRIC_PRESETS["rag"]()} == {
            "error_free", "answer_relevancy", "faithfulness"
        }
        assert {m.name for m in METRIC_PRESETS["safety"]()} == {"toxicity", "bias"}

    def test_evaluate_metrics_single_metric(self):
        results = evaluate_metrics(make_run(), ErrorFreeMetric())
        assert len(results) == 1
        assert results[0].passed

    def test_evaluate_metrics_list(self):
        run = make_run(output='{"ok": true}')
        results = evaluate_metrics(run, [ErrorFreeMetric(), JSONValidityMetric()])
        assert [r.passed for r in results] == [True, True]

    def test_evaluate_metrics_forwards_context(self):
        run = make_run(output="Paris is the capital.")
        results = evaluate_metrics(run, [FaithfulnessMetric()], context="Paris is the capital of France.")
        # All sentences supported by the source → fallback should pass at 0.7.
        assert results[0].passed

    def test_assert_metrics_all_pass(self):
        results = assert_metrics(make_run(output="Paris"), [ContainsMetric("paris"), ErrorFreeMetric()])
        assert len(results) == 2

    def test_assert_metrics_lists_every_failure(self):
        run = make_run(output="Paris")
        with pytest.raises(AssertionError) as excinfo:
            assert_metrics(run, [ContainsMetric("Berlin"), ErrorFreeMetric(), ExactMatchMetric("Rome")])
        message = str(excinfo.value)
        assert "2/3 metric(s) failed" in message
        assert "contains" in message
        assert "exact_match" in message
        assert "error_free" not in message  # passing metric not reported as failed

    def test_assert_preset_roundtrip(self):
        run = make_run(
            input="What is the capital of France?",
            output="The capital of France is Paris.",
            duration_ms=120.0,
        )
        results = assert_preset(run, "smoke")
        assert all(r.passed for r in results)

    def test_assert_preset_unknown_name_raises_keyerror(self):
        with pytest.raises(KeyError, match="nope"):
            assert_preset(make_run(), "nope")

    def test_metric_result_bool_and_str(self):
        result = MetricResult("m", 0.9, True, "fine", 0.7)
        assert bool(result) is True
        assert "m: 0.90" in str(result)
        assert "PASS" in str(result)


# ── Test-case integration ─────────────────────────────────────────────────


class _Agent:
    def __call__(self, text):
        return f"Echo: {text}"


class EchoTests(AgentTestCase, MetricAssertionMixin):
    agent = _Agent()


class TestTestCaseIntegration:
    def test_assert_meets_preset_method(self):
        case = EchoTests()
        run = case.invoke("hello world")
        assert "Echo: hello world" in run.output
        results = case.assert_meets_preset(run, "smoke")
        assert all(r.passed for r in results)

    def test_assert_metrics_pass_method_reports_failures(self):
        case = EchoTests()
        run = case.invoke("hello")
        with pytest.raises(AssertionError, match="exact_match"):
            case.assert_metrics_pass(run, [ExactMatchMetric("something else")])

    def test_top_level_exports(self):
        import agenttest

        assert agenttest.METRIC_PRESETS is METRIC_PRESETS
        assert callable(agenttest.evaluate_metrics)
        assert callable(agenttest.assert_metrics)
        assert callable(agenttest.assert_preset)
        assert issubclass(agenttest.LatencyMetric, BaseMetric)


# ── Custom metric subclassing ─────────────────────────────────────────────


class OddLengthMetric(BaseMetric):
    name = "odd_length"

    def __init__(self):
        super().__init__(threshold=0.5)

    def _evaluate(self, run, context):
        if len(run.output) % 2 == 1:
            return self._result(1.0, "Output length is odd")
        return self._result(0.0, "Output length is even")


class TestCustomMetrics:
    def test_user_defined_metric_composes_with_presets(self):
        run = make_run(output="abc")
        results = evaluate_metrics(run, [OddLengthMetric(), ErrorFreeMetric()])
        assert results[0].passed
        assert results[1].passed

        even_run = make_run(output="abcd")
        with pytest.raises(AssertionError, match="odd_length"):
            assert_metrics(even_run, [OddLengthMetric()])
