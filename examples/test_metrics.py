"""
Example: Metric presets — DeepEval-style scored quality evaluation.

Deterministic metrics, LLM-as-judge metrics (with a fake judge here), and
one-line named presets.

Run with: python examples/test_metrics.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agenttest import (
    METRIC_PRESETS,
    AnswerRelevancyMetric,
    ErrorFreeMetric,
    FaithfulnessMetric,
    LatencyMetric,
    TokenUsageMetric,
    assert_preset,
    evaluate_metrics,
)
from agenttest.core.case import AgentRun

# ─────────────────────────────────────────────────────────────────
# A fake RAG agent over a source document
# ─────────────────────────────────────────────────────────────────

SOURCE = (
    "Q3 revenue was $4.2M, up 12% quarter over quarter. "
    "Costs fell 3% due to vendor consolidation. "
    "We hired 18 engineers, ending headcount at 240."
)


def rag_agent(input_text: str, **kwargs) -> dict:
    """Grounded, concise, and uses one retrieval tool."""
    return {
        "output": (
            "Q3 revenue grew 12% quarter over quarter to $4.2M, "
            "costs fell 3% after vendor consolidation, and "
            "headcount ended at 240 after hiring 18 engineers."
        ),
        "tool_calls": [{"name": "docs_search", "input": input_text, "output": SOURCE}],
        "duration_ms": 412.0,
        "tokens_used": 380,
    }


def hallucinating_agent(input_text: str, **kwargs) -> dict:
    """Invents numbers that are not in the source — Faithfulness should catch it."""
    return {
        "output": "Q3 revenue was $9.9M and we fired 40% of staff.",
        "tool_calls": [{"name": "docs_search", "input": input_text, "output": SOURCE}],
    }


def summarizing_agent(input_text: str, **kwargs) -> dict:
    """Produces a real compressed summary of the source — the 'summarization' target."""
    return {
        "output": (
            "Q3 revenue grew 12% to $4.2M, costs fell 3%, and "
            "headcount rose to 240 after hiring 18 engineers."
        ),
        "tool_calls": [{"name": "docs_read", "input": input_text, "output": SOURCE}],
    }


def fake_judge(prompt: str) -> str:
    """Stand-in for a real judge LLM.

    A real judge reads the question/response/source embedded in the prompt;
    this fake simply checks for the hallucinated figures of the bad agent.
    """
    if "$9.9M" in prompt or "fired" in prompt:
        return '{"score": 0.15, "reason": "Contradicts the source document."}'
    return '```json\n{"score": 0.9, "reason": "Grounded and on-topic."}\n```'


def to_run(agent, question: str) -> AgentRun:
    """Invoke an agent and lift its dict payload into an AgentRun."""
    payload = agent(question)
    return AgentRun(
        input=question,
        output=payload.get("output", ""),
        tool_calls=payload.get("tool_calls", []),
        duration_ms=payload.get("duration_ms", 0.0),
        tokens_used=payload.get("tokens_used", 0),
    )


# ─────────────────────────────────────────────────────────────────
# Demo
# ─────────────────────────────────────────────────────────────────

def show(result):
    status = "PASS" if result.passed else "FAIL"
    print(f"  [{status}] {result.metric_name:<24} score={result.score:.2f}  {result.reason}")


def main():
    question = "How did Q3 revenue, costs, and headcount develop?"
    run = to_run(rag_agent, question)
    bad = to_run(hallucinating_agent, question)

    print("─" * 70)
    print("1. Single metric → MetricResult (score, not just pass/fail)")
    print("─" * 70)
    show(LatencyMetric(max_ms=2_000).measure(run))

    print()
    print("─" * 70)
    print("2. Batch evaluation (no assert — collect the report)")
    print("─" * 70)
    for r in evaluate_metrics(
        run,
        [ErrorFreeMetric(), LatencyMetric(max_ms=2_000), TokenUsageMetric(max_tokens=4_000)],
    ):
        show(r)

    print()
    print("─" * 70)
    print("3. LLM-as-judge (fake judge here; real one via llm_fn=your_model)")
    print("─" * 70)
    show(AnswerRelevancyMetric(llm_fn=fake_judge).measure(run, context=question))
    show(FaithfulnessMetric(llm_fn=fake_judge).measure(run, context=SOURCE))
    show(FaithfulnessMetric(llm_fn=fake_judge).measure(bad, context=SOURCE))

    print()
    print("─" * 70)
    print("4. Named presets — one-liners with opinionated defaults")
    print("─" * 70)
    print(f"  available presets: {sorted(METRIC_PRESETS)}")
    for r in assert_preset(run, "rag", context=SOURCE):
        show(r)

    print()
    print("  Presets cover more than RAG — 'summarization' on a real summary:")
    summary_run = to_run(summarizing_agent, "Summarize the Q3 report.")
    for r in assert_preset(summary_run, "summarization", context=SOURCE):
        show(r)

    print()
    print("✅ The grounded agent passes 'rag'; the hallucinating one would not:")
    try:
        assert_preset(bad, "rag", context=SOURCE)
    except AssertionError as exc:
        print(f"  AssertionError: {exc}")

    print()
    print("Done. Try plugging a real judge:  AnswerRelevancyMetric(llm_fn=my_llm)")


if __name__ == "__main__":
    main()
