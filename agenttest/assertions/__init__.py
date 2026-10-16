"""
AgentTest assertions package.

Provides various assertion types for testing AI agents:
- behavior: Tool call and behavior assertions
- output: Output content assertions
- stability: Consistency and stability assertions
- trace: Execution trace assertions
- structured: Structured output (JSON Schema / pydantic) assertions
- judge: LLM-as-Judge semantic assertions (optional dependency)
"""

from .behavior import (
    assert_max_tool_calls,
    assert_no_tool_called,
    assert_tool_called,
    assert_tool_called_before,
    assert_tool_sequence,
)
from .output import (
    assert_output_contains,
    assert_output_contains_all,
    assert_output_json,
    assert_output_json_schema,
    assert_output_length,
    assert_output_matches,
    assert_output_not_contains,
    assert_output_sentiment,
)
from .stability import StabilityAssertion
from .structured import (
    assert_output_contains_fields,
    assert_output_field_equals,
    assert_output_is_valid_yaml,
    assert_output_structured,
)
from .trace import (
    assert_no_reasoning_loops,
    assert_reasoning_covers,
    assert_reasoning_order,
    assert_reasoning_step,
    assert_reasoning_steps,
    assert_reasoning_time,
    assert_step_count,
    assert_tool_reasoning_alignment,
)

# Optional: LLM-as-Judge assertions
try:
    from .judge import (
        JudgeAssertionMixin,
        JudgeCriterion,
        JudgeResult,
        LLMJudge,
    )
    _JUDGE_AVAILABLE = True
except ImportError:
    _JUDGE_AVAILABLE = False

__all__ = [
    # behavior
    "assert_max_tool_calls",
    "assert_no_tool_called",
    "assert_tool_called",
    "assert_tool_called_before",
    "assert_tool_sequence",
    # output
    "assert_output_contains",
    "assert_output_contains_all",
    "assert_output_json",
    "assert_output_json_schema",
    "assert_output_length",
    "assert_output_matches",
    "assert_output_not_contains",
    "assert_output_sentiment",
    # stability
    "StabilityAssertion",
    # structured
    "assert_output_contains_fields",
    "assert_output_field_equals",
    "assert_output_is_valid_yaml",
    "assert_output_structured",
    # trace
    "assert_no_reasoning_loops",
    "assert_reasoning_covers",
    "assert_reasoning_order",
    "assert_reasoning_step",
    "assert_reasoning_steps",
    "assert_reasoning_time",
    "assert_step_count",
    "assert_tool_reasoning_alignment",
]

if _JUDGE_AVAILABLE:
    __all__.extend([
        "LLMJudge",
        "JudgeResult",
        "JudgeCriterion",
        "JudgeAssertionMixin",
    ])
