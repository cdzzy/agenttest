"""
Output assertions — verify what the agent actually said.

These assertions answer: "Is the agent's output correct and appropriate?"
"""
from __future__ import annotations

import re
from typing import List, Optional, Tuple

from agenttest.core.case import AgentRun


def _normalize_text(text: str, case_sensitive: bool) -> str:
    """Normalize text for case-insensitive comparison."""
    return text if case_sensitive else text.lower()


def _normalize_output(run: AgentRun, case_sensitive: bool) -> str:
    """Get output with optional case normalization."""
    return run.output if case_sensitive else run.output.lower()


def assert_output_contains(run: AgentRun, text: str, case_sensitive: bool = False):
    """
    Assert that the agent's output contains the expected text.

    Example:
        assert_output_contains(run, "Paris")
        assert_output_contains(run, "ERROR", case_sensitive=True)
    """
    output = _normalize_output(run, case_sensitive)
    needle = _normalize_text(text, case_sensitive)

    if needle not in output:
        raise AssertionError(
            f"Expected output to contain: {text!r}\n"
            f"Actual output: {run.output!r}"
        )


def assert_output_not_contains(run: AgentRun, text: str, case_sensitive: bool = False):
    """
    Assert that the agent's output does NOT contain the given text.

    Example:
        assert_output_not_contains(run, "I don't know")
        assert_output_not_contains(run, "error")
    """
    output = _normalize_output(run, case_sensitive)
    needle = _normalize_text(text, case_sensitive)

    if needle in output:
        raise AssertionError(
            f"Expected output to NOT contain: {text!r}\n"
            f"Actual output: {run.output!r}"
        )


def assert_output_matches(run: AgentRun, pattern: str, flags: int = re.IGNORECASE):
    """
    Assert that the agent's output matches a regex pattern.

    Example:
        assert_output_matches(run, r"\\d{4}-\\d{2}-\\d{2}")  # ISO date format
        assert_output_matches(run, r"(yes|no)", re.IGNORECASE)
    """
    if not re.search(pattern, run.output, flags):
        raise AssertionError(
            f"Expected output to match pattern: {pattern!r}\n"
            f"Actual output: {run.output!r}"
        )


def assert_output_length(run: AgentRun, min_chars: int = 0, max_chars: Optional[int] = None):
    """
    Assert the output is within a character length range.

    Example:
        assert_output_length(run, min_chars=10, max_chars=500)
    """
    length = len(run.output)

    if length < min_chars:
        raise AssertionError(
            f"Output too short: {length} chars (minimum: {min_chars})\n"
            f"Output: {run.output!r}"
        )

    if max_chars is not None and length > max_chars:
        raise AssertionError(
            f"Output too long: {length} chars (maximum: {max_chars})\n"
            f"Output (truncated): {run.output[:200]!r}..."
        )


def assert_output_sentiment(run: AgentRun, expected: str):
    """
    Simple keyword-based sentiment check.
    expected: "positive", "negative", or "neutral"

    Note: For production use, plug in a proper sentiment model.
    This is a lightweight heuristic for quick sanity checks.

    Example:
        assert_output_sentiment(run, "positive")
    """
    positive_words = {"great", "excellent", "good", "happy", "success", "wonderful",
                      "amazing", "helpful", "glad", "perfect", "love", "thank"}
    negative_words = {"error", "fail", "bad", "sorry", "cannot", "unable", "problem",
                      "issue", "wrong", "unfortunately", "hate", "terrible"}

    words = set(run.output.lower().split())
    pos_count = len(words & positive_words)
    neg_count = len(words & negative_words)

    if expected == "positive" and pos_count <= neg_count:
        raise AssertionError(
            f"Expected positive sentiment, but output seems neutral or negative.\n"
            f"Positive signals: {words & positive_words}\n"
            f"Negative signals: {words & negative_words}\n"
            f"Output: {run.output!r}"
        )
    elif expected == "negative" and neg_count <= pos_count:
        raise AssertionError(
            f"Expected negative sentiment, but output seems neutral or positive.\n"
            f"Output: {run.output!r}"
        )
    elif expected == "neutral" and (pos_count > 3 or neg_count > 3):
        raise AssertionError(
            f"Expected neutral sentiment, but output seems strongly {('positive' if pos_count > neg_count else 'negative')}.\n"
            f"Output: {run.output!r}"
        )


def assert_output_contains_all(run: AgentRun, texts: List[str], case_sensitive: bool = False):
    """
    Assert the output contains ALL of the given strings.

    Example:
        assert_output_contains_all(run, ["name", "age", "email"])
    """
    output = _normalize_output(run, case_sensitive)
    missing = [text for text in texts if _normalize_text(text, case_sensitive) not in output]

    if missing:
        raise AssertionError(
            f"Output missing {len(missing)} expected string(s): {missing}\n"
            f"Output: {run.output!r}"
        )


def assert_output_json(run: AgentRun):
    """Assert that the agent's output is valid JSON."""
    import json
    try:
        # Try to find JSON block in output
        output = run.output.strip()
        # Handle markdown code blocks
        if output.startswith("```"):
            lines = output.split("\n")
            output = "\n".join(lines[1:-1])
        json.loads(output)
    except json.JSONDecodeError as e:
        raise AssertionError(
            f"Expected valid JSON output, but got parse error: {e}\n"
            f"Output: {run.output!r}"
        )


def assert_output_json_schema(run: AgentRun, schema: dict):
    """
    Assert that the agent's JSON output conforms to a JSON Schema.

    This is useful for validating structured agent outputs — e.g., ensuring
    an agent returns a well-formed API response, form, or data object.

    The schema follows JSON Schema draft-07 specification.
    Requires the 'jsonschema' package: pip install jsonschema

    Example:
        schema = {
            "type": "object",
            "required": ["name", "email"],
            "properties": {
                "name": {"type": "string"},
                "email": {"type": "string", "format": "email"},
                "age": {"type": "integer", "minimum": 0},
            },
        }
        assert_output_json_schema(run, schema)
    """
    import json

    try:
        output = run.output.strip()
        # Handle markdown code blocks
        if output.startswith("```"):
            lines = output.split("\n")
            output = "\n".join(lines[1:-1])
        parsed = json.loads(output)
    except json.JSONDecodeError as e:
        raise AssertionError(
            f"Expected JSON output conforming to schema, but got parse error: {e}\n"
            f"Output: {run.output!r}"
        )

    try:
        import jsonschema
        jsonschema.validate(instance=parsed, schema=schema)
    except ImportError:
        raise ImportError(
            "jsonschema is required for assert_output_json_schema. "
            "Install it with: pip install jsonschema"
        )
    except jsonschema.ValidationError as e:
        raise AssertionError(
            f"JSON output does not conform to schema.\n"
            f"Validation error: {e.message}\n"
            f"Path in data: {'.'.join(str(p) for p in e.path)}\n"
            f"Actual output: {json.dumps(parsed, indent=2)[:500]}"
        )
    except jsonschema.SchemaError as e:
        raise AssertionError(f"Invalid JSON Schema: {e.message}\nSchema: {json.dumps(schema, indent=2)[:200]}")


