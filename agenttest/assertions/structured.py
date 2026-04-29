"""
Structured output assertions — validate agent outputs against schemas or models.

These assertions answer: "Does the agent's output conform to the expected structure?"
"""
from __future__ import annotations

import json
import re
from typing import Any, Callable, Optional, Type, TypeVar

from agenttest.core.case import AgentRun

T = TypeVar("T")


def assert_output_structured(
    run: AgentRun,
    schema: Optional[dict] = None,
    model: Optional[Type] = None,
    validator: Optional[Callable[[Any], bool]] = None,
):
    """
    Assert that the agent's output conforms to a structured schema or model.

    Three validation modes (mutually exclusive, checked in this order):
    1. schema: JSON Schema (draft-07) validation
    2. model: Pydantic model validation
    3. validator: Custom validation function

    Examples:
        # JSON Schema validation
        assert_output_structured(run, schema={
            "type": "object",
            "required": ["status", "data"],
            "properties": {
                "status": {"type": "string", "enum": ["ok", "error"]},
                "data": {"type": "object"},
            },
        })

        # Pydantic model validation
        from pydantic import BaseModel
        class Response(BaseModel):
            status: str
            data: dict

        assert_output_structured(run, model=Response)

        # Custom validator
        assert_output_structured(run, validator=lambda x: x.get("code") == 200)
    """
    # Parse JSON from output
    try:
        output = run.output.strip()
        # Handle markdown code blocks
        if output.startswith("```"):
            lines = output.split("\n")
            output = "\n".join(lines[1:-1])
            # Also remove language identifier if present
            if lines[0].startswith("```"):
                first_line = lines[0].replace("```", "").strip()
                if first_line and not first_line.startswith("json"):
                    output = first_line + "\n" + output
        parsed = json.loads(output)
    except json.JSONDecodeError as e:
        raise AssertionError(
            f"Expected structured output (JSON), but got parse error: {e}\n"
            f"Output: {run.output!r}"
        )

    # Validate based on mode
    if schema is not None:
        _validate_json_schema(run, parsed, schema)
    elif model is not None:
        _validate_pydantic(run, parsed, model)
    elif validator is not None:
        _validate_custom(run, parsed, validator)
    else:
        raise ValueError(
            "assert_output_structured requires one of: schema, model, or validator"
        )


def _validate_json_schema(run: AgentRun, parsed: Any, schema: dict):
    """Validate parsed output against JSON Schema."""
    try:
        import jsonschema
        jsonschema.validate(instance=parsed, schema=schema)
    except ImportError:
        raise ImportError(
            "jsonschema is required for schema validation. "
            "Install it with: pip install jsonschema"
        )
    except jsonschema.ValidationError as e:
        raise AssertionError(
            f"Output does not conform to JSON Schema.\n"
            f"Validation error: {e.message}\n"
            f"Path in data: {'.'.join(str(p) for p in e.path)}\n"
            f"Actual output: {json.dumps(parsed, indent=2)[:500]}"
        )
    except jsonschema.SchemaError as e:
        raise AssertionError(f"Invalid JSON Schema: {e.message}")


def _validate_pydantic(run: AgentRun, parsed: Any, model: Type[T]) -> T:
    """Validate parsed output against a Pydantic model."""
    try:
        from pydantic import BaseModel, ValidationError
        if not isinstance(model, type) or not issubclass(model, BaseModel):
            raise TypeError(f"Expected a Pydantic BaseModel, got {model}")
        
        try:
            return model.model_validate(parsed)
        except ValidationError as e:
            raise AssertionError(
                f"Output does not conform to Pydantic model {model.__name__}.\n"
                f"Validation errors:\n{e}\n"
                f"Actual output: {json.dumps(parsed, indent=2)[:500]}"
            )
    except ImportError:
        raise ImportError(
            "pydantic is required for model validation. "
            "Install it with: pip install pydantic"
        )


def _validate_custom(run: AgentRun, parsed: Any, validator: Callable[[Any], bool]):
    """Validate using a custom function."""
    try:
        result = validator(parsed)
        if not result:
            raise AssertionError(
                f"Custom validation failed.\n"
                f"Validator function returned: {result}\n"
                f"Output: {json.dumps(parsed, indent=2)[:500]}"
            )
    except AssertionError:
        raise
    except Exception as e:
        raise AssertionError(
            f"Custom validation raised an exception: {e}\n"
            f"Output: {json.dumps(parsed, indent=2)[:500]}"
        )


def assert_output_is_valid_yaml(run: AgentRun):
    """
    Assert that the agent's output is valid YAML.

    Example:
        assert_output_is_valid_yaml(run)
    """
    try:
        import yaml
        output = run.output.strip()
        # Handle markdown code blocks
        if output.startswith("```"):
            lines = output.split("\n")
            output = "\n".join(lines[1:-1])
        yaml.safe_load(output)
    except ImportError:
        raise ImportError(
            "PyYAML is required for YAML validation. "
            "Install it with: pip install pyyaml"
        )
    except yaml.YAMLError as e:
        raise AssertionError(
            f"Expected valid YAML output, but got parse error: {e}\n"
            f"Output: {run.output!r}"
        )


def assert_output_field_equals(run: AgentRun, field_path: str, expected: Any):
    """
    Assert that a specific field in JSON/YAML output equals the expected value.

    field_path uses dot notation for nested fields: "user.name" or "data.items[0].id"

    Example:
        assert_output_field_equals(run, "status", "success")
        assert_output_field_equals(run, "user.email", "test@example.com")
    """
    import json
    
    try:
        output = run.output.strip()
        if output.startswith("```"):
            lines = output.split("\n")
            output = "\n".join(lines[1:-1])
        parsed = json.loads(output)
    except json.JSONDecodeError:
        # Try YAML
        try:
            import yaml
            output = run.output.strip()
            if output.startswith("```"):
                lines = output.split("\n")
                output = "\n".join(lines[1:-1])
            parsed = yaml.safe_load(output)
        except yaml.YAMLError:
            raise AssertionError(
                f"Expected JSON or YAML output for field access.\n"
                f"Output: {run.output!r}"
            )

    # Navigate to the field
    current = parsed
    parts = re.split(r'\[(\d+)\]|\.', field_path)
    
    for part in parts:
        if not part:
            continue
        if isinstance(current, dict):
            current = current.get(part)
        elif isinstance(current, list):
            try:
                idx = int(part)
                current = current[idx]
            except (ValueError, IndexError):
                current = None
        else:
            current = None
        
        if current is None:
            raise AssertionError(
                f"Field '{field_path}' not found in output.\n"
                f"Output: {json.dumps(parsed, indent=2)[:500]}"
            )

    if current != expected:
        raise AssertionError(
            f"Field '{field_path}' has unexpected value.\n"
            f"Expected: {expected!r}\n"
            f"Actual: {current!r}"
        )


def assert_output_contains_fields(run: AgentRun, required_fields: list[str]):
    """
    Assert that the output contains all the specified fields at the top level.

    Works with JSON objects. Field names are case-sensitive.

    Example:
        assert_output_contains_fields(run, ["id", "name", "email"])
    """
    import json

    try:
        output = run.output.strip()
        if output.startswith("```"):
            lines = output.split("\n")
            output = "\n".join(lines[1:-1])
        parsed = json.loads(output)
    except json.JSONDecodeError:
        raise AssertionError(
            f"Expected JSON output for field validation.\n"
            f"Output: {run.output!r}"
        )

    if not isinstance(parsed, dict):
        raise AssertionError(
            f"Expected JSON object, got {type(parsed).__name__}.\n"
            f"Output: {json.dumps(parsed, indent=2)[:200]}"
        )

    missing = [f for f in required_fields if f not in parsed]
    if missing:
        raise AssertionError(
            f"Output missing required field(s): {missing}\n"
            f"Present fields: {list(parsed.keys())}\n"
            f"Output: {json.dumps(parsed, indent=2)[:500]}"
        )

