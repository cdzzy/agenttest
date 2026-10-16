"""Package exports and version consistency tests."""

import re
from pathlib import Path

import agenttest

STRUCTURED_ASSERTIONS = [
    "assert_output_structured",
    "assert_output_is_valid_yaml",
    "assert_output_field_equals",
    "assert_output_contains_fields",
]


def test_structured_assertions_exported_top_level():
    """Structured-output assertions must be importable from the package root."""
    for name in STRUCTURED_ASSERTIONS:
        assert hasattr(agenttest, name), f"{name} missing from agenttest top level"
        assert name in agenttest.__all__, f"{name} missing from agenttest.__all__"


def test_assertions_wildcard_exports_structured():
    """`from agenttest.assertions import *` must expose the structured set."""
    namespace = {}
    exec("from agenttest.assertions import *", namespace)
    for name in STRUCTURED_ASSERTIONS:
        assert name in namespace, f"{name} missing from assertions wildcard export"


def test_package_version_matches_pyproject():
    """__version__ must never drift from the declared package version."""
    pyproject = Path(agenttest.__file__).resolve().parent.parent / "pyproject.toml"
    match = re.search(
        r'^version\s*=\s*"([^"]+)"',
        pyproject.read_text(encoding="utf-8"),
        re.M,
    )
    assert match is not None
    assert agenttest.__version__ == match.group(1)
