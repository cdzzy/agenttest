"""
Test JSON Schema Assertion — validate structured agent outputs.

agenttest can validate that an agent's JSON output conforms to a JSON Schema,
which is essential for structured outputs like API responses, forms, or data objects.
"""

from agenttest import AgentTestCase, AgentTestRunner


def structured_agent(input_text: str) -> dict:
    """
    An agent that returns structured JSON data instead of plain text.
    Useful for tools, API integrations, and data pipelines.
    """
    import json

    if "contact" in input_text.lower():
        return {
            "output": json.dumps({
                "name": "Jane Doe",
                "email": "jane@example.com",
                "age": 32,
            }),
            "tool_calls": [],
        }
    elif "product" in input_text.lower():
        return {
            "output": json.dumps({
                "product_id": "SKU-12345",
                "price": 29.99,
                "currency": "USD",
                "in_stock": True,
            }),
            "tool_calls": [],
        }
    return {"output": "{}", "tool_calls": []}


class StructuredOutputTests(AgentTestCase):
    agent = structured_agent

    def test_contact_response_schema(self):
        """Verify the agent returns a valid contact object."""
        run = self.invoke("Get contact info for Jane")

        # Validate JSON structure matches expected schema
        schema = {
            "type": "object",
            "required": ["name", "email"],
            "properties": {
                "name": {"type": "string"},
                "email": {"type": "string"},
                "age": {"type": "integer"},
            },
        }

        from agenttest.assertions.output import assert_output_json_schema
        assert_output_json_schema(run, schema)

    def test_product_response_schema(self):
        """Verify the agent returns a valid product object."""
        run = self.invoke("Get product info for SKU-12345")

        schema = {
            "type": "object",
            "required": ["product_id", "price", "currency", "in_stock"],
            "properties": {
                "product_id": {"type": "string"},
                "price": {"type": "number", "minimum": 0},
                "currency": {"type": "string", "enum": ["USD", "EUR", "CNY"]},
                "in_stock": {"type": "boolean"},
            },
        }

        from agenttest.assertions.output import assert_output_json_schema
        assert_output_json_schema(run, schema)

    def test_rejects_invalid_email(self):
        """Invalid email should fail schema validation."""
        run = self.invoke("Get contact with bad email")

        # Patch the agent to return invalid data for this test
        def bad_agent(_):
            import json
            return {
                "output": json.dumps({"name": "Bob", "email": "not-an-email"}),
                "tool_calls": [],
            }

        self.agent = bad_agent
        run = self.invoke("Get contact")

        schema = {
            "type": "object",
            "required": ["name", "email"],
            "properties": {
                "name": {"type": "string"},
                "email": {"type": "string", "format": "email"},
            },
        }

        from agenttest.assertions.output import assert_output_json_schema
        try:
            assert_output_json_schema(run, schema)
            self.fail("Expected schema validation to fail")
        except AssertionError:
            pass  # Expected — invalid email should raise


if __name__ == "__main__":
    import sys
    print("Run with: pytest test_json_schema.py -v")
    print()

    runner = AgentTestRunner()
    suite = runner._discover_tests(StructuredOutputTests)
    runner.run_suite(suite)

