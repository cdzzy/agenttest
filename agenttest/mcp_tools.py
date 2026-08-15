"""
MCP Server testing toolkit (Issue #7).

Verify that an MCP server's tools respond correctly and conform to the MCP
spec, without a full integration harness. Works with any MCP server exposing
the stdio transport.

Usage::

    from agenttest.mcp_tools import MCPServerClient, assert_tool_exists

    with MCPServerClient(command="npx", args=["-y", "@anthropic/mcp-server-brave"],
                         env={"BRAVE_API_KEY": "test-key"}) as server:
        tools = server.list_tools()
        assert_tool_exists(tools, "brave_web_search")
        result = server.call_tool("brave_web_search", {"query": "AI agents 2026"})
        assert_tool_response_valid(result)
"""

from __future__ import annotations

import json
import os
import subprocess
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class MCPTool:
    """Metadata for a single MCP tool."""
    name: str
    description: str = ""
    input_schema: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "MCPTool":
        return cls(
            name=d.get("name", ""),
            description=d.get("description", ""),
            input_schema=d.get("inputSchema", {}),
        )


class MCPServerError(Exception):
    """Raised when an MCP server returns an error."""


def _rpc(method: str, params: Dict[str, Any], request_id: int = 1) -> Dict[str, Any]:
    return {"jsonrpc": "2.0", "id": request_id, "method": method, "params": params}


class MCPServerClient:
    """
    A minimal MCP stdio client for testing servers.

    Spawns a server subprocess and speaks newline-delimited JSON-RPC 2.0.

    Args:
        command: Executable to launch (e.g. ``"npx"``).
        args: Command-line arguments.
        env: Environment variables (merged over ``os.environ``).
        cwd: Working directory.
    """

    def __init__(
        self,
        command: str,
        args: Optional[List[str]] = None,
        env: Optional[Dict[str, str]] = None,
        cwd: Optional[str] = None,
    ) -> None:
        self.command = command
        self.args = args or []
        self.env = {**os.environ, **(env or {})}
        self.cwd = cwd
        self._process: Optional[subprocess.Popen] = None
        self._request_id = 0

    # ── Lifecycle ─────────────────────────────────────────────────────────

    def start(self) -> "MCPServerClient":
        self._process = subprocess.Popen(
            [self.command, *self.args],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env=self.env,
            cwd=self.cwd,
            text=True,
            encoding="utf-8",
        )
        self._send("initialize", {
            "protocolVersion": "2024-11-05",
            "capabilities": {},
            "clientInfo": {"name": "agenttest", "version": "0.1.0"},
        })
        return self

    def stop(self) -> None:
        if self._process and self._process.stdin:
            try:
                self._process.stdin.close()
            except Exception:
                pass
        if self._process:
            self._process.terminate()
            try:
                self._process.wait(timeout=5)
            except Exception:
                self._process.kill()
            self._process = None

    def __enter__(self) -> "MCPServerClient":
        return self.start()

    def __exit__(self, *exc) -> None:
        self.stop()

    # ── JSON-RPC ──────────────────────────────────────────────────────────

    def _send(self, method: str, params: Dict[str, Any]) -> Dict[str, Any]:
        if self._process is None or self._process.stdin is None:
            raise MCPServerError("MCP server is not started")
        self._request_id += 1
        request = _rpc(method, params, self._request_id)
        self._process.stdin.write(json.dumps(request) + "\n")
        self._process.stdin.flush()

        line = self._process.stdout.readline() if self._process.stdout else ""
        if not line:
            raise MCPServerError("MCP server closed the connection")
        response = json.loads(line)
        if "error" in response:
            raise MCPServerError(f"MCP error: {response['error']}")
        return response.get("result", {})

    # ── MCP operations ────────────────────────────────────────────────────

    def list_tools(self) -> List[MCPTool]:
        """Return the tools exposed by the server."""
        result = self._send("tools/list", {})
        tools = result.get("tools", [])
        return [MCPTool.from_dict(t) for t in tools]

    def call_tool(self, name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """Invoke a tool and return its raw result."""
        result = self._send("tools/call", {"name": name, "arguments": arguments})
        if result.get("isError"):
            content = result.get("content", [])
            raise MCPServerError(f"Tool '{name}' returned an error: {content}")
        return result

    def with_env(self, env: Dict[str, str]) -> "MCPServerClient":
        """Return a new client with extra/replaced environment variables."""
        merged = {**self.env, **env}
        return MCPServerClient(self.command, self.args, merged, self.cwd)


# ── Assertion helpers ────────────────────────────────────────────────────────

def assert_tool_exists(tools: List[MCPTool], name: str) -> None:
    """Assert that a tool with the given name is exposed."""
    if not any(t.name == name for t in tools):
        raise AssertionError(
            f"Tool '{name}' not found. Available: {[t.name for t in tools]}"
        )


def assert_tool_response_valid(result: Dict[str, Any]) -> None:
    """Assert that a tool call returned a valid (non-error) response."""
    if result.get("isError"):
        raise AssertionError(f"Tool returned an error: {result.get('content')}")
    content = result.get("content")
    if not content:
        raise AssertionError(f"Tool returned no content: {result}")


def assert_response_contains(result: Dict[str, Any], expected_types: List[str]) -> None:
    """Assert the response content includes the expected content types."""
    content = result.get("content", [])
    types = {item.get("type") for item in content if isinstance(item, dict)}
    for expected in expected_types:
        if expected not in types:
            raise AssertionError(
                f"Expected content type '{expected}' not in response types {types}"
            )
