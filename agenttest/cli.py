"""
AgentTest CLI — command-line entry point.

Commands:
    agenttest edit --ui       Launch the visual test editor (browser)
    agenttest --version       Print version
"""

from __future__ import annotations

import argparse
import sys


def _open_editor() -> int:
    """Open the visual test editor in the default browser."""
    import webbrowser
    from pathlib import Path

    editor_path = Path(__file__).resolve().parent / "ui" / "editor.html"
    if not editor_path.exists():
        print(f"Error: editor not found at {editor_path}", file=sys.stderr)
        return 1

    url = editor_path.as_uri()
    print(f"Opening AgentTest editor: {url}")
    webbrowser.open(url)
    print("Use the editor to build test cases, then export to Python.")
    return 0


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        prog="agenttest",
        description="AgentTest — the testing framework for AI agents",
    )
    parser.add_argument("--version", action="store_true", help="Print version and exit")
    subparsers = parser.add_subparsers(dest="command")

    edit_parser = subparsers.add_parser("edit", help="Open the visual test editor")
    edit_parser.add_argument("--ui", action="store_true", help="Launch in browser")

    args = parser.parse_args(argv)

    if args.version:
        from agenttest import __version__
        print(f"agenttest {__version__}")
        return 0

    if args.command == "edit":
        return _open_editor()

    parser.print_help()
    return 0


if __name__ == "__main__":
    sys.exit(main())
