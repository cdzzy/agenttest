"""
HTML/JSON test reports for AgentTest (Roadmap).

Turns raw ``List[TestResult]`` output into shareable artifacts: a JSON report
for CI tooling and a self-contained HTML report for humans (summary cards,
per-test table, failure details). Zero dependencies.

Usage::

    from agenttest.report import save_report

    results = runner.run_suite(suite)
    html_path = save_report(results, "report.html")   # or .json
    # save_report returns the path it wrote
"""

from __future__ import annotations

import html
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional, Union

from agenttest.core.case import TestResult, TestStatus

JSON = Union[dict, list, str, int, float, bool, None]


def _status_counts(results: List[TestResult]) -> dict:
    return {
        "passed": sum(1 for r in results if r.status == TestStatus.PASSED),
        "failed": sum(1 for r in results if r.status == TestStatus.FAILED),
        "errors": sum(1 for r in results if r.status == TestStatus.ERROR),
        "skipped": sum(1 for r in results if r.status == TestStatus.SKIPPED),
        "total": len(results),
    }


def build_report(results: List[TestResult]) -> dict:
    """Build a JSON-serializable report dict from a list of TestResult."""
    counts = _status_counts(results)
    return {
        "summary": {
            **counts,
            "pass_rate": round(counts["passed"] / counts["total"] * 100, 1) if counts["total"] else 0.0,
            "duration_ms": round(sum(r.duration_ms for r in results), 1),
            "generated_at": datetime.now(timezone.utc).isoformat(),
        },
        "results": [
            {
                "name": r.test_name,
                "status": r.status.value,
                "duration_ms": round(r.duration_ms, 1),
                "error": r.error_message,
                "assertions": [
                    {"name": a.assertion_name, "passed": a.passed, "message": a.message}
                    for a in r.assertion_results
                ],
            }
            for r in results
        ],
    }


def report_to_json(results: List[TestResult]) -> str:
    """Serialize results as pretty-printed JSON."""
    return json.dumps(build_report(results), indent=2, ensure_ascii=False)


def report_to_html(results: List[TestResult]) -> str:
    """Render a self-contained HTML report (no external assets)."""
    report = build_report(results)
    summary = report["summary"]

    rows = []
    for r in report["results"]:
        badge = {
            "passed": "pass",
            "failed": "fail",
            "error": "error",
            "skipped": "skip",
        }.get(r["status"], "pass")

        error_html = ""
        if r["error"]:
            error_html = (
                f'<details><summary>details</summary><pre class="err">{html.escape(r["error"])}</pre></details>'
            )

        rows.append(
            "<tr>"
            f'<td><span class="badge badge-{badge}">{r["status"]}</span></td>'
            f"<td>{html.escape(r['name'])}</td>"
            f"<td class=\"num\">{r['duration_ms']}ms</td>"
            f"<td>{error_html}</td>"
            "</tr>"
        )

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8" />
<title>AgentTest Report</title>
<style>
  body {{ font-family: -apple-system,'Segoe UI',Roboto,sans-serif; background:#0f1117; color:#e6e6e6; margin:0; padding:24px; }}
  h1 {{ font-size:22px; }} h1 span {{ color:#6ea8fe; }}
  .grid {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(140px,1fr)); gap:12px; margin:18px 0; }}
  .card {{ background:#1a1e29; border:1px solid #2a2e3a; border-radius:8px; padding:14px; text-align:center; }}
  .card b {{ display:block; font-size:24px; }}
  table {{ width:100%; border-collapse:collapse; margin-top:12px; }}
  th,td {{ text-align:left; padding:8px 10px; border-bottom:1px solid #2a2e3a; font-size:14px; }}
  .badge {{ padding:2px 8px; border-radius:9999px; font-size:12px; }}
  .badge-pass {{ background:#064e3b; color:#10b981; }}
  .badge-fail {{ background:#7f1d1d; color:#ef4444; }}
  .badge-error {{ background:#7f1d1d; color:#fbbf24; }}
  .badge-skip {{ background:#1e3a5f; color:#93c5fd; }}
  .num {{ text-align:right; }}
  pre.err {{ background:#0a0c12; padding:10px; border-radius:6px; overflow-x:auto; }}
  details {{ font-size:12px; color:#93c5fd; }}
</style>
</head>
<body>
  <h1>AgentTest <span>Report</span></h1>
  <div class="grid">
    <div class="card"><b>{summary['total']}</b>total</div>
    <div class="card"><b style="color:#10b981">{summary['passed']}</b>passed</div>
    <div class="card"><b style="color:#ef4444">{summary['failed']}</b>failed</div>
    <div class="card"><b style="color:#fbbf24">{summary['errors']}</b>errors</div>
    <div class="card"><b style="color:#93c5fd">{summary['skipped']}</b>skipped</div>
    <div class="card"><b>{summary['pass_rate']}%</b>pass rate</div>
    <div class="card"><b>{summary['duration_ms']}ms</b>total time</div>
  </div>
  <table>
    <thead><tr><th>Status</th><th>Test</th><th>Duration</th><th>Details</th></tr></thead>
    <tbody>{''.join(rows)}</tbody>
  </table>
  <p style="color:#64748b;font-size:12px">Generated {summary['generated_at']}</p>
</body>
</html>
"""


def save_report(
    results: List[TestResult],
    path: str,
    fmt: Optional[str] = None,
) -> str:
    """
    Write results to a file, inferring format from the extension unless ``fmt``
    is given. Returns the absolute path written.

    Supported formats: ``html``, ``json``.
    """
    ext = (fmt or Path(path).suffix.lstrip(".")).lower()
    if ext == "html":
        content = report_to_html(results)
    elif ext == "json":
        content = report_to_json(results)
    else:
        raise ValueError(f"Unsupported report format: {ext} (use .html or .json)")

    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(content, encoding="utf-8")
    return str(out.absolute())