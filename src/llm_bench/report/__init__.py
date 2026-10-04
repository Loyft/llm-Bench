"""Report generators."""

from llm_bench.report.html import write_html_report
from llm_bench.report.terminal import render_terminal_report

__all__ = ["write_html_report", "render_terminal_report"]
