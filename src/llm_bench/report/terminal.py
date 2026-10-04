"""Rich terminal ranking / profile report."""

from __future__ import annotations

from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from llm_bench.metrics import sparkline
from llm_bench.schema import BenchReport, SampleResult
from llm_bench.ui.theme import console

TROPHY = r"""
      ___________
     '._==_==_=_.'
     .-\:      /-.
    | (|:.     |) |
     '-|:.     |-'
       \::.    /
        '::. .'
          ) (
        _.' '._
       `"""""""`
"""


def _latency_series(samples: list[SampleResult], model: str, key: str = "total_ms") -> list[float]:
    vals: list[float] = []
    for s in samples:
        if s.model != model or s.error:
            continue
        v = s.metrics.get(key)
        if isinstance(v, (int, float)):
            vals.append(float(v))
    return vals


def render_terminal_report(report: BenchReport) -> None:
    console.print()

    if report.errors:
        err_lines = Text()
        seen: set[str] = set()
        for s in report.errors:
            key = f"{s.model}|{s.task}|{s.error}"
            if key in seen:
                continue
            seen.add(key)
            err_lines.append(f"  [{s.model}] {s.task}: ", style="keygen.label")
            err_lines.append(f"{s.error}\n", style="keygen.fail")
        console.print(
            Panel(
                err_lines,
                title="[keygen.fail]ERRORS[/]",
                border_style="red",
            )
        )
        if not report.ok:
            console.print(
                Panel(
                    Text(
                        "No successful benchmark samples.\n"
                        "Fix the errors above (e.g. start the provider), then re-run.",
                        style="keygen.amber",
                    ),
                    border_style="red",
                    title="[keygen.fail]EMPTY REPORT[/]",
                )
            )
            _print_footer(report)
            return

    m = report.manifest
    spec = Table.grid(padding=(0, 2))
    spec.add_column(style="keygen.dim")
    spec.add_column(style="keygen.amber")
    spec.add_row("models", ", ".join(m.models))
    spec.add_row("provider", m.provider.value)
    spec.add_row("tasks", ", ".join(m.tasks))
    spec.add_row("context", f"{m.context} tokens")
    spec.add_row("tools", "on" if m.tools else "off")
    spec.add_row("mode", f"{m.mode} · runs {m.runs} · warmup {m.warmup}")
    console.print(Panel(spec, title="[keygen.label]TEST SPECS[/]", border_style="keygen.border"))

    if report.mode == "compare":
        console.print(
            Panel(
                Text(TROPHY, style="keygen.amber")
                + Text("\n  UNLOCKED — RANKING BOARD", style="keygen.title"),
                border_style="keygen.border",
            )
        )
    else:
        console.print(
            Panel(
                Text("  PROFILE MODE — SINGLE TARGET ANALYSIS", style="keygen.title"),
                border_style="keygen.border",
            )
        )

    # Overall scores
    score_table = Table(
        title="Overall",
        border_style="keygen.border",
        header_style="keygen.accent",
    )
    score_table.add_column("Rank", style="keygen.amber")
    score_table.add_column("Model", style="keygen.label")
    score_table.add_column("Overall")
    score_table.add_column("Speed")
    score_table.add_column("Quality")
    score_table.add_column("Tools")
    score_table.add_column("Context")

    for i, sc in enumerate(report.scores, start=1):
        rank = "★1" if i == 1 and report.mode == "compare" else str(i)
        score_table.add_row(
            rank,
            sc.model,
            f"{sc.overall:.3f}",
            f"{sc.breakdown.get('speed', 0):.3f}",
            f"{sc.breakdown.get('quality', 0):.3f}",
            f"{sc.breakdown.get('tools', 0):.3f}",
            f"{sc.breakdown.get('context', 0):.3f}",
        )
    console.print(score_table)

    # Per-task metrics
    metrics_table = Table(
        title="Task Metrics (means)",
        border_style="magenta",
        header_style="keygen.accent",
    )
    metrics_table.add_column("Model")
    metrics_table.add_column("Task")
    metrics_table.add_column("Decode (tok/s)")
    metrics_table.add_column("Prefill (tok/s)")
    metrics_table.add_column("TTFT (ms)")
    metrics_table.add_column("Score (0–1)")
    metrics_table.add_column("Samples")

    for s in report.summaries:
        if s.task == "setup":
            continue
        metrics_table.add_row(
            s.model,
            s.task,
            _fmt_unit(s.metrics.get("tokens_per_sec"), "tok/s"),
            _fmt_unit(s.metrics.get("prefill_tokens_per_sec"), "tok/s"),
            _fmt_unit(s.metrics.get("ttft_ms"), "ms"),
            _fmt(s.metrics.get("score") if "score" in s.metrics else s.metrics.get("success_rate")),
            f"{s.samples - s.errors}/{s.samples}",
        )
    console.print(metrics_table)

    # Sparklines
    spark_table = Table(title="Latency Sparklines", border_style="keygen.border")
    spark_table.add_column("Model")
    spark_table.add_column("total_ms")
    for sc in report.scores:
        series = _latency_series(report.samples, sc.model)
        spark_table.add_row(sc.model, sparkline(series) or "—")
    console.print(spark_table)

    mem = report.memory
    if mem.samples:
        mem_table = Table(title="Memory Peaks (MB)", border_style="keygen.border")
        mem_table.add_column("Metric", style="keygen.label")
        mem_table.add_column("Peak MB", style="keygen.amber")
        mem_table.add_row("System RAM used", _fmt(mem.peak_system_used_mb))
        mem_table.add_row("Provider process RSS", _fmt(mem.peak_process_rss_mb))
        mem_table.add_row("Model weights RAM", _fmt(mem.peak_model_ram_mb))
        mem_table.add_row("Model VRAM", _fmt(mem.peak_model_vram_mb))
        mem_table.add_row("KV-cache RAM (est.)", _fmt(mem.peak_kv_ram_mb))
        mem_table.add_row("Context RAM (est.)", _fmt(mem.peak_context_ram_mb))
        console.print(mem_table)

        sys_series = [
            float(s["system_used_mb"])
            for s in mem.samples
            if isinstance(s.get("system_used_mb"), (int, float))
        ]
        kv_series = [
            float(s["kv_ram_mb"])
            for s in mem.samples
            if isinstance(s.get("kv_ram_mb"), (int, float))
        ]
        console.print(
            f"[keygen.dim]RAM over time[/]  system {sparkline(sys_series) or '—'}   "
            f"kv {sparkline(kv_series) or '—'}"
        )

    _print_footer(report)


def _print_footer(report: BenchReport) -> None:
    h = report.manifest.host
    console.print(
        f"\n[keygen.dim]host:[/] {h.os} | {h.cpu} | {h.ram_gb} GB RAM"
        + (f" | GPU: {h.gpu}" if h.gpu else "")
    )
    console.print(
        f"[keygen.dim]run:[/] {report.manifest.run_id} | "
        f"models={report.manifest.model_count} | mode={report.mode}"
    )


def _fmt(v: object) -> str:
    if v is None:
        return "—"
    if isinstance(v, float):
        return f"{v:.2f}"
    return str(v)


def _fmt_unit(v: object, unit: str) -> str:
    if v is None:
        return "—"
    if isinstance(v, float):
        if unit == "ms":
            return f"{v:.1f} {unit}"
        return f"{v:.2f} {unit}"
    return f"{v} {unit}"
