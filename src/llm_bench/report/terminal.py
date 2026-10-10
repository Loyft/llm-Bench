"""Rich terminal ranking / profile report."""

from __future__ import annotations

from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from llm_bench.metrics import sparkline
from llm_bench.report.view import capability_matrix, category_rows, speed_rows, task_rows
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
    matrix = capability_matrix(report)
    score_table = Table(
        title="Overall",
        border_style="keygen.border",
        header_style="keygen.accent",
    )
    score_table.add_column("Rank", style="keygen.amber")
    score_table.add_column("Model", style="keygen.label")
    score_table.add_column("Overall")
    score_table.add_column("Perf")
    score_table.add_column("Capability")
    score_table.add_column("Reliab.")
    for i, row in enumerate(matrix["rows"], start=1):
        rank = "★1" if i == 1 and report.mode == "compare" else str(i)
        score_table.add_row(
            rank,
            row["model"],
            f"{row['overall']:.3f}",
            _fmt3(row["performance"]),
            _fmt3(row["capability"]),
            f"{row['reliability']:.0%}",
        )
    console.print(score_table)

    if matrix["cols"]:
        cap_table = Table(
            title="Capability breakdown (0–1, higher is better)",
            border_style="keygen.border",
            header_style="keygen.accent",
        )
        cap_table.add_column("Model", style="keygen.label", no_wrap=True)
        for label in matrix["short_labels"]:
            cap_table.add_column(label, justify="right", no_wrap=True)
        for row in matrix["rows"]:
            cap_table.add_row(
                row["model"], *[_heat(row["values"].get(k)) for k in matrix["cols"]]
            )
        console.print(cap_table)

    cats = category_rows(report)
    if cats:
        cat_table = Table(
            title="Sub-scores by category",
            border_style="keygen.border",
            header_style="keygen.accent",
        )
        cat_table.add_column("Model", style="keygen.label")
        cat_table.add_column("Task", style="keygen.amber")
        cat_table.add_column("Categories")
        for row in cats:
            cat_table.add_row(
                row["model"],
                row["task"],
                "  ".join(f"{c['name']} {_heat(c['value'])}" for c in row["categories"]),
            )
        console.print(cat_table)

    speed = speed_rows(report)
    if speed:
        sp = Table(
            title="Speed detail",
            border_style="keygen.border",
            header_style="keygen.accent",
        )
        sp.add_column("Model", style="keygen.label")
        sp.add_column("Decode tok/s\nmean · p50 · p95", justify="right")
        sp.add_column("TTFT ms\nmean · p50 · p95", justify="right")
        sp.add_column("ITL ms\nmean · p95 · max", justify="right")
        sp.add_column("Prefill\ntok/s", justify="right")
        sp.add_column("Concurrent agg tok/s\n×1 · ×2 · ×4 (scaling)", justify="right")
        for r in speed:
            conc = " · ".join(_fmt1(r[k]) for k in ("agg_c1", "agg_c2", "agg_c4"))
            if r["scaling_c4"]:
                conc += f" ({r['scaling_c4']:.1f}x)"
            sp.add_row(
                r["model"],
                " · ".join(_fmt1(r[k]) for k in ("decode_mean", "decode_p50", "decode_p95")),
                " · ".join(_fmt1(r[k]) for k in ("ttft_mean", "ttft_p50", "ttft_p95")),
                " · ".join(_fmt1(r[k]) for k in ("itl_mean", "itl_p95", "itl_max")),
                _fmt1(r["prefill"]),
                conc if any(r[k] for k in ("agg_c1", "agg_c2", "agg_c4")) else "—",
            )
        console.print(sp)

    # Per-task metrics
    metrics_table = Table(
        title="Task Metrics (means)",
        border_style="keygen.border",
        header_style="keygen.accent",
    )
    metrics_table.add_column("Model")
    metrics_table.add_column("Task")
    metrics_table.add_column("Score (0–1)", justify="right")
    metrics_table.add_column("OK/Total", justify="right")
    metrics_table.add_column("Avg tokens", justify="right")
    metrics_table.add_column("Truncated", justify="right")
    metrics_table.add_column("Notes")

    for r in task_rows(report):
        metrics_table.add_row(
            r["model"],
            r["task"],
            _heat(r["score"]),
            f"{r['ok']}/{r['samples']}",
            _fmt1(r["avg_tokens"]),
            f"{r['truncated']:.0%}" if r["truncated"] is not None else "—",
            ", ".join(f"{k} {v}" for k, v in r["extras"]) or "",
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


def _fmt1(v: object) -> str:
    return "—" if v is None else f"{float(v):.1f}"  # type: ignore[arg-type]


def _fmt3(v: object) -> str:
    return "—" if v is None else f"{float(v):.3f}"  # type: ignore[arg-type]


def _heat(v: object) -> str:
    """Score cell coloured by band (green >= .8, amber >= .5, red below)."""
    if v is None:
        return "—"
    f = float(v)  # type: ignore[arg-type]
    colour = "keygen.ok" if f >= 0.8 else "keygen.amber" if f >= 0.5 else "keygen.fail"
    return f"[{colour}]{f:.2f}[/]"


def _fmt_unit(v: object, unit: str) -> str:
    if v is None:
        return "—"
    if isinstance(v, float):
        if unit == "ms":
            return f"{v:.1f} {unit}"
        return f"{v:.2f} {unit}"
    return f"{v} {unit}"
