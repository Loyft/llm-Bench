"""Rich terminal ranking / profile report."""

from __future__ import annotations

from rich import box
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from llm_bench.metrics import sparkline
from llm_bench.report.view import (
    build_insights,
    capability_matrix,
    category_rows,
    speed_rows,
    speed_series,
    task_rows,
)
from llm_bench.schema import BenchReport, SampleResult
from llm_bench.ui.theme import CORAL, MODEL_COLORS, YELLOW, console

# Retro-future table look: double-ruled head, no inner grid.
_BOX = box.MINIMAL_DOUBLE_HEAD

ROCKET = r"""
      /\
     /  \
    / () \
   |  __  |
   | |__| |
  /|      |\
  /_|_/\_|_\
     /||\
    ' || '
"""

_TAG_STYLE = {
    "TOP": YELLOW, "SCORE": YELLOW, "SPEED": "#7fe0c3", "LATENCY": "#7fe0c3",
    "TRADE-OFF": "#9bb0ff", "SCALING": "#9bb0ff", "PROFILE": "#ff8fb8", "WARN": "#ff6b6b",
}


def _title(text: str) -> str:
    """Panel / table title framed like the HTML report's sector headings."""
    return f"[{CORAL}]╣[/] [bold {YELLOW}]{text}[/] [{CORAL}]╠[/]"


def _panel(body: object, title: str, **kw: object) -> Panel:
    return Panel(body, title=_title(title), box=box.DOUBLE, border_style="keygen.border", **kw)  # type: ignore[arg-type]


def _table(title: str, **kw: object) -> Table:
    return Table(
        title=_title(title),
        box=_BOX,
        border_style="keygen.border",
        header_style="keygen.accent",
        **kw,  # type: ignore[arg-type]
    )


def _model_colour(i: int) -> str:
    return MODEL_COLORS[i % len(MODEL_COLORS)]


def _bars(title: str, labels: list[str], values: list[float], fmt: str, top: float | None = None) -> Table:
    """ASCII bar chart: dim ░ track, model-coloured █ fill."""
    width = 34
    peak = top or max(values + [1e-9])
    grid = Table.grid(padding=(0, 1))
    grid.add_column(no_wrap=True, max_width=24, overflow="ellipsis")
    grid.add_column(no_wrap=True)
    grid.add_column(justify="right", no_wrap=True)
    for i, (lbl, v) in enumerate(zip(labels, values, strict=True)):
        n = max(0, min(width, round(v / peak * width)))
        bar = Text()
        bar.append("█" * n, _model_colour(i))
        bar.append("░" * (width - n), "keygen.dim")
        grid.add_row(Text(lbl, style=_model_colour(i)), bar, Text(fmt.format(v), style="keygen.accent"))
    return _panel(grid, title, padding=(0, 2))


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
                title="[keygen.fail]╣ ERRORS ╠[/]",
                box=box.DOUBLE,
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
                    box=box.DOUBLE,
                    border_style="red",
                    title="[keygen.fail]╣ EMPTY REPORT ╠[/]",
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
    console.print(_panel(spec, "Test specifications", padding=(0, 2)))

    if report.mode == "compare":
        console.print(
            _panel(
                Text(ROCKET, style=YELLOW) + Text("\n  FLAGSHIP DETERMINED — LEADERBOARD", style="keygen.title"),
                "Mission result",
            )
        )
    else:
        console.print(
            _panel(Text("  PROFILE MODE — SINGLE TARGET SURVEY", style="keygen.title"), "Mission result")
        )

    insights = build_insights(report)
    if insights:
        deb = Table.grid(padding=(0, 2))
        deb.add_column(no_wrap=True)
        deb.add_column()
        for i in insights:
            deb.add_row(Text(f"[{i['tag']}]", style=f"bold {_TAG_STYLE.get(i['tag'], YELLOW)}"), i["text"])
        console.print(_panel(deb, "Mission debrief", padding=(0, 2)))

    # Overall scores
    matrix = capability_matrix(report)
    score_table = _table("Overall")
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
            Text(row["model"], style=_model_colour(i - 1)),
            f"{row['overall']:.3f}",
            _fmt3(row["performance"]),
            _fmt3(row["capability"]),
            f"{row['reliability']:.0%}",
        )
    console.print(score_table)

    labels = [sc.model for sc in report.scores]
    console.print(_bars("Overall score (0–1)", labels, [sc.overall for sc in report.scores], "{:.3f}", 1.0))
    tok_s, prefill, ttft = speed_series(report)
    if any(tok_s):
        console.print(_bars("Decode tok/s — higher is better", labels, tok_s, "{:.1f}"))
    if any(prefill) and len(labels) > 1:
        console.print(_bars("Prefill tok/s — higher is better", labels, prefill, "{:.0f}"))
    if any(ttft):
        console.print(_bars("Time to first token (ms) — lower is better", labels, ttft, "{:.0f}"))

    if matrix["cols"]:
        cap_table = _table("Capability breakdown (0–1, higher is better)")
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
        cat_table = _table("Sub-scores by category")
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
        sp = _table("Speed detail")
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
    metrics_table = _table("Task Metrics (means)")
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
    spark_table = _table("Latency sparklines")
    spark_table.add_column("Model")
    spark_table.add_column("total_ms")
    for sc in report.scores:
        series = _latency_series(report.samples, sc.model)
        spark_table.add_row(sc.model, sparkline(series) or "—")
    console.print(spark_table)

    mem = report.memory
    if mem.samples:
        mem_table = _table("Memory peaks (MB)")
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
    console.print(f"\n[{CORAL}]{'═' * 20} END OF TRANSMISSION {'═' * 20}[/]")
    console.print(
        f"[keygen.dim]host:[/] {h.os} | {h.cpu} | {h.ram_gb} GB RAM"
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
    """Score cell: shade glyph + value, coloured by band (mint >= .8, gold >= .5, red below)."""
    if v is None:
        return "—"
    f = float(v)  # type: ignore[arg-type]
    colour = "keygen.ok" if f >= 0.8 else "keygen.title" if f >= 0.5 else "keygen.fail"
    glyph = "█" if f >= 0.9 else "▓" if f >= 0.7 else "▒" if f >= 0.5 else "░" if f >= 0.25 else "·"
    return f"[{colour}]{glyph * 2} {f:.2f}[/]"


def _fmt_unit(v: object, unit: str) -> str:
    if v is None:
        return "—"
    if isinstance(v, float):
        if unit == "ms":
            return f"{v:.1f} {unit}"
        return f"{v:.2f} {unit}"
    return f"{v} {unit}"
