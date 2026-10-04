"""Jinja2 + Chart.js HTML report writer."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from jinja2 import Environment, PackageLoader, select_autoescape

from llm_bench.schema import BenchReport


def _profile_metrics(report: BenchReport) -> dict[str, Any]:
    """Flatten key metrics for single-model raw-value cards."""
    sc = report.scores[0] if report.scores else None
    speed = None
    latency = None
    if sc:
        speed = sc.tasks.get("generation_speed") or sc.tasks.get("speed")
        latency = sc.tasks.get("prompt_latency") or sc.tasks.get("latency")

    def _num(block, key: str) -> float | None:
        if not block:
            return None
        v = block.metrics.get(key)
        return float(v) if isinstance(v, (int, float)) else None

    decode = _num(speed, "tokens_per_sec")
    prefill = _num(speed, "prefill_tokens_per_sec")
    if prefill is None:
        prefill = _num(latency, "prefill_tokens_per_sec")
    ttft = _num(speed, "ttft_ms")
    if ttft is None:
        ttft = _num(latency, "ttft_ms")

    return {
        "decode_tps": decode,
        "prefill_tps": prefill,
        "ttft_ms": ttft,
        "overall": sc.overall if sc else 0.0,
        "quality": sc.breakdown.get("quality", 0.0) if sc else 0.0,
        "tools": sc.breakdown.get("tools", 0.0) if sc else 0.0,
        "context": sc.breakdown.get("context", 0.0) if sc else 0.0,
    }


def write_html_report(report: BenchReport, dest: Path) -> Path:
    env = Environment(
        loader=PackageLoader("llm_bench.report", "templates"),
        autoescape=select_autoescape(["html", "xml"]),
    )
    template = env.get_template("report.html.j2")

    labels = [s.model for s in report.scores]
    tok_s = []
    prefill = []
    ttft = []
    overall = [s.overall for s in report.scores]
    quality = []
    tools = []
    context = []

    for sc in report.scores:
        speed = sc.tasks.get("generation_speed") or sc.tasks.get("speed")
        latency = sc.tasks.get("prompt_latency") or sc.tasks.get("latency")
        tps = None
        pref = None
        tt = None
        src = speed or latency
        if src:
            tps = src.metrics.get("tokens_per_sec")
            pref = src.metrics.get("prefill_tokens_per_sec")
            tt = src.metrics.get("ttft_ms")
        if pref is None and latency:
            pref = latency.metrics.get("prefill_tokens_per_sec")
        tok_s.append(float(tps) if isinstance(tps, (int, float)) else 0.0)
        prefill.append(float(pref) if isinstance(pref, (int, float)) else 0.0)
        ttft.append(float(tt) if isinstance(tt, (int, float)) else 0.0)
        quality.append(sc.breakdown.get("quality", 0.0))
        tools.append(sc.breakdown.get("tools", 0.0))
        context.append(sc.breakdown.get("context", 0.0))

    metric_keys = [
        "tokens_per_sec",
        "prefill_tokens_per_sec",
        "ttft_ms",
        "score",
        "success_rate",
    ]
    heatmap = []
    for s in report.summaries:
        if s.task == "setup":
            continue
        row: dict[str, Any] = {"model": s.model, "task": s.task}
        for k in metric_keys:
            v = s.metrics.get(k)
            row[k] = float(v) if isinstance(v, (int, float)) else None
        heatmap.append(row)

    errors = [
        {"model": s.model, "task": s.task, "error": s.error}
        for s in report.errors
        if s.error
    ]

    mem = report.memory
    mem_t = [s.get("t_s", 0) for s in mem.samples]
    mem_system = [s.get("system_used_mb") for s in mem.samples]
    mem_process = [s.get("process_rss_mb") for s in mem.samples]
    mem_model = [s.get("model_ram_mb") for s in mem.samples]
    mem_kv = [s.get("kv_ram_mb") for s in mem.samples]
    mem_context = [s.get("context_ram_mb") for s in mem.samples]

    html = template.render(
        report=report,
        mode=report.mode,
        title="LLM-BENCH Report",
        ok=report.ok,
        labels_json=json.dumps(labels),
        tok_s_json=json.dumps(tok_s),
        prefill_json=json.dumps(prefill),
        ttft_json=json.dumps(ttft),
        overall_json=json.dumps(overall),
        quality_json=json.dumps(quality),
        tools_json=json.dumps(tools),
        context_json=json.dumps(context),
        heatmap=heatmap,
        scores=report.scores,
        summaries=report.summaries,
        errors=errors,
        manifest=report.manifest,
        memory=mem,
        profile_metrics=_profile_metrics(report),
        mem_t_json=json.dumps(mem_t),
        mem_system_json=json.dumps(mem_system),
        mem_process_json=json.dumps(mem_process),
        mem_model_json=json.dumps(mem_model),
        mem_kv_json=json.dumps(mem_kv),
        mem_context_json=json.dumps(mem_context),
    )

    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(html, encoding="utf-8")
    return dest
