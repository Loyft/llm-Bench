"""Jinja2 + Chart.js HTML report writer."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from jinja2 import Environment, PackageLoader, select_autoescape

from llm_bench.report.view import (
    build_insights,
    capability_matrix,
    category_rows,
    failures,
    health_rows,
    speed_rows,
    speed_series,
    task_rows,
)
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
        "performance": sc.performance if sc else None,
        "capability": sc.capability if sc else None,
        "reliability": sc.reliability if sc else 1.0,
    }


def _needle_series(report: BenchReport) -> list[dict[str, Any]]:
    """Long-context recall at 10/50/90% depth, per model."""
    out = []
    for row in category_rows(report):
        if row["task"] != "long_context":
            continue
        by = {c["name"]: c["value"] for c in row["categories"]}
        keys = ["needle @10%", "needle @50%", "needle @90%"]
        if all(k in by for k in keys):
            out.append({"label": row["model"], "data": [by[k] for k in keys]})
    return out


def write_html_report(report: BenchReport, dest: Path) -> Path:
    env = Environment(
        loader=PackageLoader("llm_bench.report", "templates"),
        autoescape=select_autoescape(["html", "xml"]),
    )
    template = env.get_template("report.html.j2")

    labels = [s.model for s in report.scores]
    overall = [s.overall for s in report.scores]
    tok_s, prefill, ttft = speed_series(report)

    matrix = capability_matrix(report)
    radar = [
        {"label": row["model"], "data": [row["values"].get(k) or 0.0 for k in matrix["cols"]]}
        for row in matrix["rows"]
    ]

    speed = speed_rows(report)
    health = health_rows(report)
    scaling = [
        {"label": r["model"], "data": [r["agg_c1"], r["agg_c2"], r["agg_c4"]]}
        for r in speed
        if r["agg_c1"] or r["agg_c2"] or r["agg_c4"]
    ]
    star_map = [
        {
            "label": sc.model,
            "x": tok_s[i],
            "y": sc.capability if sc.capability is not None else sc.overall,
        }
        for i, sc in enumerate(report.scores)
    ]
    composition = [
        {
            "label": sc.model,
            "performance": sc.performance,
            "capability": sc.capability,
            "reliability": sc.reliability,
        }
        for sc in report.scores
    ]
    insights = build_insights(report)

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
        matrix=matrix,
        radar_labels_json=json.dumps(matrix["labels"]),
        radar_json=json.dumps(radar),
        speed=speed,
        insights=insights,
        needle_json=json.dumps(_needle_series(report)),
        scaling_json=json.dumps(scaling),
        star_json=json.dumps(star_map),
        composition_json=json.dumps(composition),
        categories=category_rows(report),
        tasks=task_rows(report),
        health=health,
        failures=failures(report),
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
