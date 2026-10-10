"""Derived tables shared by the terminal and HTML reports."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from llm_bench.schema import BenchReport, SampleResult, TaskSummary

# Breakdown key -> display label, in display order.
CAPABILITY_LABELS: dict[str, str] = {
    "reasoning": "Reasoning",
    "coding": "Coding",
    "knowledge": "Knowledge",
    "instructions": "Instructions",
    "tools": "Tool use",
    "context": "Long context",
    "hallucination": "Honesty",
    "multi_turn": "Multi-turn",
    "consistency": "Consistency",
    "quality": "Sanity",
}
# Compact headers for narrow terminals.
SHORT_LABELS: dict[str, str] = {
    "reasoning": "Reason",
    "coding": "Code",
    "knowledge": "Know",
    "instructions": "Instr",
    "tools": "Tools",
    "context": "Ctx",
    "hallucination": "Honest",
    "multi_turn": "Multi",
    "consistency": "Consist",
    "quality": "Sanity",
}
PERFORMANCE_LABELS: dict[str, str] = {"speed": "Decode/TTFT", "prefill": "Prefill"}

_CATEGORY_LABELS = {
    "depth_10": "needle @10%",
    "depth_50": "needle @50%",
    "depth_90": "needle @90%",
    "ctx_30": "30% of ctx",
    "ctx_70": "70% of ctx",
}


def _num(v: Any) -> float | None:
    return float(v) if isinstance(v, (int, float)) and not isinstance(v, bool) else None


def present_capability_keys(report: BenchReport) -> list[str]:
    keys = {k for sc in report.scores for k in sc.breakdown}
    return [k for k in CAPABILITY_LABELS if k in keys]


def capability_matrix(report: BenchReport) -> dict[str, Any]:
    keys = present_capability_keys(report)
    rows = [
        {
            "model": sc.model,
            "overall": sc.overall,
            "performance": sc.performance,
            "capability": sc.capability,
            "reliability": sc.reliability,
            "values": {k: sc.breakdown.get(k) for k in keys},
        }
        for sc in report.scores
    ]
    return {
        "cols": keys,
        "labels": [CAPABILITY_LABELS[k] for k in keys],
        "short_labels": [SHORT_LABELS[k] for k in keys],
        "rows": rows,
    }


def _metric(block: TaskSummary | None, key: str) -> float | None:
    return _num(block.metrics.get(key)) if block else None


def speed_rows(report: BenchReport) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for sc in report.scores:
        gen = sc.tasks.get("generation_speed")
        lat = sc.tasks.get("prompt_latency")
        thr = sc.tasks.get("throughput")
        if not (gen or lat or thr):
            continue

        c1, c2, c4 = (_metric(thr, f"agg_tps_c{n}") for n in (1, 2, 4))
        out.append(
            {
                "model": sc.model,
                "decode_mean": _metric(gen, "tokens_per_sec"),
                "decode_p50": _metric(gen, "tokens_per_sec_p50"),
                "decode_p95": _metric(gen, "tokens_per_sec_p95"),
                "ttft_mean": _metric(gen, "ttft_ms"),
                "ttft_p50": _metric(gen, "ttft_ms_p50"),
                "ttft_p95": _metric(gen, "ttft_ms_p95"),
                "itl_mean": _metric(gen, "itl_ms_mean"),
                "itl_p95": _metric(gen, "itl_ms_p95"),
                "itl_max": _metric(gen, "itl_ms_max"),
                "prefill": _metric(lat, "prefill_tokens_per_sec")
                or _metric(gen, "prefill_tokens_per_sec"),
                "agg_c1": c1,
                "agg_c2": c2,
                "agg_c4": c4,
                "scaling_c4": (c4 / c1) if c1 and c4 else None,
            }
        )
    return out


def category_rows(report: BenchReport) -> list[dict[str, Any]]:
    """Per (model, task) sub-scores, e.g. reasoning -> arithmetic/logic/pattern."""
    out: list[dict[str, Any]] = []
    for s in report.summaries:
        if s.task == "setup":
            continue
        cats = []
        for k, v in sorted(s.metrics.items()):
            if k.startswith("score_") and not k.endswith(("_std", "_p50", "_p95")):
                name = k[len("score_") :]
                val = _num(v)
                if val is not None:
                    label = _CATEGORY_LABELS.get(name, name.replace("_", " "))
                    cats.append({"name": label, "value": val})
        if cats:
            out.append({"model": s.model, "task": s.task, "categories": cats})
    return out


def _getter(metrics: dict[str, Any]) -> Callable[[str], float | None]:
    def g(key: str) -> float | None:
        return _num(metrics.get(key))

    return g


def task_rows(report: BenchReport) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for s in report.summaries:
        if s.task == "setup":
            continue
        g = _getter(s.metrics)

        score = g("score") if "score" in s.metrics else g("success_rate")
        extras: list[tuple[str, str]] = []
        if (v := g("hallucinated")) is not None:
            extras.append(("hallucination rate", f"{v:.0%}"))
        if (v := g("over_refused")) is not None:
            extras.append(("over-refusal", f"{v:.0%}"))
        if (v := g("native_tool_call")) is not None:
            extras.append(("native tool calls", f"{v:.0%}"))
        if (v := g("test_pass_rate")) is not None:
            extras.append(("unit tests passed", f"{v:.0%}"))
        if (v := g("accuracy")) is not None:
            extras.append(("accuracy", f"{v:.0%}"))
        if (v := g("all_constraints_met")) is not None:
            extras.append(("all constraints met", f"{v:.0%}"))
        if (v := g("prompt_tokens_vs_target")) is not None:
            extras.append(("prompt kept by server", f"{v:.0%}"))
        if (v := g("thinking_chars")) is not None and v > 0:
            extras.append(("thinking chars/item", f"{v:.0f}"))
        out.append(
            {
                "model": s.model,
                "task": s.task,
                "ok": s.samples - s.errors,
                "samples": s.samples,
                "errors": s.errors,
                "score": score,
                "decode_tps": g("tokens_per_sec"),
                "prefill_tps": g("prefill_tokens_per_sec"),
                "ttft_ms": g("ttft_ms"),
                "avg_tokens": g("completion_tokens"),
                "truncated": g("truncated"),
                "extras": extras,
            }
        )
    return out


def health_rows(report: BenchReport) -> list[dict[str, Any]]:
    """Run-quality stats per model: reliability, truncation, verbosity."""
    out: list[dict[str, Any]] = []
    for sc in report.scores:
        toks: list[float] = []
        trunc: list[float] = []
        empty: list[float] = []
        for s in sc.tasks.values():
            if s.task in {"throughput"}:
                continue
            if (v := _num(s.metrics.get("completion_tokens"))) is not None:
                toks.append(v)
            if (v := _num(s.metrics.get("truncated"))) is not None:
                trunc.append(v)
            if (v := _num(s.metrics.get("empty"))) is not None:
                empty.append(v)
        out.append(
            {
                "model": sc.model,
                "reliability": sc.reliability,
                "avg_tokens": sum(toks) / len(toks) if toks else None,
                "truncated": sum(trunc) / len(trunc) if trunc else None,
                "empty": sum(empty) / len(empty) if empty else None,
            }
        )
    return out


def failures(report: BenchReport, limit: int = 40) -> list[dict[str, Any]]:
    """Wrong-answer samples (not transport errors) for drill-down."""
    out: list[dict[str, Any]] = []
    for s in report.samples:
        if s.error or s.task == "setup":
            continue
        score = _num(s.metrics.get("score"))
        if score is None or score >= 0.999:
            continue
        out.append(_failure_row(s, score))
        if len(out) >= limit:
            break
    return out


def _failure_row(s: SampleResult, score: float) -> dict[str, Any]:
    d = s.detail
    ident = d.get("item_id") or d.get("scenario") or f"#{s.run_index}"
    shown = d.get("response") or d.get("text_preview") or d.get("code") or d.get("answers")
    return {
        "model": s.model,
        "task": s.task,
        "item": str(ident),
        "score": score,
        "expected": d.get("expected"),
        "got": str(shown)[:200] if shown is not None else "",
        "note": d.get("tests_passed") or d.get("exec_error") or "",
    }
