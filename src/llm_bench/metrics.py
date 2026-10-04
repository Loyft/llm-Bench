"""Metric helpers and overall scoring."""

from __future__ import annotations

import statistics
from collections import defaultdict
from typing import Any

from llm_bench.schema import ModelScore, SampleResult, TaskSummary

# Weights for overall score (configurable conceptually; fixed defaults for v0.1)
DEFAULT_WEIGHTS = {
    "speed": 0.40,
    "quality": 0.35,
    "tools": 0.15,
    "context": 0.10,
    "latency": 0.0,  # folded into speed presentation; not double-counted
}


def safe_mean(values: list[float]) -> float | None:
    return statistics.mean(values) if values else None


def safe_stdev(values: list[float]) -> float | None:
    if len(values) < 2:
        return 0.0 if values else None
    return statistics.pstdev(values)


def tokens_per_sec(completion_tokens: int | None, total_ms: float | None, ttft_ms: float | None) -> float | None:
    if not completion_tokens or not total_ms:
        return None
    decode_ms = total_ms - (ttft_ms or 0.0)
    if decode_ms <= 0:
        decode_ms = total_ms
    return (completion_tokens / decode_ms) * 1000.0


def aggregate_samples(samples: list[SampleResult]) -> list[TaskSummary]:
    grouped: dict[tuple[str, str, str], list[SampleResult]] = defaultdict(list)
    for s in samples:
        grouped[(s.task, s.model, s.provider)].append(s)

    summaries: list[TaskSummary] = []
    for (task, model, provider), items in grouped.items():
        numeric: dict[str, list[float]] = defaultdict(list)
        errors = 0
        for item in items:
            if item.error:
                errors += 1
                continue
            for k, v in item.metrics.items():
                if isinstance(v, bool):
                    numeric[k].append(1.0 if v else 0.0)
                elif isinstance(v, (int, float)) and v is not None:
                    numeric[k].append(float(v))

        metrics: dict[str, float | int | str | bool | None] = {}
        for k, vals in numeric.items():
            mean = safe_mean(vals)
            if mean is not None:
                metrics[k] = round(mean, 4)
                std = safe_stdev(vals)
                if std is not None:
                    metrics[f"{k}_std"] = round(std, 4)

        summaries.append(
            TaskSummary(
                task=task,
                model=model,
                provider=provider,
                metrics=metrics,
                samples=len(items),
                errors=errors,
            )
        )
    return summaries


def _norm_higher_better(value: float | None, peers: list[float]) -> float:
    if value is None or not peers:
        return 0.0
    lo, hi = min(peers), max(peers)
    if hi <= lo:
        return 1.0
    return (value - lo) / (hi - lo)


def _norm_lower_better(value: float | None, peers: list[float]) -> float:
    if value is None or not peers:
        return 0.0
    lo, hi = min(peers), max(peers)
    if hi <= lo:
        return 1.0
    return (hi - value) / (hi - lo)


def score_models(summaries: list[TaskSummary]) -> list[ModelScore]:
    by_model: dict[tuple[str, str], dict[str, TaskSummary]] = defaultdict(dict)
    for s in summaries:
        by_model[(s.model, s.provider)][s.task] = s

    # Collect peer metrics for relative scoring
    tok_s: list[float] = []
    ttft: list[float] = []
    quality: list[float] = []
    tools: list[float] = []
    context: list[float] = []

    for tasks in by_model.values():
        speed = tasks.get("speed") or tasks.get("generation_speed")
        if speed:
            v = speed.metrics.get("tokens_per_sec")
            if isinstance(v, (int, float)):
                tok_s.append(float(v))
            v = speed.metrics.get("ttft_ms")
            if isinstance(v, (int, float)):
                ttft.append(float(v))
        lat = tasks.get("latency") or tasks.get("prompt_latency")
        if lat:
            v = lat.metrics.get("ttft_ms")
            if isinstance(v, (int, float)):
                ttft.append(float(v))
        q = tasks.get("quality") or tasks.get("instruction_quality")
        if q:
            v = q.metrics.get("score")
            if isinstance(v, (int, float)):
                quality.append(float(v))
        t = tasks.get("tools") or tasks.get("tool_calling")
        if t:
            v = t.metrics.get("success_rate")
            if isinstance(v, (int, float)):
                tools.append(float(v))
        c = tasks.get("context") or tasks.get("long_context")
        if c:
            v = c.metrics.get("score")
            if isinstance(v, (int, float)):
                context.append(float(v))

    scores: list[ModelScore] = []
    for (model, provider), tasks in by_model.items():
        breakdown: dict[str, float] = {}

        speed = tasks.get("speed") or tasks.get("generation_speed")
        if speed:
            tps = speed.metrics.get("tokens_per_sec")
            tps_f = float(tps) if isinstance(tps, (int, float)) else None
            tt = speed.metrics.get("ttft_ms")
            tt_f = float(tt) if isinstance(tt, (int, float)) else None
            # Absolute bands when single model; relative when peers exist
            if len(tok_s) <= 1:
                # Informal reference: 0–80 tok/s maps to 0–1
                speed_score = min(1.0, (tps_f or 0.0) / 80.0) * 0.7 + (
                    max(0.0, 1.0 - (tt_f or 1000.0) / 2000.0) * 0.3
                )
            else:
                speed_score = 0.7 * _norm_higher_better(tps_f, tok_s) + 0.3 * _norm_lower_better(
                    tt_f, ttft
                )
            breakdown["speed"] = round(speed_score, 4)

        q = tasks.get("quality") or tasks.get("instruction_quality")
        if q:
            v = q.metrics.get("score")
            qf = float(v) if isinstance(v, (int, float)) else 0.0
            breakdown["quality"] = round(qf if qf <= 1.0 else qf / 100.0, 4)

        t = tasks.get("tools") or tasks.get("tool_calling")
        if t:
            v = t.metrics.get("success_rate")
            tf = float(v) if isinstance(v, (int, float)) else 0.0
            breakdown["tools"] = round(tf if tf <= 1.0 else tf / 100.0, 4)

        c = tasks.get("context") or tasks.get("long_context")
        if c:
            v = c.metrics.get("score")
            cf = float(v) if isinstance(v, (int, float)) else 0.0
            breakdown["context"] = round(cf if cf <= 1.0 else cf / 100.0, 4)

        total_w = 0.0
        overall = 0.0
        for key, weight in DEFAULT_WEIGHTS.items():
            if key in breakdown and weight > 0:
                overall += breakdown[key] * weight
                total_w += weight
        if total_w > 0:
            overall /= total_w

        scores.append(
            ModelScore(
                model=model,
                provider=provider,
                overall=round(overall, 4),
                breakdown=breakdown,
                tasks=tasks,
            )
        )

    scores.sort(key=lambda s: s.overall, reverse=True)
    return scores


def sparkline(values: list[float], width: int = 16) -> str:
    if not values:
        return ""
    blocks = "▁▂▃▄▅▆▇█"
    lo, hi = min(values), max(values)
    if hi <= lo:
        return blocks[0] * min(width, len(values))
    # downsample
    step = max(1, len(values) // width)
    sampled = values[::step][:width]
    out = []
    for v in sampled:
        idx = int((v - lo) / (hi - lo) * (len(blocks) - 1))
        out.append(blocks[idx])
    return "".join(out)
