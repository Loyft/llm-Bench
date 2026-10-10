"""Metric helpers and overall scoring."""

from __future__ import annotations

import statistics
from collections import defaultdict

from llm_bench.providers.base import percentile
from llm_bench.schema import ModelScore, SampleResult, TaskSummary

# Capability tasks: task name -> (breakdown key, metric to read, weight).
# Weights sum to 1.0; categories a run did not execute are dropped and the rest renormalised.
CAPABILITY_WEIGHTS: dict[str, tuple[str, str, float]] = {
    "reasoning": ("reasoning", "score", 0.20),
    "coding": ("coding", "score", 0.20),
    "knowledge": ("knowledge", "score", 0.15),
    "instruction_following": ("instructions", "score", 0.12),
    "tool_calling": ("tools", "success_rate", 0.10),
    "long_context": ("context", "score", 0.08),
    "hallucination": ("hallucination", "score", 0.07),
    "multi_turn": ("multi_turn", "score", 0.04),
    "instruction_quality": ("quality", "score", 0.03),
    "consistency": ("consistency", "score", 0.01),
}
# Within performance: decode speed + TTFT vs prefill throughput.
PERFORMANCE_WEIGHTS = {"speed": 0.7, "prefill": 0.3}
# Overall = blend of "how fast" and "how good".
OVERALL_WEIGHTS = {"performance": 0.3, "capability": 0.7}

# Metrics that get p50/p95 in summaries (latency/throughput are skewed; means hide tails).
PERCENTILE_KEYS = {
    "ttft_ms",
    "total_ms",
    "itl_ms_mean",
    "tokens_per_sec",
    "decode_tokens_per_sec",
    "prefill_tokens_per_sec",
    "completion_tokens",
}
MINMAX_KEYS = {"tokens_per_sec", "ttft_ms"}


def _is_accuracy_key(key: str) -> bool:
    return key in {"score", "passed", "success", "success_rate"} or key.startswith("score_")


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
            for k, v in item.metrics.items():
                # A failed request is a wrong answer, not a missing one: keep its accuracy
                # fields (zeros) so errors cannot inflate a model's score.
                if item.error and not _is_accuracy_key(k):
                    continue
                if isinstance(v, bool):
                    numeric[k].append(1.0 if v else 0.0)
                elif isinstance(v, (int, float)):
                    numeric[k].append(float(v))

        metrics: dict[str, float | int | str | bool | None] = {}
        for k, vals in numeric.items():
            mean = safe_mean(vals)
            if mean is None:
                continue
            metrics[k] = round(mean, 4)
            std = safe_stdev(vals)
            if std is not None:
                metrics[f"{k}_std"] = round(std, 4)
            if k in PERCENTILE_KEYS and len(vals) >= 2:
                metrics[f"{k}_p50"] = round(percentile(vals, 50) or 0.0, 4)
                metrics[f"{k}_p95"] = round(percentile(vals, 95) or 0.0, 4)
            if k in MINMAX_KEYS and len(vals) >= 2:
                metrics[f"{k}_min"] = round(min(vals), 4)
                metrics[f"{k}_max"] = round(max(vals), 4)

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
    """Value as a fraction of the best peer (1.0 = best). Ratio, not min-max, so a small
    gap between two models stays small instead of stretching to 1.0 vs 0.0."""
    if value is None or not peers:
        return 0.0
    hi = max(peers)
    return 1.0 if hi <= 0 else max(0.0, min(1.0, value / hi))


def _norm_lower_better(value: float | None, peers: list[float]) -> float:
    """Best (lowest) peer as a fraction of the value (1.0 = best)."""
    if value is None or not peers:
        return 0.0
    lo = min(peers)
    if value <= 0 or lo <= 0:
        return 1.0 if value <= lo else 0.0
    return max(0.0, min(1.0, lo / value))


def _num(block: TaskSummary | None, key: str) -> float | None:
    if block is None:
        return None
    v = block.metrics.get(key)
    return float(v) if isinstance(v, (int, float)) and not isinstance(v, bool) else None


def _unit(v: float | None) -> float:
    if v is None:
        return 0.0
    return max(0.0, min(1.0, v if v <= 1.0 else v / 100.0))


def _weighted(parts: dict[str, float], weights: dict[str, float]) -> float | None:
    used = {k: w for k, w in weights.items() if k in parts and w > 0}
    total = sum(used.values())
    if total <= 0:
        return None
    return sum(parts[k] * w for k, w in used.items()) / total


def score_models(summaries: list[TaskSummary]) -> list[ModelScore]:
    by_model: dict[tuple[str, str], dict[str, TaskSummary]] = defaultdict(dict)
    for s in summaries:
        by_model[(s.model, s.provider)][s.task] = s

    # Peer pools for relative speed scoring when several models are compared
    tok_s: list[float] = []
    ttft: list[float] = []
    prefill: list[float] = []
    for tasks in by_model.values():
        speed = tasks.get("generation_speed")
        lat = tasks.get("prompt_latency")
        if (v := _num(speed, "tokens_per_sec")) is not None:
            tok_s.append(v)
        for block in (speed, lat):
            if (v := _num(block, "ttft_ms")) is not None:
                ttft.append(v)
        if (v := _num(lat, "prefill_tokens_per_sec")) is not None:
            prefill.append(v)
        elif (v := _num(speed, "prefill_tokens_per_sec")) is not None:
            prefill.append(v)

    scores: list[ModelScore] = []
    for (model, provider), tasks in by_model.items():
        breakdown: dict[str, float] = {}

        speed = tasks.get("generation_speed")
        if speed:
            tps_f = _num(speed, "tokens_per_sec")
            tt_f = _num(speed, "ttft_ms")
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

        lat = tasks.get("prompt_latency")
        pf = _num(lat, "prefill_tokens_per_sec")
        if pf is None:
            pf = _num(speed, "prefill_tokens_per_sec")
        if pf is not None:
            pf_score = (
                min(1.0, pf / 1000.0) if len(prefill) <= 1 else _norm_higher_better(pf, prefill)
            )
            breakdown["prefill"] = round(pf_score, 4)

        capability_parts: dict[str, float] = {}
        capability_weights: dict[str, float] = {}
        for task_name, (key, metric, weight) in CAPABILITY_WEIGHTS.items():
            block = tasks.get(task_name)
            if block is None:
                continue
            value = _unit(_num(block, metric))
            breakdown[key] = round(value, 4)
            capability_parts[key] = value
            capability_weights[key] = weight

        performance = _weighted(
            {k: breakdown[k] for k in PERFORMANCE_WEIGHTS if k in breakdown},
            PERFORMANCE_WEIGHTS,
        )
        capability = _weighted(capability_parts, capability_weights)
        overall = (
            _weighted(
                {
                    k: v
                    for k, v in (("performance", performance), ("capability", capability))
                    if v is not None
                },
                OVERALL_WEIGHTS,
            )
            or 0.0
        )

        total_samples = sum(t.samples for t in tasks.values())
        total_errors = sum(t.errors for t in tasks.values())
        reliability = 1.0 - (total_errors / total_samples) if total_samples else 0.0

        scores.append(
            ModelScore(
                model=model,
                provider=provider,
                overall=round(overall, 4),
                performance=round(performance, 4) if performance is not None else None,
                capability=round(capability, 4) if capability is not None else None,
                reliability=round(reliability, 4),
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
