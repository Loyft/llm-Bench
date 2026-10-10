"""Tests for metric aggregation and scoring."""

from llm_bench.metrics import aggregate_samples, score_models, sparkline, tokens_per_sec
from llm_bench.schema import SampleResult


def test_tokens_per_sec() -> None:
    tps = tokens_per_sec(100, total_ms=2000.0, ttft_ms=200.0)
    assert tps is not None
    assert 50 < tps < 60


def test_aggregate_and_score() -> None:
    samples = [
        SampleResult(
            task="generation_speed",
            model="a",
            provider="ollama",
            run_index=0,
            metrics={"tokens_per_sec": 40.0, "ttft_ms": 100.0},
        ),
        SampleResult(
            task="generation_speed",
            model="b",
            provider="ollama",
            run_index=0,
            metrics={"tokens_per_sec": 80.0, "ttft_ms": 50.0},
        ),
        SampleResult(
            task="instruction_quality",
            model="a",
            provider="ollama",
            run_index=0,
            metrics={"score": 0.5},
        ),
        SampleResult(
            task="instruction_quality",
            model="b",
            provider="ollama",
            run_index=0,
            metrics={"score": 1.0},
        ),
    ]
    summaries = aggregate_samples(samples)
    assert len(summaries) == 4
    scores = score_models(summaries)
    assert scores[0].model == "b"
    assert scores[0].overall >= scores[1].overall


def test_sparkline() -> None:
    s = sparkline([1.0, 2.0, 3.0, 4.0])
    assert len(s) >= 1
    assert sparkline([]) == ""


def test_peer_scoring_is_ratio_to_best_not_min_max() -> None:
    """A 20% speed gap between two models must not look like 1.0 vs ~0.2."""
    def speed(model: str, tps: float, ttft: float) -> SampleResult:
        return SampleResult(task="generation_speed", model=model, provider="p", run_index=0,
                            metrics={"tokens_per_sec": tps, "ttft_ms": ttft})

    scores = {s.model: s for s in score_models(aggregate_samples(
        [speed("fast", 100.0, 50.0), speed("slow", 80.0, 62.5)]))}
    assert scores["fast"].breakdown["speed"] == 1.0
    assert abs(scores["slow"].breakdown["speed"] - (0.7 * 0.8 + 0.3 * 0.8)) < 1e-6
