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
