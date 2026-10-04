"""Prefill metrics and memory timeline basics."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest

from llm_bench.config import BenchConfig
from llm_bench.providers.base import compute_prefill_tps
from llm_bench.runner import BenchRunner
from llm_bench.schema import ProviderName
from mock_provider import MockProvider


def test_compute_prefill_tps() -> None:
    assert compute_prefill_tps(100, prefill_ms=50) == pytest.approx(2000.0)
    assert compute_prefill_tps(100, ttft_ms=100) == pytest.approx(1000.0)
    assert compute_prefill_tps(None, prefill_ms=50) is None


@pytest.mark.asyncio
async def test_runner_captures_prefill_and_memory(tmp_path: Path) -> None:
    config = BenchConfig(
        provider=ProviderName.OPENAI,
        models=["mock-model"],
        tasks=["speed", "latency"],
        tools=False,
        runs=1,
        warmup=0,
        report_dir=tmp_path / "reports",
        workspace=tmp_path / "ws",
        keep_workspace=False,
    )
    mock = MockProvider()
    with patch("llm_bench.runner.get_provider", return_value=mock):
        runner = BenchRunner(config)
        report = await runner.run()
        runner.workspace.cleanup()

    speed = next(s for s in report.summaries if s.task == "generation_speed")
    assert speed.metrics.get("prefill_tokens_per_sec") is not None
    assert float(speed.metrics["prefill_tokens_per_sec"]) > 0

    assert report.memory.samples
    assert report.memory.peak_system_used_mb > 0
