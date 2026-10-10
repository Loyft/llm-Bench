"""Benchmark task base class."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from llm_bench.config import BenchConfig
from llm_bench.providers.base import BaseProvider
from llm_bench.schema import ModelRef, SampleResult


class BenchTask(ABC):
    name: str
    aliases: tuple[str, ...] = ()
    requires_tools: bool = False
    requires_code_exec: bool = False
    # "performance" tasks measure speed; "capability" tasks measure answer quality.
    kind: str = "capability"
    description: str = ""

    def sample_count(self, config: BenchConfig) -> int:
        """How many samples the runner should collect for this task."""
        return config.runs

    @abstractmethod
    async def run(
        self,
        provider: BaseProvider,
        model: ModelRef,
        config: BenchConfig,
        *,
        run_index: int,
    ) -> SampleResult:
        """Execute one sample for this task."""

    def make_sample(
        self,
        model: ModelRef,
        run_index: int,
        metrics: dict[str, Any],
        *,
        detail: dict[str, Any] | None = None,
        error: str | None = None,
    ) -> SampleResult:
        return SampleResult(
            task=self.name,
            model=model.label,
            provider=model.provider.value,
            run_index=run_index,
            metrics=metrics,
            detail=detail or {},
            error=error,
        )
