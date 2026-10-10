"""Concurrent-request throughput: does the server scale when several requests overlap?"""

from __future__ import annotations

import asyncio
import time

from llm_bench.benchmarks.tasks.base import BenchTask
from llm_bench.benchmarks.tasks.common import chat_speed_metrics
from llm_bench.benchmarks.tasks.generation_speed import PROMPT
from llm_bench.config import BenchConfig
from llm_bench.providers.base import BaseProvider, percentile
from llm_bench.schema import ChatMessage, ChatRequest, ModelRef, SampleResult

LEVELS = (1, 2, 4)
MAX_TOKENS = 128


class ThroughputTask(BenchTask):
    name = "throughput"
    aliases = ("concurrency",)
    kind = "performance"
    description = f"aggregate tok/s with {', '.join(map(str, LEVELS))} concurrent requests"

    def sample_count(self, config: BenchConfig) -> int:
        return len(LEVELS)

    async def run(
        self,
        provider: BaseProvider,
        model: ModelRef,
        config: BenchConfig,
        *,
        run_index: int,
    ) -> SampleResult:
        n = LEVELS[run_index % len(LEVELS)]

        def make_req(k: int) -> ChatRequest:
            return ChatRequest(
                model=model,
                messages=[ChatMessage(role="user", content=f"{PROMPT} (variant {k})")],
                max_tokens=MAX_TOKENS,
                temperature=0.0,
                seed=config.seed,
                system=config.system,
                timeout=config.timeout,
            )

        start = time.perf_counter()
        results = await asyncio.gather(*(provider.chat(make_req(k)) for k in range(n)))
        wall_s = time.perf_counter() - start

        ok = [r for r in results if not r.error]
        if not ok:
            err = next((r.error for r in results if r.error), "all requests failed")
            return self.make_sample(model, run_index, {"concurrency": n}, error=err)

        tokens = sum(r.completion_tokens or 0 for r in ok)
        per_req = [m for r in ok if (m := chat_speed_metrics(r)["tokens_per_sec"])]
        totals = [r.total_ms for r in ok if r.total_ms]
        agg = tokens / wall_s if wall_s > 0 else None
        return self.make_sample(
            model,
            run_index,
            {
                "concurrency": n,
                f"agg_tps_c{n}": agg,
                f"per_request_tps_c{n}": sum(per_req) / len(per_req) if per_req else None,
                f"p95_total_ms_c{n}": percentile(totals, 95),
                f"failed_c{n}": len(results) - len(ok),
                "wall_ms": wall_s * 1000.0,
                "completion_tokens": tokens,
            },
            detail={"requests": n, "ok": len(ok)},
        )
