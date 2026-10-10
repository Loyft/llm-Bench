"""Generation speed / decode throughput task."""

from __future__ import annotations

from llm_bench.benchmarks.tasks.base import BenchTask
from llm_bench.benchmarks.tasks.common import chat_speed_metrics
from llm_bench.config import BenchConfig
from llm_bench.providers.base import BaseProvider
from llm_bench.schema import ChatMessage, ChatRequest, ModelRef, SampleResult

PROMPT = (
    "Write a detailed explanation of how a CPU executes instructions, "
    "covering fetch, decode, execute, and write-back. Use clear paragraphs."
)


class GenerationSpeedTask(BenchTask):
    name = "generation_speed"
    aliases = ("speed",)
    kind = "performance"
    description = "decode tok/s, TTFT and inter-token latency on a 256-token generation"

    async def run(
        self,
        provider: BaseProvider,
        model: ModelRef,
        config: BenchConfig,
        *,
        run_index: int,
    ) -> SampleResult:
        prompt = PROMPT
        req = ChatRequest(
            model=model,
            messages=[ChatMessage(role="user", content=prompt)],
            max_tokens=256,
            temperature=0.0,
            seed=config.seed,
            system=config.system,
            timeout=config.timeout,
        )
        result = await provider.chat(req)
        if result.error:
            return self.make_sample(
                model,
                run_index,
                {},
                error=result.error,
            )
        return self.make_sample(
            model,
            run_index,
            chat_speed_metrics(result),
            detail={"text_preview": (result.text or "")[:200]},
        )
