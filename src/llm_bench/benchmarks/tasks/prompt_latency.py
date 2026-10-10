"""Prompt / prefill latency across sizes."""

from __future__ import annotations

from llm_bench.benchmarks.tasks.base import BenchTask
from llm_bench.benchmarks.tasks.common import chat_speed_metrics
from llm_bench.config import BenchConfig
from llm_bench.providers.base import BaseProvider
from llm_bench.schema import ChatMessage, ChatRequest, ModelRef, SampleResult


def _pad_prompt(base: str, target_chars: int) -> str:
    filler = " The quick brown fox jumps over the lazy dog."
    out = base
    while len(out) < target_chars:
        out += filler
    return out[:target_chars]


class PromptLatencyTask(BenchTask):
    name = "prompt_latency"
    aliases = ("latency",)
    kind = "performance"
    description = "prefill speed and TTFT for 0.5k / 2k / 8k-character prompts"

    async def run(
        self,
        provider: BaseProvider,
        model: ModelRef,
        config: BenchConfig,
        *,
        run_index: int,
    ) -> SampleResult:
        # Cycle sizes by run index
        sizes = [500, 2000, 8000]
        target = sizes[run_index % len(sizes)]
        prompt = _pad_prompt(
            "Summarize the following text in one sentence:\n\n",
            target,
        )
        req = ChatRequest(
            model=model,
            messages=[ChatMessage(role="user", content=prompt)],
            max_tokens=64,
            temperature=0.0,
            seed=config.seed,
            system=config.system,
            timeout=config.timeout,
        )
        result = await provider.chat(req)
        if result.error:
            return self.make_sample(model, run_index, {}, error=result.error)

        metrics = chat_speed_metrics(result)
        metrics["prompt_chars"] = target
        return self.make_sample(model, run_index, metrics)
