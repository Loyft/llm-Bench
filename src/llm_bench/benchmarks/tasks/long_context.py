"""Long context handling near configured window."""

from __future__ import annotations

from llm_bench.benchmarks.tasks.base import BenchTask
from llm_bench.config import BenchConfig
from llm_bench.providers.base import BaseProvider
from llm_bench.schema import ChatMessage, ChatRequest, ModelRef, SampleResult


class LongContextTask(BenchTask):
    name = "long_context"
    aliases = ("context",)

    async def run(
        self,
        provider: BaseProvider,
        model: ModelRef,
        config: BenchConfig,
        *,
        run_index: int,
    ) -> SampleResult:
        # Approximate tokens with ~4 chars/token; use a fraction of context
        frac = 0.25 + 0.15 * (run_index % 3)
        target_tokens = max(512, int(config.context * frac))
        target_chars = target_tokens * 4
        needle = f"SECRET_CODE_{run_index}_ALPHA"
        body = ("lorem ipsum dolor sit amet " * 200) 
        # Build context with needle near the middle
        half = max(1, target_chars // 2)
        text = (body * ((half // len(body)) + 1))[:half]
        text += f"\n\nRemember this code: {needle}\n\n"
        text += (body * ((half // len(body)) + 1))[:half]
        prompt = (
            f"You are given a long document.\n\n{text}\n\n"
            "What is the SECRET code mentioned in the document? "
            "Reply with only the code."
        )
        req = ChatRequest(
            model=model,
            messages=[ChatMessage(role="user", content=prompt)],
            max_tokens=64,
            temperature=0.0,
            seed=config.seed,
            system=config.system,
            timeout=max(config.timeout, 180.0),
        )
        result = await provider.chat(req)
        if result.error:
            return self.make_sample(
                model,
                run_index,
                {"score": 0.0, "target_tokens": target_tokens},
                error=result.error,
            )

        found = needle in (result.text or "")
        # Partial credit if ALPHA or SECRET_CODE appears
        partial = ("SECRET_CODE" in (result.text or "")) or ("ALPHA" in (result.text or ""))
        score = 1.0 if found else (0.4 if partial else 0.0)

        return self.make_sample(
            model,
            run_index,
            {
                "score": score,
                "needle_found": found,
                "target_tokens": target_tokens,
                "ttft_ms": result.ttft_ms,
                "total_ms": result.total_ms,
                "prompt_tokens": result.prompt_tokens,
            },
            detail={"response": (result.text or "")[:200]},
        )
