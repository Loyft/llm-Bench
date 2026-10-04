"""Instruction following quality with offline rubrics."""

from __future__ import annotations

import re

from llm_bench.benchmarks.tasks.base import BenchTask
from llm_bench.config import BenchConfig
from llm_bench.providers.base import BaseProvider
from llm_bench.schema import ChatMessage, ChatRequest, ModelRef, SampleResult

# Fixed prompt suite — deterministic offline checks
PROMPTS = [
    {
        "id": "math",
        "prompt": "What is 17 * 19? Reply with only the number.",
        "check": lambda t: "323" in re.sub(r"\s+", "", t),
    },
    {
        "id": "json",
        "prompt": (
            'Return a JSON object with keys "name" and "age" for a person named Ada who is 36. '
            "Reply with JSON only."
        ),
        "check": lambda t: '"name"' in t and '"age"' in t and "Ada" in t and "36" in t,
    },
    {
        "id": "code",
        "prompt": (
            "Write a Python function named add(a, b) that returns the sum of a and b. "
            "Reply with code only."
        ),
        "check": lambda t: "def add" in t and "return" in t,
    },
    {
        "id": "summary",
        "prompt": (
            "Summarize in exactly one sentence: Photosynthesis converts light energy into "
            "chemical energy in plants."
        ),
        "check": lambda t: 1 <= t.strip().count(".") <= 2 and "photo" in t.lower(),
    },
]


class InstructionQualityTask(BenchTask):
    name = "instruction_quality"
    aliases = ("quality",)

    async def run(
        self,
        provider: BaseProvider,
        model: ModelRef,
        config: BenchConfig,
        *,
        run_index: int,
    ) -> SampleResult:
        item = PROMPTS[run_index % len(PROMPTS)]
        req = ChatRequest(
            model=model,
            messages=[ChatMessage(role="user", content=item["prompt"])],
            max_tokens=128,
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
                {"score": 0.0, "passed": False},
                error=result.error,
                detail={"prompt_id": item["id"]},
            )

        text = result.text or ""
        passed = bool(item["check"](text))
        return self.make_sample(
            model,
            run_index,
            {
                "score": 1.0 if passed else 0.0,
                "passed": passed,
                "ttft_ms": result.ttft_ms,
                "total_ms": result.total_ms,
            },
            detail={"prompt_id": item["id"], "response": text[:300]},
        )
