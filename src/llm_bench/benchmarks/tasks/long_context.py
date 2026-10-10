"""Needle-in-a-haystack retrieval across context lengths and needle depths."""

from __future__ import annotations

import random

from llm_bench.benchmarks.tasks.base import BenchTask
from llm_bench.benchmarks.tasks.common import response_metrics, split_thinking
from llm_bench.config import BenchConfig
from llm_bench.providers.base import BaseProvider
from llm_bench.schema import ChatMessage, ChatRequest, ModelRef, SampleResult

# (fraction of configured context filled, needle depth 0=start..1=end)
GRID = [(frac, depth) for frac in (0.3, 0.7) for depth in (0.1, 0.5, 0.9)]

_SUBJECTS = ["The committee", "A passing traveler", "The old librarian", "Our neighbour",
             "The river pilot", "A young baker", "The night watchman", "The harbour master"]
_VERBS = ["noted", "remembered", "described", "questioned", "recorded", "praised", "doubted"]
_OBJECTS = ["the quiet harbour", "a stack of unsorted letters", "the price of salt",
            "an unfinished bridge", "the weather over the hills", "a missing umbrella",
            "the schedule for the ferry", "a recipe for dark bread"]
_TAILS = ["before sunrise.", "without much enthusiasm.", "in the middle of spring.",
          "during the long winter.", "after the market closed.", "as the lamps were lit."]
_WORDS = ["AMBER", "COBALT", "JADE", "SILVER", "CRIMSON", "IVORY", "TEAL", "SAFFRON"]


def filler(rng: random.Random, n_chars: int) -> str:
    out: list[str] = []
    size = 0
    while size < n_chars:
        s = (f"{rng.choice(_SUBJECTS)} {rng.choice(_VERBS)} {rng.choice(_OBJECTS)} "
             f"{rng.choice(_TAILS)}")
        out.append(s)
        size += len(s) + 1
    return " ".join(out)


def build_haystack(target_chars: int, depth: float, needle: str, seed: int) -> str:
    rng = random.Random(seed)
    before = filler(rng, int(target_chars * depth))
    after = filler(rng, int(target_chars * (1 - depth)))
    return f"{before}\n\n{needle}\n\n{after}"


class LongContextTask(BenchTask):
    name = "long_context"
    aliases = ("context",)
    kind = "capability"
    description = "needle retrieval at several context fractions and needle depths"

    def sample_count(self, config: BenchConfig) -> int:
        n = len(GRID)
        return min(n, config.max_items) if config.max_items else n

    async def run(
        self,
        provider: BaseProvider,
        model: ModelRef,
        config: BenchConfig,
        *,
        run_index: int,
    ) -> SampleResult:
        frac, depth = GRID[run_index % len(GRID)]
        # ~4 chars per token; leave headroom for the instruction and the answer
        target_tokens = max(512, int(config.context * frac))
        rng = random.Random(config.seed * 1000 + run_index)
        code = f"{rng.choice(_WORDS)}-{rng.randint(1000, 9999)}"
        needle = f"IMPORTANT: The vault access code is {code}."
        text = build_haystack(target_tokens * 4, depth, needle, config.seed + run_index)
        prompt = (
            f"Read the following document.\n\n{text}\n\n"
            "What is the vault access code mentioned in the document? Reply with only the code."
        )
        max_tokens = 64
        req = ChatRequest(
            model=model,
            messages=[ChatMessage(role="user", content=prompt)],
            max_tokens=max_tokens,
            temperature=0.0,
            seed=config.seed,
            system=config.system,
            timeout=max(config.timeout, 180.0),
        )
        result = await provider.chat(req)
        depth_key = f"score_depth_{int(depth * 100)}"
        frac_key = f"score_ctx_{int(frac * 100)}"
        if result.error:
            return self.make_sample(
                model, run_index,
                {"score": 0.0, depth_key: 0.0, frac_key: 0.0, "target_tokens": target_tokens},
                error=result.error, detail={"depth": depth, "frac": frac},
            )

        answer, _ = split_thinking(result.text or "")
        found = code.lower() in answer.lower()
        score = 1.0 if found else 0.0
        # How much of the prompt the server actually kept: <1 means it silently truncated.
        kept = (
            round(result.prompt_tokens / target_tokens, 3)
            if result.prompt_tokens
            else None
        )
        return self.make_sample(
            model,
            run_index,
            {
                "score": score,
                "needle_found": found,
                depth_key: score,
                frac_key: score,
                "target_tokens": target_tokens,
                "prompt_tokens_vs_target": kept,
                **response_metrics(result, max_tokens),
            },
            detail={"depth": depth, "frac": frac, "expected": code, "response": answer[:200]},
        )
