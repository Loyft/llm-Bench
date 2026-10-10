"""Answer stability under sampling: ask the same question several times at temperature > 0."""

from __future__ import annotations

from collections import Counter

from llm_bench.benchmarks.tasks.base import BenchTask
from llm_bench.benchmarks.tasks.common import response_metrics, split_thinking
from llm_bench.benchmarks.tasks.reasoning import INSTRUCTION, check_answer
from llm_bench.benchmarks.tasks.reasoning import ITEMS as REASONING_ITEMS
from llm_bench.config import BenchConfig
from llm_bench.providers.base import BaseProvider
from llm_bench.schema import ChatMessage, ChatRequest, ModelRef, SampleResult

QUESTION_IDS = ["change", "machines", "ages", "tallest", "sheep"]
SAMPLES_PER_QUESTION = 3
TEMPERATURE = 0.8


class ConsistencyTask(BenchTask):
    name = "consistency"
    aliases = ("stability",)
    description = f"same question x{SAMPLES_PER_QUESTION} at T={TEMPERATURE}: agreement + accuracy"
    kind = "capability"
    items = [it for it in REASONING_ITEMS if it["id"] in QUESTION_IDS]

    def sample_count(self, config: BenchConfig) -> int:
        n = len(self.items)
        return min(n, config.max_items) if config.max_items else n

    async def run(
        self,
        provider: BaseProvider,
        model: ModelRef,
        config: BenchConfig,
        *,
        run_index: int,
    ) -> SampleResult:
        item = self.items[run_index % len(self.items)]
        answers: list[str] = []
        correct = 0
        tok_total = 0
        truncated = 0
        last_error: str | None = None
        for k in range(SAMPLES_PER_QUESTION):
            req = ChatRequest(
                model=model,
                messages=[ChatMessage(role="user", content=f"{INSTRUCTION}\n\n{item['prompt']}")],
                max_tokens=512,
                temperature=TEMPERATURE,
                seed=config.seed + 1 + k,
                system=config.system,
                timeout=config.timeout,
            )
            result = await provider.chat(req)
            if result.error:
                last_error = result.error
                continue
            text, _ = split_thinking(result.text or "")
            ok, extracted = check_answer(item, text)
            answers.append(str(extracted).strip().lower() or "<none>")
            correct += int(ok)
            tok_total += result.completion_tokens or 0
            truncated += int(response_metrics(result, req.max_tokens)["truncated"])

        if not answers:
            return self.make_sample(
                model, run_index, {"score": 0.0, "accuracy": 0.0}, error=last_error,
                detail={"item_id": item["id"]},
            )
        modal_n = Counter(answers).most_common(1)[0][1]
        agreement = modal_n / SAMPLES_PER_QUESTION
        return self.make_sample(
            model,
            run_index,
            {
                "score": round(agreement, 4),
                "agreement": round(agreement, 4),
                "accuracy": round(correct / SAMPLES_PER_QUESTION, 4),
                "unique_answers": len(set(answers)),
                "completion_tokens": tok_total / max(1, len(answers)),
                "truncated": truncated > 0,
            },
            detail={"item_id": item["id"], "answers": answers, "expected": item["answer"]},
        )
