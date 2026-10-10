"""Base class for tasks that grade a fixed bank of prompts offline."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from llm_bench.benchmarks.tasks.base import BenchTask
from llm_bench.benchmarks.tasks.common import response_metrics, split_thinking
from llm_bench.config import BenchConfig
from llm_bench.providers.base import BaseProvider
from llm_bench.schema import ChatMessage, ChatRequest, ModelRef, SampleResult


@dataclass
class Grade:
    score: float
    extra: dict[str, Any] = field(default_factory=dict)
    detail: dict[str, Any] = field(default_factory=dict)


def interleave_by_category(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Round-robin items across categories so a ``--max-items`` cap stays balanced."""
    buckets: dict[str, list[dict[str, Any]]] = {}
    for item in items:
        buckets.setdefault(item.get("category", ""), []).append(item)
    out: list[dict[str, Any]] = []
    while any(buckets.values()):
        for cat in list(buckets):
            if buckets[cat]:
                out.append(buckets[cat].pop(0))
    return out


class ItemTask(BenchTask):
    """One sample == one item from ``items``; ``score`` is 0..1 per item.

    Subclasses define ``items`` (dicts with at least ``id`` and ``category``),
    ``build_messages`` and ``grade``. Samples sweep the bank once (temperature 0
    makes repeats pointless); ``--max-items`` caps the sweep.
    """

    items: list[dict[str, Any]] = []
    max_tokens: int = 256
    temperature: float = 0.0
    timeout_floor: float = 0.0

    def sample_count(self, config: BenchConfig) -> int:
        n = len(self.items)
        return min(n, config.max_items) if config.max_items else n

    def build_messages(self, item: dict[str, Any]) -> list[ChatMessage]:
        return [ChatMessage(role="user", content=item["prompt"])]

    def system_for(self, item: dict[str, Any], config: BenchConfig) -> str | None:
        return item.get("system") or config.system

    def grade(self, item: dict[str, Any], answer: str) -> Grade:
        raise NotImplementedError

    def gold_response(self, item: dict[str, Any]) -> str:
        """A response that should score 1.0 (used by tests to validate graders)."""
        raise NotImplementedError

    async def respond(
        self,
        provider: BaseProvider,
        model: ModelRef,
        config: BenchConfig,
        item: dict[str, Any],
    ):
        req = ChatRequest(
            model=model,
            messages=self.build_messages(item),
            max_tokens=item.get("max_tokens", self.max_tokens),
            temperature=self.temperature,
            seed=config.seed,
            system=self.system_for(item, config),
            timeout=max(config.timeout, self.timeout_floor),
        )
        return req, await provider.chat(req)

    async def run(
        self,
        provider: BaseProvider,
        model: ModelRef,
        config: BenchConfig,
        *,
        run_index: int,
    ) -> SampleResult:
        item = self.items[run_index % len(self.items)]
        cat = item.get("category", "")
        base_detail = {"item_id": item["id"], "category": cat}
        req, result = await self.respond(provider, model, config, item)
        if result.error:
            return self.make_sample(
                model,
                run_index,
                {"score": 0.0, "passed": False, **({f"score_{cat}": 0.0} if cat else {})},
                error=result.error,
                detail=base_detail,
            )

        answer, thinking = split_thinking(result.text or "")
        grade = self.grade(item, answer)
        metrics: dict[str, Any] = {
            "score": round(grade.score, 4),
            "passed": grade.score >= 0.999,
            **response_metrics(result, req.max_tokens),
            **grade.extra,
        }
        if cat:
            metrics[f"score_{cat}"] = round(grade.score, 4)
        if thinking:
            metrics["thinking_chars"] = len(thinking)
        return self.make_sample(
            model,
            run_index,
            metrics,
            detail={
                **base_detail,
                "prompt": item.get("prompt", "")[:300],
                "response": answer[:400],
                **grade.detail,
            },
        )
