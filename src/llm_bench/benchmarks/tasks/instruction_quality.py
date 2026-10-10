"""Basic instruction-following sanity checks (fast smoke test; see ``instructions`` for depth)."""

from __future__ import annotations

import re
from typing import Any

from llm_bench.benchmarks.tasks.items import Grade, ItemTask

# Fixed prompt suite — deterministic offline checks
PROMPTS: list[dict[str, Any]] = [
    {
        "id": "math",
        "category": "sanity",
        "prompt": "What is 17 * 19? Reply with only the number.",
        "check": lambda t: "323" in re.sub(r"\s+", "", t),
        "gold": "323",
    },
    {
        "id": "json",
        "category": "sanity",
        "prompt": (
            'Return a JSON object with keys "name" and "age" for a person named Ada who is 36. '
            "Reply with JSON only."
        ),
        "check": lambda t: '"name"' in t and '"age"' in t and "Ada" in t and "36" in t,
        "gold": '{"name": "Ada", "age": 36}',
    },
    {
        "id": "code",
        "category": "sanity",
        "prompt": (
            "Write a Python function named add(a, b) that returns the sum of a and b. "
            "Reply with code only."
        ),
        "check": lambda t: "def add" in t and "return" in t,
        "gold": "def add(a, b):\n    return a + b",
    },
    {
        "id": "summary",
        "category": "sanity",
        "prompt": (
            "Summarize in exactly one sentence: Photosynthesis converts light energy into "
            "chemical energy in plants."
        ),
        "check": lambda t: 1 <= t.strip().count(".") <= 2 and "photo" in t.lower(),
        "gold": "Photosynthesis turns light into chemical energy in plants.",
    },
]


class InstructionQualityTask(ItemTask):
    name = "instruction_quality"
    aliases = ("quality",)
    description = "4-prompt sanity smoke test (math, JSON, code, summary)"
    items = PROMPTS
    max_tokens = 256

    def grade(self, item: dict[str, Any], answer: str) -> Grade:
        return Grade(1.0 if item["check"](answer) else 0.0)

    def gold_response(self, item: dict[str, Any]) -> str:
        return item["gold"]
