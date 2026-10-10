"""Multi-turn conversations: memory, corrections and system-prompt persistence."""

from __future__ import annotations

import re
from typing import Any

from llm_bench.benchmarks.tasks.items import Grade, ItemTask, interleave_by_category
from llm_bench.schema import ChatMessage


def _conv(*turns: tuple[str, str]) -> list[dict[str, str]]:
    return [{"role": r, "content": c} for r, c in turns]


def _one_word(t: str) -> bool:
    return len(re.findall(r"[A-Za-z]+", t)) == 1


ITEMS: list[dict[str, Any]] = [
    {"id": "recall_name", "category": "memory",
     "turns": _conv(("user", "My name is Priya and I have a dog called Biscuit."),
                    ("assistant", "Nice to meet you, Priya! Biscuit sounds adorable."),
                    ("user", "What is my dog's name?")),
     "check": lambda t: "biscuit" in t.lower(), "gold": "Your dog's name is Biscuit."},
    {"id": "correction", "category": "memory",
     "turns": _conv(("user", "I live in Paris."),
                    ("assistant", "Paris is a lovely city!"),
                    ("user", "Actually, I moved to Rome last week. Where do I live now?")),
     "check": lambda t: "rome" in t.lower(), "gold": "You live in Rome now."},
    {"id": "variable_chain", "category": "memory",
     "turns": _conv(("user", "Let x = 7."), ("assistant", "Okay, x = 7."),
                    ("user", "Now let y = x * 3. What is y + 1? Reply with just the number.")),
     "check": lambda t: re.search(r"\b22\b", t) is not None, "gold": "22"},
    {"id": "persona_suffix", "category": "system_prompt",
     "system": "You are a pirate. End every single reply with the exact word 'Arr!'",
     "turns": _conv(("user", "What is 2 + 2?")),
     "check": lambda t: t.strip().endswith("Arr!") and ("4" in t or "four" in t.lower()),
     "gold": "That be 4, matey. Arr!"},
    {"id": "one_word_persistent", "category": "system_prompt",
     "system": "Reply with exactly one word and nothing else, in every turn.",
     "turns": _conv(("user", "Name a fruit."), ("assistant", "Apple"),
                    ("user", "Name a color.")),
     "check": _one_word, "gold": "Blue"},
    {"id": "language_switch", "category": "instruction_carryover",
     "turns": _conv(("user", "For the rest of this chat answer in French only."),
                    ("assistant", "D'accord, je répondrai en français."),
                    ("user", "What is the capital of Italy?")),
     "check": lambda t: "rome" in t.lower() and re.search(
         r"\b(la capitale|est|c'est|capitale)\b", t.lower()) is not None
     and re.search(r"\b(the capital|is the)\b", t.lower()) is None,
     "gold": "La capitale de l'Italie est Rome."},
]


class MultiTurnTask(ItemTask):
    name = "multi_turn"
    aliases = ("chat", "multiturn")
    description = "conversation memory, corrections, system-prompt and instruction carry-over"
    items = interleave_by_category(ITEMS)
    max_tokens = 128

    def build_messages(self, item: dict[str, Any]) -> list[ChatMessage]:
        return [ChatMessage(**turn) for turn in item["turns"]]

    def grade(self, item: dict[str, Any], answer: str) -> Grade:
        return Grade(1.0 if item["check"](answer) else 0.0)

    def gold_response(self, item: dict[str, Any]) -> str:
        return item["gold"]
