"""IFEval-style instruction following: every constraint is verified programmatically."""

from __future__ import annotations

import json
import re
from collections.abc import Callable
from typing import Any

from llm_bench.benchmarks.tasks.items import Grade, ItemTask, interleave_by_category

Check = tuple[str, Callable[[str], bool]]


def _words(t: str) -> list[str]:
    return re.findall(r"[A-Za-z0-9'’-]+", t)


def _sentences(t: str) -> list[str]:
    return [s for s in re.split(r"(?<=[.!?])\s+", t.strip()) if s.strip()]


def no_commas() -> Check:
    return "no commas", lambda t: "," not in t


def min_words(n: int) -> Check:
    return f">= {n} words", lambda t: len(_words(t)) >= n


def max_words(n: int) -> Check:
    return f"<= {n} words", lambda t: 0 < len(_words(t)) <= n


def all_lower() -> Check:
    return "all lowercase", lambda t: t == t.lower() and any(c.isalpha() for c in t)


def all_upper() -> Check:
    return "ALL CAPS", lambda t: t == t.upper() and any(c.isalpha() for c in t)


def ends_with(phrase: str) -> Check:
    return f"ends with '{phrase}'", lambda t: t.strip().endswith(phrase)


def contains_words(*words: str) -> Check:
    return (
        f"contains {', '.join(words)}",
        lambda t: all(re.search(rf"\b{re.escape(w)}\b", t, re.I) for w in words),
    )


def forbids_word(word: str) -> Check:
    return f"never uses '{word}'", lambda t: not re.search(rf"\b{re.escape(word)}\b", t, re.I)


def sentence_count(n: int) -> Check:
    return f"exactly {n} sentences", lambda t: len(_sentences(t)) == n


def bullet_count(n: int) -> Check:
    def fn(t: str) -> bool:
        lines = [ln for ln in t.strip().splitlines() if ln.strip()]
        return len(lines) == n and all(re.match(r"\s*[-*]\s+\S", ln) for ln in lines)

    return f"exactly {n} '- ' bullets and nothing else", fn


def numbered_count(n: int) -> Check:
    def fn(t: str) -> bool:
        lines = [ln for ln in t.strip().splitlines() if ln.strip()]
        return len(lines) == n and all(
            re.match(rf"\s*{i}[.)]\s+\S", ln) for i, ln in enumerate(lines, start=1)
        )

    return f"exactly {n} numbered lines and nothing else", fn


def exact(expected: str) -> Check:
    return f"exactly '{expected}'", lambda t: t.strip() == expected


def json_with(keys: list[str], values: dict[str, str] | None = None) -> Check:
    def fn(t: str) -> bool:
        try:
            data = json.loads(t.strip())
        except (json.JSONDecodeError, ValueError):
            return False
        if not isinstance(data, dict) or not all(k in data for k in keys):
            return False
        return all(
            str(data.get(k, "")).strip().lower() == v.lower() for k, v in (values or {}).items()
        )

    return "bare valid JSON with required keys", fn


def has_title() -> Check:
    return "title in <<double angle brackets>>", lambda t: re.search(r"<<[^<>\n]+>>", t) is not None


def paragraphs(n: int, sep: str = "***") -> Check:
    def fn(t: str) -> bool:
        parts = [p for p in t.split(sep)]
        return len(parts) == n and all(p.strip() for p in parts)

    return f"{n} paragraphs separated by {sep}", fn


def postscript() -> Check:
    return "has a P.S.", lambda t: re.search(r"(^|\n)\s*P\.S\.", t) is not None


ITEMS: list[dict[str, Any]] = [
    {"id": "no_commas", "category": "format",
     "prompt": "Write a short paragraph (at least 20 words) about autumn. Do not use any commas.",
     "checks": [no_commas(), min_words(20)],
     "gold": "Autumn arrives quietly as the leaves turn gold and red across the hills and the "
             "air grows crisp and cool while families gather indoors for warm meals and long "
             "evenings by the fire."},
    {"id": "bullets", "category": "format",
     "prompt": "List exactly 3 benefits of regular exercise as markdown bullet points, each "
               "starting with '- '. Output nothing except the bullets.",
     "checks": [bullet_count(3)],
     "gold": "- Stronger heart\n- Better mood\n- Improved sleep"},
    {"id": "lowercase", "category": "casing",
     "prompt": "Write two sentences about the ocean. Your entire response must be in lowercase "
               "letters, with no capital letters at all.",
     "checks": [all_lower(), sentence_count(2)],
     "gold": "the ocean is vast and deep. it covers most of our planet."},
    {"id": "uppercase_short", "category": "casing",
     "prompt": "Explain what a variable is in programming. Use ALL CAPITAL LETTERS and at most "
               "40 words.",
     "checks": [all_upper(), max_words(40)],
     "gold": "A VARIABLE IS A NAMED STORAGE LOCATION THAT HOLDS A VALUE WHICH CAN CHANGE WHILE "
             "A PROGRAM RUNS."},
    {"id": "max_words", "category": "length",
     "prompt": "Describe your ideal weekend in at most 30 words.",
     "checks": [max_words(30)],
     "gold": "Sleeping in, a long walk by the lake, good coffee, and an evening with friends."},
    {"id": "three_sentences", "category": "length",
     "prompt": "Write a story about a robot in exactly 3 sentences. Include the words 'lantern' "
               "and 'river'.",
     "checks": [sentence_count(3), contains_words("lantern", "river")],
     "gold": "A robot named Bolt carried a lantern to the river. He watched the water glow "
             "beneath the stars. Then he smiled and walked home."},
    {"id": "end_phrase", "category": "structure",
     "prompt": "Give two tips for learning a language. Finish your response with the exact "
               "phrase 'Is there anything else I can help with?' and nothing after it.",
     "checks": [ends_with("Is there anything else I can help with?")],
     "gold": "Practice daily and speak with natives.\nIs there anything else I can help with?"},
    {"id": "json_only", "category": "format",
     "prompt": "Reply with only a JSON object (no markdown fences, no commentary) with the keys "
               "\"city\" and \"country\" describing the capital of France.",
     "checks": [json_with(["city", "country"], {"city": "Paris", "country": "France"})],
     "gold": '{"city": "Paris", "country": "France"}'},
    {"id": "repeat_exact", "category": "precision",
     "prompt": "Reply with the word banana repeated exactly 5 times, separated by single "
               "spaces, and nothing else.",
     "checks": [exact("banana banana banana banana banana")],
     "gold": "banana banana banana banana banana"},
    {"id": "title", "category": "structure",
     "prompt": "Write a title wrapped in double angle brackets, like <<My Title>>, followed by "
               "one sentence about space.",
     "checks": [has_title(), min_words(8)],
     "gold": "<<Beyond the Sky>>\nSpace is a vast and silent expanse filled with stars."},
    {"id": "paragraphs", "category": "structure",
     "prompt": "Write exactly two paragraphs about tea. Separate them with the markdown divider "
               "*** on its own and write nothing else.",
     "checks": [paragraphs(2)],
     "gold": "Tea is brewed from the leaves of Camellia sinensis.\n***\nIt is enjoyed hot or "
             "cold around the world."},
    {"id": "numbered", "category": "format",
     "prompt": "List 4 colors as a numbered list (1. 2. 3. 4.) with one color per line and no "
               "other text.",
     "checks": [numbered_count(4)],
     "gold": "1. Red\n2. Blue\n3. Green\n4. Yellow"},
    {"id": "forbidden_word", "category": "precision",
     "prompt": "Describe a sunrise in 2-3 sentences without ever using the word 'the' "
               "(in any capitalization).",
     "checks": [forbids_word("the"), min_words(8)],
     "gold": "A golden glow spreads over distant hills. Birds begin to sing as warm light "
             "fills our quiet valley."},
    {"id": "postscript", "category": "structure",
     "prompt": "Write a short thank-you note to a teacher. End with a postscript that starts "
               "with 'P.S.'",
     "checks": [postscript(), min_words(15)],
     "gold": "Dear Ms. Lee, thank you for believing in me and making every lesson an adventure "
             "this year.\nP.S. I will miss your class."},
]


class InstructionFollowingTask(ItemTask):
    name = "instruction_following"
    aliases = ("instructions", "ifeval")
    description = "IFEval-style verifiable constraints (length, casing, format, forbidden words)"
    items = interleave_by_category(ITEMS)
    max_tokens = 384

    def grade(self, item: dict[str, Any], answer: str) -> Grade:
        results = {name: bool(fn(answer)) for name, fn in item["checks"]}
        frac = sum(results.values()) / len(results)
        return Grade(
            frac,
            extra={"all_constraints_met": frac == 1.0},
            detail={"checks": results},
        )

    def gold_response(self, item: dict[str, Any]) -> str:
        return item["gold"]
