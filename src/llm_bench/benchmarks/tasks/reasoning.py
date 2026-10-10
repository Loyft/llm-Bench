"""Multi-step reasoning: word problems, arithmetic and logic with a single checkable answer."""

from __future__ import annotations

import re
from typing import Any

from llm_bench.benchmarks.tasks.common import final_answer, last_number, parse_number
from llm_bench.benchmarks.tasks.items import Grade, ItemTask, interleave_by_category
from llm_bench.schema import ChatMessage

INSTRUCTION = (
    "Solve the problem. Reason briefly, then finish with a final line of the form "
    "'ANSWER: <answer>' containing only the answer (a number, or a single word for "
    "yes/no and name questions)."
)

# kind "num" compares numbers; kind "text" matches the word (singular/plural tolerant).
ITEMS: list[dict[str, Any]] = [
    {"id": "change", "category": "arithmetic", "kind": "num", "answer": 29,
     "prompt": "A shop sells pens at $3 each. Maria buys 7 pens and pays with a $50 bill. "
               "How many dollars of change does she get?"},
    {"id": "distance", "category": "arithmetic", "kind": "num", "answer": 270,
     "prompt": "A train travels at 60 km/h for 2.5 hours, then at 80 km/h for 1.5 hours. "
               "What is the total distance in km?"},
    {"id": "machines", "category": "word_problem", "kind": "num", "answer": 5,
     "prompt": "If 5 machines take 5 minutes to make 5 widgets, how many minutes would "
               "100 machines take to make 100 widgets?"},
    {"id": "batball", "category": "word_problem", "kind": "num", "answer": 5,
     "prompt": "A bat and a ball cost $1.10 in total. The bat costs $1.00 more than the ball. "
               "How many cents does the ball cost?"},
    {"id": "discount", "category": "arithmetic", "kind": "num", "answer": 75,
     "prompt": "A jacket was reduced by 20% and now costs $60. What was the original price "
               "in dollars?"},
    {"id": "ages", "category": "word_problem", "kind": "num", "answer": 20,
     "prompt": "Sam is twice as old as Tim. In 6 years the sum of their ages will be 42. "
               "How old is Sam now?"},
    {"id": "divisible", "category": "counting", "kind": "num", "answer": 22,
     "prompt": "How many positive integers less than 50 are divisible by 3 or by 5?"},
    {"id": "tank", "category": "word_problem", "kind": "num", "answer": 2,
     "prompt": "Pipe A fills a tank in 6 hours and pipe B fills it in 3 hours. With both "
               "pipes open, how many hours does it take to fill the tank?"},
    {"id": "multiples7", "category": "counting", "kind": "num", "answer": 735,
     "prompt": "What is the sum of all multiples of 7 from 1 to 100?"},
    {"id": "rectangle", "category": "arithmetic", "kind": "num", "answer": 72,
     "prompt": "A rectangle has a perimeter of 36 and its length is twice its width. "
               "What is its area?"},
    {"id": "syllogism", "category": "logic", "kind": "text", "answer": "yes",
     "prompt": "All bloops are razzies. All razzies are lazzies. Is it necessarily true that "
               "all bloops are lazzies? Answer yes or no."},
    {"id": "tallest", "category": "logic", "kind": "text", "answer": "bob",
     "prompt": "Alice is taller than Bob. Bob is taller than Carol. Dave is shorter than "
               "Carol. Who is the second tallest of the four?"},
    {"id": "boxes", "category": "logic", "kind": "text", "answer": "apples",
     "prompt": "Three boxes are labeled 'Apples', 'Oranges' and 'Mixed', and every label is "
               "wrong. You take one fruit from the box labeled 'Mixed' and it is an apple. "
               "What does the box labeled 'Mixed' actually contain? Answer with one word: "
               "apples, oranges or mixed."},
    {"id": "distractor", "category": "logic", "kind": "num", "answer": 2,
     "prompt": "I have 3 apples. I eat 2 pears and give 1 apple to a friend. "
               "How many apples do I have left?"},
    {"id": "sequence", "category": "pattern", "kind": "num", "answer": 42,
     "prompt": "What number comes next in the sequence 2, 6, 12, 20, 30, ...?"},
    {"id": "weekday", "category": "pattern", "kind": "text", "answer": "friday",
     "prompt": "Today is Wednesday. What day of the week will it be 100 days from now?"},
    {"id": "sheep", "category": "logic", "kind": "num", "answer": 9,
     "prompt": "A farmer has 17 sheep. All but 9 of them run away. How many sheep are left?"},
    {"id": "clock", "category": "pattern", "kind": "num", "answer": 7.5,
     "prompt": "What is the smaller angle, in degrees, between the hour and minute hands of "
               "a clock at 3:15?"},
]

_PLURAL_TOLERANT = {"apples": r"\bapples?\b"}


def _text_match(expected: str, answer: str) -> bool:
    pattern = _PLURAL_TOLERANT.get(expected, rf"\b{re.escape(expected)}\b")
    return re.search(pattern, answer.lower()) is not None


def check_answer(item: dict[str, Any], text: str) -> tuple[bool, str]:
    """Return (correct, extracted_answer)."""
    extracted = final_answer(text)
    if item["kind"] == "num":
        value = parse_number(extracted)
        if value is None:
            value = last_number(text)
        ok = value is not None and abs(value - float(item["answer"])) < 1e-6
        return ok, extracted
    return _text_match(item["answer"], extracted), extracted


class ReasoningTask(ItemTask):
    name = "reasoning"
    aliases = ("reason",)
    description = "multi-step word problems, arithmetic and logic (exact-answer match)"
    items = interleave_by_category(ITEMS)
    max_tokens = 768

    def build_messages(self, item):  # type: ignore[no-untyped-def]
        return [ChatMessage(role="user", content=f"{INSTRUCTION}\n\n{item['prompt']}")]

    def grade(self, item: dict[str, Any], answer: str) -> Grade:
        ok, extracted = check_answer(item, answer)
        return Grade(
            1.0 if ok else 0.0, detail={"extracted": extracted, "expected": item["answer"]}
        )

    def gold_response(self, item: dict[str, Any]) -> str:
        return f"Working it out step by step.\nANSWER: {item['answer']}"
