"""General-knowledge multiple choice (MMLU-style, single letter answer)."""

from __future__ import annotations

import re
from typing import Any

from llm_bench.benchmarks.tasks.items import Grade, ItemTask, interleave_by_category
from llm_bench.schema import ChatMessage

INSTRUCTION = "Answer the multiple-choice question. Reply with only the letter (A, B, C or D)."


def _q(id_: str, cat: str, question: str, options: list[str], answer: str) -> dict[str, Any]:
    """Raw question; ``_layout`` later fixes option order so answers are position-balanced."""
    return {"id": id_, "category": cat, "question": question, "options": options, "answer": answer}


def _layout(raw: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Rotate options so the correct answer cycles A, B, C, D (removes positional bias)."""
    letters = "ABCD"
    out = []
    for i, q in enumerate(raw):
        opts = list(q["options"])
        correct = opts[letters.index(q["answer"])]
        shift = (letters.index(q["answer"]) - i % 4) % 4
        opts = opts[shift:] + opts[:shift]
        assert opts[i % 4] == correct
        body = "\n".join(f"{letters[j]}) {opt}" for j, opt in enumerate(opts))
        out.append(
            {
                "id": q["id"],
                "category": q["category"],
                "prompt": f"{q['question']}\n{body}",
                "answer": letters[i % 4],
                "max_tokens": 192,
            }
        )
    return out


_RAW: list[dict[str, Any]] = [
    _q("gold", "science", "What is the chemical symbol for gold?", ["Ag", "Au", "Gd", "Go"], "B"),
    _q("mars", "science", "Which planet is known as the Red Planet?",
       ["Venus", "Jupiter", "Mars", "Mercury"], "C"),
    _q("mito", "science", "Which organelle is the main site of ATP production in a cell?",
       ["Ribosome", "Mitochondrion", "Nucleus", "Golgi apparatus"], "B"),
    _q("light", "science", "Approximately how fast does light travel in a vacuum (m/s)?",
       ["3 x 10^6", "3 x 10^8", "3 x 10^10", "3 x 10^5"], "B"),
    _q("ph", "science", "What is the pH of pure water at 25 degrees Celsius?",
       ["0", "7", "10", "14"], "B"),
    _q("dna", "science", "In DNA, adenine pairs with which base?",
       ["Cytosine", "Guanine", "Thymine", "Uracil"], "C"),
    _q("force", "science", "Which is the SI unit of force?",
       ["Joule", "Watt", "Pascal", "Newton"], "D"),
    _q("canberra", "geography", "What is the capital of Australia?",
       ["Sydney", "Melbourne", "Canberra", "Perth"], "C"),
    _q("ocean", "geography", "Which is the largest ocean on Earth?",
       ["Atlantic", "Indian", "Arctic", "Pacific"], "D"),
    _q("nile", "geography", "Which river flows through Cairo?",
       ["Nile", "Tigris", "Congo", "Niger"], "A"),
    _q("everest", "geography", "Mount Everest is part of which mountain range?",
       ["Andes", "Alps", "Himalayas", "Rockies"], "C"),
    _q("ww2", "history", "In which year did World War II end?",
       ["1943", "1945", "1947", "1950"], "B"),
    _q("austen", "history", "Who wrote the novel 'Pride and Prejudice'?",
       ["Charlotte Bronte", "Jane Austen", "George Eliot", "Mary Shelley"], "B"),
    _q("president", "history", "Who was the first President of the United States?",
       ["Thomas Jefferson", "John Adams", "George Washington", "Benjamin Franklin"], "C"),
    _q("cpu", "technology", "What does CPU stand for?",
       ["Central Processing Unit", "Computer Personal Unit", "Central Program Utility",
        "Core Processing Utility"], "A"),
    _q("binary", "technology", "What is the binary number 1011 in decimal?",
       ["9", "10", "11", "13"], "C"),
    _q("bsearch", "technology", "What is the time complexity of binary search on a sorted array?",
       ["O(n)", "O(log n)", "O(n log n)", "O(1)"], "B"),
    _q("lifo", "technology", "Which data structure follows last-in, first-out (LIFO) order?",
       ["Queue", "Stack", "Heap", "Linked list"], "B"),
    _q("deriv", "math", "What is the derivative of x^2 with respect to x?",
       ["x", "2", "2x", "x^2"], "C"),
    _q("prime", "math", "Which of these numbers is prime?", ["21", "27", "29", "33"], "C"),
    _q("ephemeral", "language", "Which word is closest in meaning to 'ephemeral'?",
       ["Lasting", "Fleeting", "Ancient", "Heavy"], "B"),
    _q("inflation", "language", "In economics, inflation is best described as:",
       ["a general rise in price levels", "a fall in unemployment",
        "an increase in exports", "a decrease in interest rates"], "A"),
]

ITEMS: list[dict[str, Any]] = _layout(_RAW)

_LEADING = re.compile(r"^\s*\(?([A-Da-d])(?:[).:]|\s*$)")
_STATED = re.compile(r"(?i:answer)\s*(?:(?i:is)|:)?\s*\W*([ABCD])\b")
# Explicit forms only: a bare capital "A" in prose is far more likely an article than an answer.
_EXPLICIT = re.compile(
    r"\(([A-Da-d])\)|\b([A-D])\)|\b(?:option|choice|letter)\s+\(?([A-D])\b"
    r"|\b(?:is|be|pick|choose|select)\s+\(?([A-D])\b(?!\s+[a-z])"
)


def extract_choice(text: str) -> str | None:
    text = text.replace("*", "").replace("_", "").strip()
    for pat in (_STATED, _LEADING):
        m = pat.search(text)
        if m:
            return m.group(1).upper()
    found = [g for m in _EXPLICIT.finditer(text) for g in m.groups() if g]
    return found[-1].upper() if found else None


class KnowledgeTask(ItemTask):
    name = "knowledge"
    aliases = ("know", "mcq")
    description = "multiple-choice general knowledge across science, geography, history, tech"
    items = interleave_by_category(ITEMS)
    max_tokens = 192

    def build_messages(self, item):  # type: ignore[no-untyped-def]
        return [ChatMessage(role="user", content=f"{INSTRUCTION}\n\n{item['prompt']}")]

    def grade(self, item: dict[str, Any], answer: str) -> Grade:
        choice = extract_choice(answer)
        return Grade(
            1.0 if choice == item["answer"] else 0.0,
            extra={"answered": choice is not None},
            detail={"extracted": choice, "expected": item["answer"]},
        )

    def gold_response(self, item: dict[str, Any]) -> str:
        return item["answer"]
