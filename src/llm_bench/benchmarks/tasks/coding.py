"""Code generation graded by executing unit tests (pass@1).

Model-written Python runs in a separate interpreter (``python -I``) inside a throw-away
temp directory with a hard timeout and CPU limit. This is *not* a security sandbox:
disable the task with ``--no-code-exec`` if you do not want model output executed.
"""

from __future__ import annotations

import asyncio
import json
import os
import re
import signal
import sys
import tempfile
from typing import Any

from llm_bench.benchmarks.tasks.common import response_metrics, split_thinking
from llm_bench.benchmarks.tasks.items import Grade, ItemTask, interleave_by_category
from llm_bench.schema import ChatMessage

INSTRUCTION = (
    "Write a Python 3 function as specified. Reply with a single ```python code block "
    "containing the function (imports allowed, no tests, no example usage)."
)

_RUNNER = r'''
import json, sys
payload = json.loads(sys.stdin.read())
ns = {"__name__": "candidate"}
results = []
try:
    exec(compile(payload["code"], "<candidate>", "exec"), ns)
    fn = ns[payload["entry"]]
except BaseException as exc:
    print("__RESULT__" + json.dumps({"load_error": repr(exc)[:200]}))
    sys.exit(0)
for args, expected in payload["tests"]:
    try:
        out = fn(*args)
        out = json.loads(json.dumps(out))
        if payload.get("sort_output") and isinstance(out, list):
            out = sorted(out)
        results.append(out == expected)
    except BaseException:
        results.append(False)
print("__RESULT__" + json.dumps({"results": results}))
'''


def _t(*args: Any, expect: Any) -> tuple[list[Any], Any]:
    return list(args), expect


ITEMS: list[dict[str, Any]] = [
    {"id": "palindrome", "category": "strings", "entry": "is_palindrome",
     "prompt": "Write `is_palindrome(s: str) -> bool` that returns True if `s` reads the same "
               "forwards and backwards, ignoring case and any non-alphanumeric characters.",
     "tests": [_t("A man, a plan, a canal: Panama", expect=True), _t("hello", expect=False),
               _t("", expect=True), _t("No 'x' in Nixon", expect=True), _t("ab", expect=False)],
     "reference": "def is_palindrome(s):\n    t = [c.lower() for c in s if c.isalnum()]\n"
                  "    return t == t[::-1]\n"},
    {"id": "fizzbuzz", "category": "basics", "entry": "fizzbuzz",
     "prompt": "Write `fizzbuzz(n: int) -> list[str]` returning the strings for 1..n: "
               "'Fizz' for multiples of 3, 'Buzz' for multiples of 5, 'FizzBuzz' for both, "
               "otherwise the number as a string.",
     "tests": [_t(5, expect=["1", "2", "Fizz", "4", "Buzz"]), _t(0, expect=[]),
               _t(15, expect=[str(i) if i % 3 and i % 5 else
                              ("FizzBuzz" if i % 15 == 0 else "Fizz" if i % 3 == 0 else "Buzz")
                              for i in range(1, 16)])],
     "reference": "def fizzbuzz(n):\n    out = []\n    for i in range(1, n + 1):\n"
                  "        out.append('FizzBuzz' if i % 15 == 0 else 'Fizz' if i % 3 == 0 "
                  "else 'Buzz' if i % 5 == 0 else str(i))\n    return out\n"},
    {"id": "flatten", "category": "lists", "entry": "flatten",
     "prompt": "Write `flatten(items: list) -> list` that flattens arbitrarily nested lists "
               "into a single flat list, preserving order.",
     "tests": [_t([1, [2, [3, [4]], 5]], expect=[1, 2, 3, 4, 5]), _t([], expect=[]),
               _t([[], [[]], [1]], expect=[1]), _t([["a"], "b"], expect=["a", "b"])],
     "reference": "def flatten(items):\n    out = []\n    for x in items:\n"
                  "        if isinstance(x, list):\n            out.extend(flatten(x))\n"
                  "        else:\n            out.append(x)\n    return out\n"},
    {"id": "wordcount", "category": "strings", "entry": "word_count",
     "prompt": "Write `word_count(text: str) -> dict[str, int]` that lower-cases the text, "
               "splits it on whitespace, and returns how many times each word occurs.",
     "tests": [_t("the cat The dog", expect={"the": 2, "cat": 1, "dog": 1}), _t("", expect={}),
               _t("a  a\ta\nb", expect={"a": 3, "b": 1})],
     "reference": "def word_count(text):\n    d = {}\n    for w in text.lower().split():\n"
                  "        d[w] = d.get(w, 0) + 1\n    return d\n"},
    {"id": "intervals", "category": "algorithms", "entry": "merge_intervals",
     "prompt": "Write `merge_intervals(intervals: list[list[int]]) -> list[list[int]]` that "
               "merges all overlapping (or touching) intervals and returns them sorted by start.",
     "tests": [_t([[1, 3], [2, 6], [8, 10], [15, 18]], expect=[[1, 6], [8, 10], [15, 18]]),
               _t([[1, 4], [4, 5]], expect=[[1, 5]]), _t([], expect=[]),
               _t([[5, 6], [1, 2], [2, 3]], expect=[[1, 3], [5, 6]])],
     "reference": "def merge_intervals(intervals):\n    out = []\n"
                  "    for a, b in sorted(intervals):\n"
                  "        if out and a <= out[-1][1]:\n"
                  "            out[-1][1] = max(out[-1][1], b)\n"
                  "        else:\n            out.append([a, b])\n    return out\n"},
    {"id": "roman", "category": "strings", "entry": "roman_to_int",
     "prompt": "Write `roman_to_int(s: str) -> int` converting a Roman numeral "
               "(I, V, X, L, C, D, M with subtractive notation) to an integer.",
     "tests": [_t("III", expect=3), _t("IV", expect=4), _t("IX", expect=9),
               _t("LVIII", expect=58), _t("MCMXCIV", expect=1994)],
     "reference": "def roman_to_int(s):\n"
                  "    v = {'I':1,'V':5,'X':10,'L':50,'C':100,'D':500,'M':1000}\n"
                  "    t = 0\n    for i, c in enumerate(s):\n"
                  "        if i + 1 < len(s) and v[c] < v[s[i + 1]]:\n            t -= v[c]\n"
                  "        else:\n            t += v[c]\n    return t\n"},
    {"id": "twosum", "category": "algorithms", "entry": "two_sum", "sort_output": True,
     "prompt": "Write `two_sum(nums: list[int], target: int) -> list[int]` returning the two "
               "distinct indices [i, j] (i < j) whose values add up to `target`. Exactly one "
               "solution exists.",
     "tests": [_t([2, 7, 11, 15], 9, expect=[0, 1]), _t([3, 2, 4], 6, expect=[1, 2]),
               _t([3, 3], 6, expect=[0, 1]), _t([-1, -2, -3, -4, -5], -8, expect=[2, 4])],
     "reference": "def two_sum(nums, target):\n    seen = {}\n"
                  "    for i, n in enumerate(nums):\n"
                  "        if target - n in seen:\n            return [seen[target - n], i]\n"
                  "        seen[n] = i\n"},
    {"id": "prefix", "category": "strings", "entry": "longest_common_prefix",
     "prompt": "Write `longest_common_prefix(strs: list[str]) -> str` returning the longest "
               "common prefix of all strings, or '' if there is none (or the list is empty).",
     "tests": [_t(["flower", "flow", "flight"], expect="fl"),
               _t(["dog", "racecar", "car"], expect=""), _t([], expect=""),
               _t(["same", "same"], expect="same"), _t(["a"], expect="a")],
     "reference": "def longest_common_prefix(strs):\n    if not strs:\n        return ''\n"
                  "    p = strs[0]\n    for s in strs[1:]:\n"
                  "        while not s.startswith(p):\n            p = p[:-1]\n"
                  "    return p\n"},
    {"id": "brackets", "category": "algorithms", "entry": "is_balanced",
     "prompt": "Write `is_balanced(s: str) -> bool` that returns True if all brackets "
               "()[]{} in `s` are correctly matched and nested. Other characters are ignored.",
     "tests": [_t("()[]{}", expect=True), _t("([{}])", expect=True), _t("(]", expect=False),
               _t("((", expect=False), _t("a(b)c", expect=True), _t(")(", expect=False),
               _t("", expect=True)],
     "reference": "def is_balanced(s):\n    pairs = {')':'(',']':'[','}':'{'}\n    st = []\n"
                  "    for c in s:\n        if c in '([{':\n            st.append(c)\n"
                  "        elif c in pairs:\n"
                  "            if not st or st.pop() != pairs[c]:\n                return False\n"
                  "    return not st\n"},
    {"id": "fib", "category": "algorithms", "entry": "fib",
     "prompt": "Write `fib(n: int) -> int` returning the n-th Fibonacci number with fib(0) = 0 "
               "and fib(1) = 1. It must be efficient enough for n = 80.",
     "tests": [_t(0, expect=0), _t(1, expect=1), _t(10, expect=55),
               _t(50, expect=12586269025), _t(80, expect=23416728348467685)],
     "reference": "def fib(n):\n    a, b = 0, 1\n    for _ in range(n):\n        a, b = b, a + b\n"
                  "    return a\n"},
]

_FENCE = re.compile(r"```(?:python|py|python3)?[ \t]*\n(.*?)```", re.S | re.I)


def extract_code(text: str, entry: str) -> str:
    """Pick the fenced block that defines ``entry`` (else the text itself)."""
    blocks = _FENCE.findall(text)
    with_entry = [b for b in blocks if re.search(rf"def\s+{re.escape(entry)}\b", b)]
    if with_entry:
        return max(with_entry, key=len)
    if blocks:
        return max(blocks, key=len)
    # unterminated fence (truncated output) or bare code
    open_fence = re.search(r"```(?:python|py)?[ \t]*\n(.*)\Z", text, re.S | re.I)
    return open_fence.group(1) if open_fence else text


def _limit_resources() -> None:  # pragma: no cover - runs in the child process
    try:
        import resource

        resource.setrlimit(resource.RLIMIT_CPU, (10, 10))
        resource.setrlimit(resource.RLIMIT_NOFILE, (64, 64))
    except Exception:
        pass


async def run_tests(code: str, item: dict[str, Any], *, timeout: float = 15.0) -> dict[str, Any]:
    """Execute ``code`` against the item's tests. Returns {'results': [...]} or {'error': ...}."""
    payload = json.dumps(
        {
            "code": code,
            "entry": item["entry"],
            "tests": item["tests"],
            "sort_output": bool(item.get("sort_output")),
        }
    )
    with tempfile.TemporaryDirectory(prefix="llm-bench-code-") as cwd:
        proc = await asyncio.create_subprocess_exec(
            sys.executable,
            "-I",
            "-c",
            _RUNNER,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.DEVNULL,
            cwd=cwd,
            env={"PATH": os.environ.get("PATH", ""), "PYTHONDONTWRITEBYTECODE": "1"},
            preexec_fn=_limit_resources if os.name == "posix" else None,
            start_new_session=True,
        )
        try:
            out, _ = await asyncio.wait_for(
                proc.communicate(payload.encode()), timeout=timeout
            )
        except asyncio.TimeoutError:
            _kill_group(proc)
            await proc.wait()
            return {"error": "timeout"}
        finally:
            if proc.returncode is None:
                _kill_group(proc)
    for line in reversed(out.decode(errors="replace").splitlines()):
        if line.startswith("__RESULT__"):
            data = json.loads(line[len("__RESULT__") :])
            if "load_error" in data:
                return {"error": f"load: {data['load_error']}"}
            return data
    return {"error": "no result (crashed or exited early)"}


def _kill_group(proc: asyncio.subprocess.Process) -> None:
    try:
        os.killpg(proc.pid, signal.SIGKILL)
    except (ProcessLookupError, PermissionError, OSError):
        try:
            proc.kill()
        except ProcessLookupError:
            pass


class CodingTask(ItemTask):
    name = "coding"
    aliases = ("code",)
    requires_code_exec = True
    description = "write Python functions; graded by running unit tests (pass@1)"
    items = interleave_by_category(ITEMS)
    max_tokens = 768
    timeout_floor = 180.0

    def build_messages(self, item):  # type: ignore[no-untyped-def]
        return [ChatMessage(role="user", content=f"{INSTRUCTION}\n\n{item['prompt']}")]

    def grade(self, item: dict[str, Any], answer: str) -> Grade:  # pragma: no cover
        raise NotImplementedError("coding grading is async; see run()")

    async def run(self, provider, model, config, *, run_index):  # type: ignore[no-untyped-def]
        item = self.items[run_index % len(self.items)]
        cat = item["category"]
        detail = {"item_id": item["id"], "category": cat}
        req, result = await self.respond(provider, model, config, item)
        if result.error:
            return self.make_sample(
                model, run_index, {"score": 0.0, "passed": False, f"score_{cat}": 0.0},
                error=result.error, detail=detail,
            )
        answer, thinking = split_thinking(result.text or "")
        code = extract_code(answer, item["entry"])
        outcome = await run_tests(code, item)
        results = outcome.get("results") or []
        total = len(item["tests"])
        passed_n = sum(1 for r in results if r)
        strict = 1.0 if total and passed_n == total else 0.0
        metrics: dict[str, Any] = {
            "score": strict,
            "passed": strict == 1.0,
            "test_pass_rate": round(passed_n / total, 4) if total else 0.0,
            f"score_{cat}": strict,
            "ran": "error" not in outcome,
            **response_metrics(result, req.max_tokens),
        }
        if thinking:
            metrics["thinking_chars"] = len(thinking)
        return self.make_sample(
            model, run_index, metrics,
            detail={**detail, "tests_passed": f"{passed_n}/{total}",
                    "exec_error": outcome.get("error"), "code": code[:600]},
        )

    def gold_response(self, item: dict[str, Any]) -> str:
        return f"```python\n{item['reference']}```"
