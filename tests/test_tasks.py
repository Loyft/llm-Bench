"""Task banks, graders, parsers, and a full oracle run through the real runner."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest
from mock_provider import MockProvider, OracleProvider

from llm_bench.benchmarks.registry import resolve_tasks, task_catalog
from llm_bench.benchmarks.tasks import coding, hallucination, knowledge, reasoning
from llm_bench.benchmarks.tasks.common import (
    final_answer,
    parse_number,
    split_thinking,
    strip_thinking,
)
from llm_bench.benchmarks.tasks.consistency import ConsistencyTask
from llm_bench.benchmarks.tasks.instructions import InstructionFollowingTask
from llm_bench.benchmarks.tasks.items import ItemTask
from llm_bench.benchmarks.tasks.multi_turn import MultiTurnTask
from llm_bench.benchmarks.tasks.tool_calling import _extract_calls, _text_fallback
from llm_bench.config import BenchConfig
from llm_bench.providers.base import merge_tool_call_deltas
from llm_bench.report.html import write_html_report
from llm_bench.report.terminal import render_terminal_report
from llm_bench.runner import BenchRunner
from llm_bench.schema import ProviderName

ITEM_TASKS = [t for t in task_catalog() if isinstance(t, ItemTask) and t.name != "coding"]


# ---- bank integrity -------------------------------------------------------------------


@pytest.mark.parametrize("task", [t for t in task_catalog() if isinstance(t, ItemTask)],
                         ids=lambda t: t.name)
def test_item_ids_unique_and_categorised(task: ItemTask) -> None:
    ids = [it["id"] for it in task.items]
    assert len(ids) == len(set(ids)), "duplicate item ids"
    assert all(it.get("category") for it in task.items)


@pytest.mark.parametrize("task", ITEM_TASKS, ids=lambda t: t.name)
def test_gold_response_scores_full_marks(task: ItemTask) -> None:
    for item in task.items:
        grade = task.grade(item, task.gold_response(item))
        assert grade.score == 1.0, f"{task.name}/{item['id']}: {grade}"


def test_knowledge_answers_are_valid_letters() -> None:
    for it in knowledge.ITEMS:
        assert it["answer"] in "ABCD"
        assert it["prompt"].count("\n") == 4  # question + 4 options


def test_knowledge_answer_positions_are_balanced() -> None:
    from collections import Counter

    counts = Counter(it["answer"] for it in knowledge.ITEMS)
    assert max(counts.values()) <= 0.4 * len(knowledge.ITEMS)


@pytest.mark.asyncio
@pytest.mark.parametrize("item", coding.ITEMS, ids=lambda i: i["id"])
async def test_coding_reference_solutions_pass(item: dict) -> None:
    out = await coding.run_tests(item["reference"], item)
    assert out.get("results") == [True] * len(item["tests"]), out


@pytest.mark.asyncio
async def test_coding_rejects_wrong_slow_and_broken_code() -> None:
    item = next(i for i in coding.ITEMS if i["id"] == "fib")
    wrong = await coding.run_tests("def fib(n):\n    return n\n", item)
    assert not all(wrong["results"])
    slow = await coding.run_tests(
        "def fib(n):\n    return n if n < 2 else fib(n-1) + fib(n-2)\n", item, timeout=3.0
    )
    assert slow.get("error") == "timeout" or not all(slow.get("results", [False]))
    broken = await coding.run_tests("def fib(n:\n", item)
    assert "error" in broken
    missing = await coding.run_tests("def other(n):\n    return n\n", item)
    assert "error" in missing


# ---- graders: negative cases ----------------------------------------------------------


def test_reasoning_rejects_wrong_and_accepts_formats() -> None:
    item = next(i for i in reasoning.ITEMS if i["id"] == "change")
    assert reasoning.check_answer(item, "work...\nANSWER: $29")[0]
    assert reasoning.check_answer(item, "so she gets 29 dollars")[0]  # fallback: last number
    assert not reasoning.check_answer(item, "ANSWER: 30")[0]
    text_item = next(i for i in reasoning.ITEMS if i["id"] == "boxes")
    assert reasoning.check_answer(text_item, "ANSWER: Apples.")[0]
    assert not reasoning.check_answer(text_item, "ANSWER: oranges")[0]
    half = next(i for i in reasoning.ITEMS if i["id"] == "clock")
    assert reasoning.check_answer(half, "ANSWER: 7.5")[0]


def test_extract_choice_variants() -> None:
    ec = knowledge.extract_choice
    assert ec("B") == "B"
    assert ec("(c) Canberra") == "C"
    assert ec("The answer is D.") == "D"
    assert ec("Answer: A") == "A"
    assert ec("I think it might be b") is None  # lowercase prose is not a letter answer
    assert ec("A common mistake... the answer is C") == "C"
    assert ec("It must be B") == "B"
    assert ec("I'd go with option D, because") == "D"
    # prose that merely starts with the article "A" must not be read as choice A
    assert ec("A CPU fetches, decodes, executes, and writes back instructions.") is None
    assert ec("A stack is LIFO.") is None
    assert ec("The answer is a stack") is None


def test_hallucination_requires_abstaining_but_not_on_controls() -> None:
    task = hallucination.HallucinationTask()
    fake = next(i for i in task.items if i["id"] == "fake_paper")
    assert task.grade(fake, "It was published in Nature as 'Quantum Lemon Heat'.").score == 0.0
    assert task.grade(fake, "I'm not aware of any such paper.").score == 1.0
    assert task.grade(fake, "I\u2019m not aware of that paper.").score == 1.0
    tokyo = next(i for i in task.items if i["id"] == "tokyo")
    refused = task.grade(tokyo, "I don't know.")
    assert refused.score == 0.0 and refused.extra["over_refused"] is True
    assert task.grade(tokyo, "Tokyo.").score == 1.0
    ein = next(i for i in task.items if i["id"] == "einstein")
    assert task.grade(ein, "He won it for the photoelectric effect, not relativity.").score == 1.0
    assert task.grade(ein, "Because relativity changed physics.").score == 0.0


def test_instruction_following_partial_credit() -> None:
    task = InstructionFollowingTask()
    item = next(i for i in task.items if i["id"] == "no_commas")
    # 20+ words but contains a comma -> 1 of 2 constraints
    g = task.grade(item, ("word " * 25).strip() + ", end")
    assert g.score == 0.5 and g.extra["all_constraints_met"] is False
    assert task.grade(item, "short, text").score == 0.0
    j = next(i for i in task.items if i["id"] == "json_only")
    fenced = '```json\n{"city": "Paris", "country": "France"}\n```'
    assert task.grade(j, fenced).score == 0.0
    assert task.grade(j, '{"city": "Lyon", "country": "France"}').score == 0.0


def test_multi_turn_rejects_wrong_behaviour() -> None:
    task = MultiTurnTask()
    by_id = {i["id"]: i for i in task.items}
    assert task.grade(by_id["recall_name"], "I don't remember.").score == 0.0
    assert task.grade(by_id["language_switch"], "The capital of Italy is Rome.").score == 0.0
    assert task.grade(by_id["persona_suffix"], "It is 4.").score == 0.0
    assert task.grade(by_id["one_word_persistent"], "Blue is a nice color").score == 0.0


# ---- parsers --------------------------------------------------------------------------


def test_split_thinking() -> None:
    assert split_thinking("<think>hmm</think>42") == ("42", "<think>hmm</think>")
    ans, th = split_thinking("<think>never finished")
    assert ans == "" and "never finished" in th
    assert strip_thinking("plain") == "plain"


def test_number_and_answer_extraction() -> None:
    assert parse_number("about $1,234.50 total") == 1234.5
    assert parse_number("none") is None
    assert final_answer("blah\nANSWER: 42.\nthanks") == "42"
    assert final_answer("one\ntwo") == "two"


def test_extract_code_prefers_block_with_entry() -> None:
    text = "```python\nprint('hi')\n```\nthen\n```python\ndef f(x):\n    return x\n```"
    assert "def f" in coding.extract_code(text, "f")
    unterminated = "```python\ndef f(x):\n    return x\n"
    assert "def f" in coding.extract_code(unterminated, "f")


def test_tool_call_normalisation_and_stream_merge() -> None:
    deltas = [
        {"index": 0, "id": "c1", "function": {"name": "set_timer", "arguments": '{"min'}},
        {"index": 0, "function": {"arguments": 'utes": 15}'}},
    ]
    merged = merge_tool_call_deltas(deltas)
    assert len(merged) == 1
    ((name, args, valid),) = _extract_calls(merged)
    assert (name, args, valid) == ("set_timer", {"minutes": 15}, True)
    ((_, _, valid),) = _extract_calls([{"function": {"name": "x", "arguments": "{oops"}}])
    assert valid is False
    assert _text_fallback('Sure: {"name": "calculator", "arguments": {"expression": "1+1"}}',
                          {"calculator"})[0][0] == "calculator"


# ---- config / registry ----------------------------------------------------------------


def test_task_presets_and_default_cover_everything() -> None:
    all_names = {t.name for t in resolve_tasks(BenchConfig(models=["m"]).tasks, tools_enabled=True)}
    assert all_names == {t.name for t in task_catalog()}
    perf = resolve_tasks(BenchConfig(models=["m"], tasks="performance").tasks, tools_enabled=True)
    assert {t.kind for t in perf} == {"performance"}
    cap = resolve_tasks(BenchConfig(models=["m"], tasks="capability").tasks, tools_enabled=True)
    assert {t.kind for t in cap} == {"capability"}


def test_tools_and_code_exec_gates() -> None:
    names = ["tools", "coding", "reasoning"]
    got = [t.name for t in resolve_tasks(names, tools_enabled=False, code_exec_enabled=False)]
    assert got == ["reasoning"]


def test_max_items_caps_sample_count_but_stays_balanced() -> None:
    cfg = BenchConfig(models=["m"], max_items=4)
    task = reasoning.ReasoningTask()
    assert task.sample_count(cfg) == 4
    assert task.sample_count(BenchConfig(models=["m"])) == len(reasoning.ITEMS)
    first = {i["category"] for i in task.items[:4]}
    assert len(first) == 4  # round-robin across categories


# ---- end to end -----------------------------------------------------------------------


@pytest.mark.asyncio
async def test_oracle_run_scores_every_capability_perfectly(tmp_path: Path) -> None:
    config = BenchConfig(
        provider=ProviderName.OPENAI,
        models=["oracle"],
        tasks="all",
        tools=True,
        runs=3,
        warmup=0,
        report_dir=tmp_path / "reports",
        workspace=tmp_path / "ws",
    )
    mock = OracleProvider()
    with patch("llm_bench.runner.get_provider", return_value=mock):
        runner = BenchRunner(config)
        report = await runner.run()

    assert report.ok and not report.errors
    (score,) = report.scores
    expected_keys = {"reasoning", "coding", "knowledge", "instructions", "tools", "context",
                     "hallucination", "multi_turn", "consistency", "quality", "speed", "prefill"}
    assert expected_keys <= set(score.breakdown)
    for key in expected_keys - {"speed", "prefill"}:
        assert score.breakdown[key] == 1.0, key
    assert score.capability == 1.0
    assert score.performance is not None and score.reliability == 1.0

    # percentile stats exist for multi-sample speed tasks
    speed = score.tasks["generation_speed"].metrics
    assert "tokens_per_sec_p50" in speed and "tokens_per_sec_p95" in speed
    # throughput scales across the three concurrency levels
    thr = score.tasks["throughput"].metrics
    assert all(f"agg_tps_c{n}" in thr for n in (1, 2, 4))
    # long-context sub-scores by depth
    ctx = score.tasks["long_context"].metrics
    assert {"score_depth_10", "score_depth_50", "score_depth_90"} <= set(ctx)
    # category sub-scores recorded
    assert "score_arithmetic" in score.tasks["reasoning"].metrics

    # reports render
    dest = tmp_path / "r.html"
    write_html_report(report, dest)
    html = dest.read_text(encoding="utf-8")
    for needle in ("Capability breakdown", "Speed detail", "Run health", "Per-task results",
                   "Reasoning", "Sub-scores by category"):
        assert needle in html
    render_terminal_report(report)
    runner.workspace.cleanup()


@pytest.mark.asyncio
async def test_errors_count_as_zero_not_as_missing(tmp_path: Path) -> None:
    class Flaky(OracleProvider):
        n = 0

        async def chat(self, req):  # type: ignore[no-untyped-def]
            Flaky.n += 1
            if Flaky.n % 2 == 0:
                return self._reply("").model_copy(update={"error": "boom"})
            return await super().chat(req)

    config = BenchConfig(
        provider=ProviderName.OPENAI, models=["flaky"], tasks=["knowledge"], warmup=0,
        runs=1, report_dir=tmp_path / "r", workspace=tmp_path / "ws",
    )
    with patch("llm_bench.runner.get_provider", return_value=Flaky()):
        report = await BenchRunner(config).run()
    (score,) = report.scores
    # half the requests failed: score must be ~0.5, not 1.0 from the surviving half
    assert 0.4 <= score.breakdown["knowledge"] <= 0.6
    assert score.reliability < 0.6


@pytest.mark.asyncio
async def test_wrong_answers_are_reported(tmp_path: Path) -> None:
    config = BenchConfig(
        provider=ProviderName.OPENAI, models=["dumb"], tasks=["knowledge", "reasoning"],
        warmup=0, runs=1, max_items=3, report_dir=tmp_path / "r", workspace=tmp_path / "ws",
    )
    # plain mock answers "323" to everything -> nothing parses as a correct answer
    with patch("llm_bench.runner.get_provider", return_value=MockProvider()):
        report = await BenchRunner(config).run()
    from llm_bench.report.view import failures

    assert report.scores[0].capability == 0.0
    assert len(failures(report)) == 6


def test_consistency_uses_known_questions() -> None:
    task = ConsistencyTask()
    assert {i["id"] for i in task.items} == {"change", "machines", "ages", "tallest", "sheep"}
