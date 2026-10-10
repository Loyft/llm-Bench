"""Task registry and alias resolution."""

from __future__ import annotations

from llm_bench.benchmarks.tasks.base import BenchTask
from llm_bench.benchmarks.tasks.coding import CodingTask
from llm_bench.benchmarks.tasks.consistency import ConsistencyTask
from llm_bench.benchmarks.tasks.generation_speed import GenerationSpeedTask
from llm_bench.benchmarks.tasks.hallucination import HallucinationTask
from llm_bench.benchmarks.tasks.instruction_quality import InstructionQualityTask
from llm_bench.benchmarks.tasks.instructions import InstructionFollowingTask
from llm_bench.benchmarks.tasks.knowledge import KnowledgeTask
from llm_bench.benchmarks.tasks.long_context import LongContextTask
from llm_bench.benchmarks.tasks.multi_turn import MultiTurnTask
from llm_bench.benchmarks.tasks.prompt_latency import PromptLatencyTask
from llm_bench.benchmarks.tasks.reasoning import ReasoningTask
from llm_bench.benchmarks.tasks.throughput import ThroughputTask
from llm_bench.benchmarks.tasks.tool_calling import ToolCallingTask

_TASKS: dict[str, BenchTask] = {}
_ALIASES: dict[str, str] = {}


def _register(task: BenchTask) -> None:
    _TASKS[task.name] = task
    _ALIASES[task.name] = task.name
    for alias in task.aliases:
        _ALIASES[alias] = task.name


def _ensure_loaded() -> None:
    if _TASKS:
        return
    for task in (
        GenerationSpeedTask(),
        PromptLatencyTask(),
        ThroughputTask(),
        LongContextTask(),
        ToolCallingTask(),
        InstructionQualityTask(),
        ReasoningTask(),
        KnowledgeTask(),
        CodingTask(),
        InstructionFollowingTask(),
        HallucinationTask(),
        MultiTurnTask(),
        ConsistencyTask(),
    ):
        _register(task)


def list_tasks() -> list[str]:
    _ensure_loaded()
    return sorted(_TASKS.keys())


def task_catalog() -> list[BenchTask]:
    """All tasks, performance first, in registration order."""
    _ensure_loaded()
    return list(_TASKS.values())


def aliases() -> dict[str, str]:
    _ensure_loaded()
    return {a: t for a, t in _ALIASES.items() if a != t}


def get_task(name: str) -> BenchTask:
    _ensure_loaded()
    key = _ALIASES.get(name, name)
    if key not in _TASKS:
        raise KeyError(f"unknown task: {name}. Available: {', '.join(list_tasks())}")
    return _TASKS[key]


def resolve_tasks(
    names: list[str],
    *,
    tools_enabled: bool,
    code_exec_enabled: bool = True,
) -> list[BenchTask]:
    _ensure_loaded()
    resolved: list[BenchTask] = []
    seen: set[str] = set()
    for name in names:
        task = get_task(name)
        if task.name in seen:
            continue
        if task.requires_tools and not tools_enabled:
            continue
        if task.requires_code_exec and not code_exec_enabled:
            continue
        seen.add(task.name)
        resolved.append(task)
    return resolved
