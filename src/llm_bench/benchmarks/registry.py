"""Task registry and alias resolution."""

from __future__ import annotations

from llm_bench.benchmarks.tasks.base import BenchTask
from llm_bench.benchmarks.tasks.generation_speed import GenerationSpeedTask
from llm_bench.benchmarks.tasks.instruction_quality import InstructionQualityTask
from llm_bench.benchmarks.tasks.long_context import LongContextTask
from llm_bench.benchmarks.tasks.prompt_latency import PromptLatencyTask
from llm_bench.benchmarks.tasks.tool_calling import ToolCallingTask

_TASKS: dict[str, BenchTask] = {}
_ALIASES: dict[str, str] = {
    "speed": "generation_speed",
    "latency": "prompt_latency",
    "context": "long_context",
    "tools": "tool_calling",
    "quality": "instruction_quality",
}


def _register(task: BenchTask) -> None:
    _TASKS[task.name] = task


def _ensure_loaded() -> None:
    if _TASKS:
        return
    for task in (
        GenerationSpeedTask(),
        PromptLatencyTask(),
        LongContextTask(),
        ToolCallingTask(),
        InstructionQualityTask(),
    ):
        _register(task)


def list_tasks() -> list[str]:
    _ensure_loaded()
    return sorted(_TASKS.keys())


def get_task(name: str) -> BenchTask:
    _ensure_loaded()
    key = _ALIASES.get(name, name)
    if key not in _TASKS:
        raise KeyError(f"unknown task: {name}. Available: {', '.join(list_tasks())}")
    return _TASKS[key]


def resolve_tasks(names: list[str], *, tools_enabled: bool) -> list[BenchTask]:
    _ensure_loaded()
    resolved: list[BenchTask] = []
    seen: set[str] = set()
    for name in names:
        task = get_task(name)
        if task.name in seen:
            continue
        if task.requires_tools and not tools_enabled:
            continue
        seen.add(task.name)
        resolved.append(task)
    return resolved
