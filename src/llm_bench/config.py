"""CLI / YAML configuration models."""

from __future__ import annotations

from enum import Enum
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field, field_validator

from llm_bench.schema import ProviderName


class RunMode(str, Enum):
    SEQUENTIAL = "sequential"
    PARALLEL = "parallel"


class ToolsMode(str, Enum):
    ON = "on"
    OFF = "off"


ALL_TASKS = [
    "speed",
    "latency",
    "throughput",
    "context",
    "tools",
    "quality",
    "reasoning",
    "knowledge",
    "coding",
    "instructions",
    "hallucination",
    "multi_turn",
    "consistency",
]
PERFORMANCE_TASKS = ["speed", "latency", "throughput"]
CAPABILITY_TASKS = [t for t in ALL_TASKS if t not in PERFORMANCE_TASKS]
TASK_PRESETS = {
    "all": ALL_TASKS,
    "performance": PERFORMANCE_TASKS,
    "capability": CAPABILITY_TASKS,
}


class BenchConfig(BaseModel):
    provider: ProviderName = ProviderName.OLLAMA
    models: list[str] = Field(default_factory=list)
    base_url: str | None = None
    context: int = 8192
    tools: bool = True
    tasks: list[str] = Field(default_factory=lambda: list(ALL_TASKS))
    mode: RunMode = RunMode.SEQUENTIAL
    concurrency: int = 2
    runs: int = 3
    warmup: int = 1
    report_dir: Path = Path("./reports")
    workspace: Path | None = None
    keep_workspace: bool = False
    timeout: float = 120.0
    seed: int = 42
    system: str | None = None
    interactive: bool = False
    # Cap on items per capability task (None = full bank). Useful for quick runs.
    max_items: int | None = None
    # Coding task executes model-written Python in a subprocess; allow turning it off.
    code_exec: bool = True
    # Background music (with synced visuals) on the live run screen only.
    music: bool = True
    music_path: Path | None = None
    music_volume: float = Field(default=0.7, ge=0.0, le=1.0)

    @field_validator("models")
    @classmethod
    def at_least_one_model(cls, v: list[str]) -> list[str]:
        if not v:
            raise ValueError("at least one model is required")
        return v

    @field_validator("tasks", mode="before")
    @classmethod
    def parse_tasks(cls, v: Any) -> list[str]:
        if v is None:
            return list(ALL_TASKS)
        names = [t.strip() for t in v.split(",")] if isinstance(v, str) else list(v)
        out: list[str] = []
        for name in names:
            if not name:
                continue
            out.extend(TASK_PRESETS.get(name.lower(), [name]))
        return list(dict.fromkeys(out))

    @field_validator("max_items", mode="before")
    @classmethod
    def parse_max_items(cls, v: Any) -> int | None:
        if v in (None, "", 0, "0"):
            return None
        n = int(v)
        if n < 0:
            raise ValueError("max_items must be >= 0")
        return n

    @field_validator("tools", mode="before")
    @classmethod
    def parse_tools(cls, v: Any) -> bool:
        if isinstance(v, bool):
            return v
        if isinstance(v, str):
            return v.strip().lower() in {"on", "true", "1", "yes"}
        return bool(v)

    @field_validator("report_dir", "workspace", "music_path", mode="before")
    @classmethod
    def parse_path(cls, v: Any) -> Path | None:
        if v is None or v == "":
            return None
        return Path(v).expanduser()


def load_yaml_config(path: Path) -> dict[str, Any]:
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(data, dict):
        raise ValueError(f"config root must be a mapping: {path}")
    return data


def merge_config(
    yaml_data: dict[str, Any] | None = None,
    *,
    provider: str | None = None,
    models: list[str] | None = None,
    base_url: str | None = None,
    context: int | None = None,
    tools: str | bool | None = None,
    tasks: str | list[str] | None = None,
    mode: str | None = None,
    concurrency: int | None = None,
    runs: int | None = None,
    warmup: int | None = None,
    report: Path | str | None = None,
    workspace: Path | str | None = None,
    keep_workspace: bool | None = None,
    timeout: float | None = None,
    seed: int | None = None,
    system: str | None = None,
    max_items: int | None = None,
    code_exec: bool | None = None,
    music: bool | None = None,
    music_path: Path | str | None = None,
    music_volume: float | None = None,
) -> BenchConfig:
    data: dict[str, Any] = dict(yaml_data or {})
    # YAML may use report_dir; CLI uses --report
    if "report" in data and "report_dir" not in data:
        data["report_dir"] = data.pop("report")

    overrides: dict[str, Any] = {}
    if provider is not None:
        overrides["provider"] = provider
    if models is not None:
        overrides["models"] = models
    if base_url is not None:
        overrides["base_url"] = base_url
    if context is not None:
        overrides["context"] = context
    if tools is not None:
        overrides["tools"] = tools
    if tasks is not None:
        overrides["tasks"] = tasks
    if mode is not None:
        overrides["mode"] = mode
    if concurrency is not None:
        overrides["concurrency"] = concurrency
    if runs is not None:
        overrides["runs"] = runs
    if warmup is not None:
        overrides["warmup"] = warmup
    if report is not None:
        overrides["report_dir"] = report
    if workspace is not None:
        overrides["workspace"] = workspace
    if keep_workspace is not None:
        overrides["keep_workspace"] = keep_workspace
    if timeout is not None:
        overrides["timeout"] = timeout
    if seed is not None:
        overrides["seed"] = seed
    if system is not None:
        overrides["system"] = system
    if max_items is not None:
        overrides["max_items"] = max_items
    if code_exec is not None:
        overrides["code_exec"] = code_exec
    if music is not None:
        overrides["music"] = music
    if music_path is not None:
        overrides["music_path"] = music_path
    if music_volume is not None:
        overrides["music_volume"] = music_volume

    data.update({k: v for k, v in overrides.items() if v is not None})
    return BenchConfig.model_validate(data)
