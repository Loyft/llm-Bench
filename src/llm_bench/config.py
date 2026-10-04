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


class BenchConfig(BaseModel):
    provider: ProviderName = ProviderName.OLLAMA
    models: list[str] = Field(default_factory=list)
    base_url: str | None = None
    context: int = 8192
    tools: bool = True
    tasks: list[str] = Field(default_factory=lambda: ["speed", "quality", "tools", "context"])
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
            return ["speed", "quality", "tools", "context"]
        if isinstance(v, str):
            if v.strip().lower() == "all":
                return ["speed", "latency", "context", "tools", "quality"]
            return [t.strip() for t in v.split(",") if t.strip()]
        return list(v)

    @field_validator("tools", mode="before")
    @classmethod
    def parse_tools(cls, v: Any) -> bool:
        if isinstance(v, bool):
            return v
        if isinstance(v, str):
            return v.strip().lower() in {"on", "true", "1", "yes"}
        return bool(v)

    @field_validator("report_dir", "workspace", mode="before")
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

    data.update({k: v for k, v in overrides.items() if v is not None})
    return BenchConfig.model_validate(data)
