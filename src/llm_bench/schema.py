"""Shared result / request types for llm-bench."""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class ProviderName(str, Enum):
    OLLAMA = "ollama"
    LLAMACPP = "llamacpp"
    MLX = "mlx"
    OPENAI = "openai"


class ModelRef(BaseModel):
    """Reference to a model: name, local path, or URL."""

    id: str
    provider: ProviderName
    display_name: str | None = None
    base_url: str | None = None
    is_path: bool = False
    is_url: bool = False

    @property
    def label(self) -> str:
        return self.display_name or self.id


class ChatMessage(BaseModel):
    role: str
    content: str


class ToolSpec(BaseModel):
    name: str
    description: str = ""
    parameters: dict[str, Any] = Field(default_factory=dict)


class ChatRequest(BaseModel):
    model: ModelRef
    messages: list[ChatMessage]
    max_tokens: int = 256
    temperature: float = 0.0
    seed: int | None = 42
    tools: list[ToolSpec] | None = None
    system: str | None = None
    timeout: float = 120.0


class ChatResult(BaseModel):
    text: str = ""
    tool_calls: list[dict[str, Any]] = Field(default_factory=list)
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    total_tokens: int | None = None
    ttft_ms: float | None = None
    total_ms: float | None = None
    tokens_per_sec: float | None = None
    prefill_ms: float | None = None
    decode_ms: float | None = None
    prefill_tokens_per_sec: float | None = None
    decode_tokens_per_sec: float | None = None
    raw: dict[str, Any] = Field(default_factory=dict)
    error: str | None = None


class MemoryTimeline(BaseModel):
    peak_system_used_mb: float = 0.0
    peak_process_rss_mb: float = 0.0
    peak_model_ram_mb: float | None = None
    peak_model_vram_mb: float | None = None
    peak_kv_ram_mb: float | None = None
    peak_context_ram_mb: float | None = None
    baseline_process_rss_mb: float | None = None
    samples: list[dict[str, Any]] = Field(default_factory=list)


class SampleResult(BaseModel):
    task: str
    model: str
    provider: str
    run_index: int
    metrics: dict[str, float | int | str | bool | None] = Field(default_factory=dict)
    detail: dict[str, Any] = Field(default_factory=dict)
    error: str | None = None
    ts: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class TaskSummary(BaseModel):
    task: str
    model: str
    provider: str
    metrics: dict[str, float | int | str | bool | None] = Field(default_factory=dict)
    samples: int = 0
    errors: int = 0


class ModelScore(BaseModel):
    model: str
    provider: str
    overall: float
    breakdown: dict[str, float] = Field(default_factory=dict)
    tasks: dict[str, TaskSummary] = Field(default_factory=dict)


class HostInfo(BaseModel):
    os: str
    platform: str
    cpu: str
    ram_gb: float
    python: str
    gpu: str | None = None


class RunManifest(BaseModel):
    run_id: str
    started_at: datetime
    finished_at: datetime | None = None
    provider: ProviderName
    models: list[str]
    model_count: int
    tasks: list[str]
    context: int
    tools: bool
    mode: str
    runs: int
    warmup: int
    host: HostInfo
    config: dict[str, Any] = Field(default_factory=dict)


class BenchReport(BaseModel):
    manifest: RunManifest
    samples: list[SampleResult]
    summaries: list[TaskSummary]
    scores: list[ModelScore]
    mode: str  # "profile" | "compare"
    memory: MemoryTimeline = Field(default_factory=MemoryTimeline)

    @property
    def errors(self) -> list[SampleResult]:
        return [s for s in self.samples if s.error]

    @property
    def successful_samples(self) -> list[SampleResult]:
        return [s for s in self.samples if not s.error and s.task != "setup"]

    @property
    def ok(self) -> bool:
        """True when at least one real benchmark sample succeeded."""
        return bool(self.successful_samples)
