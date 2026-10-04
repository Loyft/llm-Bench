"""Sample host / model / KV memory during a benchmark run."""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from typing import Any

import httpx
import psutil

from llm_bench.schema import ProviderName


@dataclass
class MemorySample:
    t_s: float
    system_used_mb: float
    system_total_mb: float
    system_pct: float
    process_rss_mb: float
    model_ram_mb: float | None = None
    model_vram_mb: float | None = None
    kv_ram_mb: float | None = None
    context_ram_mb: float | None = None
    label: str = ""

    def as_dict(self) -> dict[str, Any]:
        return {
            "t_s": round(self.t_s, 3),
            "system_used_mb": round(self.system_used_mb, 2),
            "system_total_mb": round(self.system_total_mb, 2),
            "system_pct": round(self.system_pct, 2),
            "process_rss_mb": round(self.process_rss_mb, 2),
            "model_ram_mb": None if self.model_ram_mb is None else round(self.model_ram_mb, 2),
            "model_vram_mb": None if self.model_vram_mb is None else round(self.model_vram_mb, 2),
            "kv_ram_mb": None if self.kv_ram_mb is None else round(self.kv_ram_mb, 2),
            "context_ram_mb": None
            if self.context_ram_mb is None
            else round(self.context_ram_mb, 2),
            "label": self.label,
        }


@dataclass
class MemorySummary:
    peak_system_used_mb: float = 0.0
    peak_process_rss_mb: float = 0.0
    peak_model_ram_mb: float | None = None
    peak_model_vram_mb: float | None = None
    peak_kv_ram_mb: float | None = None
    peak_context_ram_mb: float | None = None
    baseline_process_rss_mb: float | None = None
    samples: list[dict[str, Any]] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return {
            "peak_system_used_mb": round(self.peak_system_used_mb, 2),
            "peak_process_rss_mb": round(self.peak_process_rss_mb, 2),
            "peak_model_ram_mb": self.peak_model_ram_mb,
            "peak_model_vram_mb": self.peak_model_vram_mb,
            "peak_kv_ram_mb": self.peak_kv_ram_mb,
            "peak_context_ram_mb": self.peak_context_ram_mb,
            "baseline_process_rss_mb": self.baseline_process_rss_mb,
            "samples": self.samples,
        }


def _bytes_to_mb(n: int | float | None) -> float | None:
    if n is None:
        return None
    return float(n) / (1024 * 1024)


def _provider_process_rss_mb(provider: ProviderName) -> float:
    """Sum RSS of processes likely owned by the active provider."""
    needles = {
        ProviderName.OLLAMA: ("ollama",),
        ProviderName.LLAMACPP: ("llama-server", "llama.cpp", "llama_cpp"),
        ProviderName.MLX: ("python", "mlx"),
        ProviderName.OPENAI: ("llama-server", "vllm", "lmstudio", "openai"),
    }.get(provider, ())
    total = 0.0
    for proc in psutil.process_iter(["name", "cmdline", "memory_info"]):
        try:
            name = (proc.info.get("name") or "").lower()
            cmd = " ".join(proc.info.get("cmdline") or []).lower()
            blob = f"{name} {cmd}"
            if any(n in blob for n in needles):
                mem = proc.info.get("memory_info")
                if mem is not None:
                    total += mem.rss / (1024 * 1024)
        except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
            continue
    return total


async def _ollama_model_memory(
    base_url: str, model_name: str | None
) -> tuple[float | None, float | None, int | None]:
    """Return (size_mb, vram_mb, context_length) from Ollama /api/ps."""
    try:
        async with httpx.AsyncClient(base_url=base_url.rstrip("/"), timeout=2.0) as client:
            r = await client.get("/api/ps")
            r.raise_for_status()
            models = r.json().get("models") or []
            chosen = None
            for m in models:
                name = m.get("name") or m.get("model") or ""
                if model_name is None or name == model_name or name.startswith(f"{model_name}"):
                    chosen = m
                    break
            if chosen is None and models:
                chosen = models[0]
            if not chosen:
                return None, None, None
            size = _bytes_to_mb(chosen.get("size"))
            vram = _bytes_to_mb(chosen.get("size_vram"))
            ctx = chosen.get("context_length")
            if ctx is None:
                details = chosen.get("details") or {}
                ctx = details.get("context_length")
            return size, vram, int(ctx) if ctx else None
    except Exception:
        return None, None, None


class MemoryMonitor:
    """Background sampler for RAM / model / KV estimates."""

    def __init__(
        self,
        *,
        provider: ProviderName,
        base_url: str | None = None,
        model_name: str | None = None,
        context_tokens: int = 8192,
        interval_s: float = 0.5,
    ) -> None:
        self.provider = provider
        self.base_url = base_url or "http://127.0.0.1:11434"
        self.model_name = model_name
        self.context_tokens = context_tokens
        self.interval_s = interval_s
        self._task: asyncio.Task | None = None
        self._running = False
        self._t0 = 0.0
        self._baseline_rss: float | None = None
        self._baseline_model: float | None = None
        self.samples: list[MemorySample] = []
        self._label = ""

    def set_label(self, label: str) -> None:
        self._label = label

    async def mark_baseline(self) -> None:
        sample = await self._capture(label="baseline")
        self._baseline_rss = sample.process_rss_mb
        self._baseline_model = sample.model_ram_mb
        self.samples.append(sample)

    async def _capture(self, label: str | None = None) -> MemorySample:
        vm = psutil.virtual_memory()
        process_rss = _provider_process_rss_mb(self.provider)
        model_ram = None
        model_vram = None
        ctx_len = None
        if self.provider == ProviderName.OLLAMA:
            model_ram, model_vram, ctx_len = await _ollama_model_memory(
                self.base_url, self.model_name
            )

        # KV ≈ process growth after model load, floored at 0
        kv = None
        if self._baseline_rss is not None:
            kv = max(0.0, process_rss - self._baseline_rss)
        elif model_ram is not None and process_rss > 0:
            kv = max(0.0, process_rss - model_ram)

        # Context RAM: scale KV by how much of the configured window is active,
        # or estimate from bytes/token if we only know model context length.
        context_ram = kv
        if kv is not None and ctx_len and self.context_tokens:
            # When KV is tiny early on, still expose a nominal context reservation estimate
            # (~0.5 MB per 1k tokens as a soft lower bound for reporting continuity).
            nominal = (self.context_tokens / 1000.0) * 0.5
            context_ram = max(kv, nominal) if kv > 0 else kv

        t_s = 0.0 if not self._t0 else time.perf_counter() - self._t0
        return MemorySample(
            t_s=t_s,
            system_used_mb=(vm.total - vm.available) / (1024 * 1024),
            system_total_mb=vm.total / (1024 * 1024),
            system_pct=vm.percent,
            process_rss_mb=process_rss,
            model_ram_mb=model_ram,
            model_vram_mb=model_vram,
            kv_ram_mb=kv,
            context_ram_mb=context_ram,
            label=label if label is not None else self._label,
        )

    async def _loop(self) -> None:
        while self._running:
            try:
                self.samples.append(await self._capture())
            except Exception:
                pass
            await asyncio.sleep(self.interval_s)

    async def start(self) -> None:
        self._t0 = time.perf_counter()
        self._running = True
        # First sample before baseline mark
        self.samples.append(await self._capture(label="start"))
        self._task = asyncio.create_task(self._loop())

    async def stop(self) -> MemorySummary:
        self._running = False
        if self._task is not None:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
            self._task = None
        try:
            self.samples.append(await self._capture(label="end"))
        except Exception:
            pass
        return self.summary()

    def summary(self) -> MemorySummary:
        if not self.samples:
            return MemorySummary()

        def _peak(attr: str) -> float | None:
            vals = [getattr(s, attr) for s in self.samples if getattr(s, attr) is not None]
            return max(vals) if vals else None

        return MemorySummary(
            peak_system_used_mb=max(s.system_used_mb for s in self.samples),
            peak_process_rss_mb=max(s.process_rss_mb for s in self.samples),
            peak_model_ram_mb=_peak("model_ram_mb"),
            peak_model_vram_mb=_peak("model_vram_mb"),
            peak_kv_ram_mb=_peak("kv_ram_mb"),
            peak_context_ram_mb=_peak("context_ram_mb"),
            baseline_process_rss_mb=self._baseline_rss,
            samples=[s.as_dict() for s in self.samples],
        )
