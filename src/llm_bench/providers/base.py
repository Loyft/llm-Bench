"""Provider protocol and shared helpers."""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from pathlib import Path
from urllib.parse import urlparse

from llm_bench.schema import ChatRequest, ChatResult, ModelRef, ProviderName
from llm_bench.workspace import Workspace


def looks_like_url(value: str) -> bool:
    parsed = urlparse(value)
    return parsed.scheme in {"http", "https"} and bool(parsed.netloc)


def looks_like_path(value: str) -> bool:
    if looks_like_url(value):
        return False
    p = Path(value)
    return value.startswith(("./", "/", "~", "../")) or p.suffix.lower() in {
        ".gguf",
        ".bin",
        ".safetensors",
    }


def resolve_model_ref(
    value: str,
    provider: ProviderName,
    *,
    base_url: str | None = None,
) -> ModelRef:
    is_url = looks_like_url(value)
    is_path = looks_like_path(value)
    model_base = base_url
    model_id = value
    if is_url and provider in {ProviderName.OLLAMA, ProviderName.LLAMACPP, ProviderName.OPENAI}:
        # URL alone means server endpoint; model id may still be needed separately.
        model_base = value.rstrip("/")
        model_id = value
    if is_path:
        model_id = str(Path(value).expanduser().resolve())
    return ModelRef(
        id=model_id,
        provider=provider,
        display_name=Path(model_id).name if is_path else model_id,
        base_url=model_base,
        is_path=is_path,
        is_url=is_url,
    )


class BaseProvider(ABC):
    name: ProviderName

    def __init__(
        self,
        *,
        base_url: str | None = None,
        workspace: Workspace | None = None,
        timeout: float = 120.0,
    ) -> None:
        self.base_url = base_url
        self.workspace = workspace
        self.timeout = timeout
        self._loaded: set[str] = set()

    @abstractmethod
    async def ensure_ready(self, model: ModelRef) -> None:
        """Load / verify model is reachable. Register cleanup if we start something."""

    @abstractmethod
    async def chat(self, req: ChatRequest) -> ChatResult:
        """Streaming chat; populate timing metrics."""

    @abstractmethod
    async def list_models(self) -> list[ModelRef]:
        """List available models when possible."""

    async def shutdown(self, model: ModelRef) -> None:
        """Unload model / stop owned resources for this model."""
        self._loaded.discard(model.id)

    async def aclose(self) -> None:
        """Close shared clients."""

    def _mark_loaded(self, model: ModelRef) -> None:
        self._loaded.add(model.id)
        if self.workspace is not None:

            async def _unload() -> None:
                await self.shutdown(model)

            # Register sync wrapper for atexit path
            def _sync_unload() -> None:
                import asyncio

                try:
                    loop = asyncio.get_event_loop()
                    if loop.is_running():
                        loop.create_task(self.shutdown(model))
                    else:
                        loop.run_until_complete(self.shutdown(model))
                except RuntimeError:
                    asyncio.run(self.shutdown(model))

            self.workspace.register_cleanup(_sync_unload)


def merge_tool_call_deltas(deltas: list[dict]) -> list[dict]:
    """Merge OpenAI-style streamed tool_call fragments (keyed by ``index``) into whole calls.

    Fragments without an ``index`` are treated as complete calls and passed through.
    """
    merged: dict[int, dict] = {}
    complete: list[dict] = []
    for d in deltas:
        if "index" not in d:
            complete.append(d)
            continue
        slot = merged.setdefault(
            d["index"], {"id": None, "type": "function", "function": {"name": "", "arguments": ""}}
        )
        if d.get("id"):
            slot["id"] = d["id"]
        fn = d.get("function") or {}
        if fn.get("name"):
            slot["function"]["name"] += fn["name"]
        if fn.get("arguments"):
            slot["function"]["arguments"] += fn["arguments"]
    return complete + [merged[i] for i in sorted(merged)]


def percentile(values: list[float], pct: float) -> float | None:
    """Linear-interpolated percentile (pct in 0..100)."""
    if not values:
        return None
    xs = sorted(values)
    if len(xs) == 1:
        return xs[0]
    k = (len(xs) - 1) * pct / 100.0
    lo = int(k)
    hi = min(lo + 1, len(xs) - 1)
    return xs[lo] + (xs[hi] - xs[lo]) * (k - lo)


def itl_stats(chunk_times: list[float] | None) -> tuple[float | None, float | None, float | None]:
    """(mean, p95, max) gaps in ms between consecutive streamed chunks."""
    if not chunk_times or len(chunk_times) < 2:
        return None, None, None
    gaps = [(b - a) * 1000.0 for a, b in zip(chunk_times, chunk_times[1:])]
    return sum(gaps) / len(gaps), percentile(gaps, 95), max(gaps)


def compute_prefill_tps(
    prompt_tokens: int | None,
    *,
    prefill_ms: float | None = None,
    ttft_ms: float | None = None,
) -> float | None:
    """Prefill tok/s from provider prefill duration, else TTFT fallback."""
    duration_ms = prefill_ms if prefill_ms and prefill_ms > 0 else ttft_ms
    if not prompt_tokens or not duration_ms or duration_ms <= 0:
        return None
    return (prompt_tokens / duration_ms) * 1000.0


def timed_chat_fields(
    *,
    text: str,
    start: float,
    first_token_at: float | None,
    completion_tokens: int | None,
    prompt_tokens: int | None = None,
    tool_calls: list | None = None,
    raw: dict | None = None,
    error: str | None = None,
    prefill_ms: float | None = None,
    decode_ms: float | None = None,
    prefill_tokens_per_sec: float | None = None,
    decode_tokens_per_sec: float | None = None,
    chunk_times: list[float] | None = None,
) -> ChatResult:
    end = time.perf_counter()
    itl_mean, itl_p95, itl_max = itl_stats(chunk_times)
    total_ms = (end - start) * 1000.0
    ttft_ms = ((first_token_at - start) * 1000.0) if first_token_at else None
    if prefill_ms is None and ttft_ms is not None:
        prefill_ms = ttft_ms
    if decode_ms is None and total_ms is not None:
        decode_ms = max(0.0, total_ms - (prefill_ms or ttft_ms or 0.0))
        if decode_ms <= 0:
            decode_ms = total_ms

    if prefill_tokens_per_sec is None:
        prefill_tokens_per_sec = compute_prefill_tps(
            prompt_tokens, prefill_ms=prefill_ms, ttft_ms=ttft_ms
        )
    tps = decode_tokens_per_sec
    if tps is None and completion_tokens and decode_ms and decode_ms > 0:
        tps = (completion_tokens / decode_ms) * 1000.0
    return ChatResult(
        text=text,
        tool_calls=tool_calls or [],
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        total_tokens=(prompt_tokens or 0) + (completion_tokens or 0)
        if prompt_tokens is not None or completion_tokens is not None
        else None,
        ttft_ms=ttft_ms,
        total_ms=total_ms,
        tokens_per_sec=tps,
        prefill_ms=prefill_ms,
        decode_ms=decode_ms,
        prefill_tokens_per_sec=prefill_tokens_per_sec,
        decode_tokens_per_sec=tps,
        itl_ms_mean=itl_mean,
        itl_ms_p95=itl_p95,
        itl_ms_max=itl_max,
        raw=raw or {},
        error=error,
    )
