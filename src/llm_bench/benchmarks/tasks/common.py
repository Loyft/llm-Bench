"""Shared metric helpers for benchmark tasks."""

from __future__ import annotations

from typing import Any

from llm_bench.schema import ChatResult


def chat_speed_metrics(result: ChatResult) -> dict[str, Any]:
    """Standard speed / prefill fields from a ChatResult."""
    return {
        "ttft_ms": result.ttft_ms,
        "total_ms": result.total_ms,
        "prefill_ms": result.prefill_ms,
        "decode_ms": result.decode_ms,
        "tokens_per_sec": result.tokens_per_sec,
        "decode_tokens_per_sec": result.decode_tokens_per_sec or result.tokens_per_sec,
        "prefill_tokens_per_sec": result.prefill_tokens_per_sec,
        "completion_tokens": result.completion_tokens,
        "prompt_tokens": result.prompt_tokens,
    }
