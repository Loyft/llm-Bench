"""Shared metric helpers for benchmark tasks."""

from __future__ import annotations

import re
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
        "itl_ms_mean": result.itl_ms_mean,
        "itl_ms_p95": result.itl_ms_p95,
        "itl_ms_max": result.itl_ms_max,
    }


def response_metrics(result: ChatResult, max_tokens: int) -> dict[str, Any]:
    """Light-weight per-response stats attached to every graded sample."""
    ct = result.completion_tokens
    return {
        "ttft_ms": result.ttft_ms,
        "total_ms": result.total_ms,
        "tokens_per_sec": result.tokens_per_sec,
        "completion_tokens": ct,
        "prompt_tokens": result.prompt_tokens,
        # Hit the token budget: the answer may have been cut off mid-thought.
        "truncated": bool(ct is not None and ct >= max_tokens),
        "empty": not (result.text or "").strip() and not result.tool_calls,
    }


_THINK_BLOCK = re.compile(r"<think(?:ing)?>.*?</think(?:ing)?>", re.S | re.I)
_THINK_OPEN = re.compile(r"<think(?:ing)?>.*\Z", re.S | re.I)


def split_thinking(text: str) -> tuple[str, str]:
    """Return (answer, thinking) for reasoning models that emit <think> blocks.

    An unterminated <think> (generation cut off while thinking) yields an empty answer.
    """
    text = text or ""
    thinking = "".join(_THINK_BLOCK.findall(text))
    answer = _THINK_BLOCK.sub("", text)
    open_match = _THINK_OPEN.search(answer)
    if open_match:
        thinking += open_match.group(0)
        answer = answer[: open_match.start()]
    return answer.strip(), thinking


def strip_thinking(text: str) -> str:
    return split_thinking(text)[0]


_NUMBER = re.compile(r"-?\d[\d,]*(?:\.\d+)?")


def parse_number(text: str) -> float | None:
    """First number in ``text`` (handles $, thousands separators)."""
    m = _NUMBER.search(text.replace("$", ""))
    if not m:
        return None
    try:
        return float(m.group(0).replace(",", ""))
    except ValueError:
        return None


def last_number(text: str) -> float | None:
    matches = _NUMBER.findall(text.replace("$", ""))
    if not matches:
        return None
    try:
        return float(matches[-1].replace(",", ""))
    except ValueError:
        return None


_ANSWER_LINE = re.compile(r"answer\s*[:=\-]\s*(.+)", re.I)


def final_answer(text: str) -> str:
    """Content after the last ``ANSWER:`` marker, else the final non-empty line."""
    matches = _ANSWER_LINE.findall(text)
    if matches:
        return matches[-1].strip().strip("*_` ").rstrip(".")
    lines = [ln.strip() for ln in text.strip().splitlines() if ln.strip()]
    return lines[-1] if lines else ""
