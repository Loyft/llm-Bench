"""Tool / function calling: selection, argument accuracy, and knowing when not to call."""

from __future__ import annotations

import json
import re
from collections.abc import Callable
from typing import Any

from llm_bench.benchmarks.tasks.base import BenchTask
from llm_bench.benchmarks.tasks.common import response_metrics
from llm_bench.config import BenchConfig
from llm_bench.providers.base import BaseProvider
from llm_bench.schema import ChatMessage, ChatRequest, ModelRef, SampleResult, ToolSpec

TOOLS = [
    ToolSpec(
        name="get_weather",
        description="Get the current weather for a city",
        parameters={
            "type": "object",
            "properties": {
                "city": {"type": "string"},
                "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]},
            },
            "required": ["city"],
        },
    ),
    ToolSpec(
        name="calculator",
        description="Evaluate an arithmetic expression",
        parameters={
            "type": "object",
            "properties": {"expression": {"type": "string"}},
            "required": ["expression"],
        },
    ),
    ToolSpec(
        name="set_timer",
        description="Start a countdown timer",
        parameters={
            "type": "object",
            "properties": {"minutes": {"type": "integer"}},
            "required": ["minutes"],
        },
    ),
    ToolSpec(
        name="send_email",
        description="Send an email",
        parameters={
            "type": "object",
            "properties": {
                "to": {"type": "string"},
                "subject": {"type": "string"},
                "body": {"type": "string"},
            },
            "required": ["to", "subject", "body"],
        },
    ),
    ToolSpec(
        name="search_web",
        description="Search the web for up-to-date information",
        parameters={
            "type": "object",
            "properties": {"query": {"type": "string"}},
            "required": ["query"],
        },
    ),
]

ArgCheck = Callable[[dict[str, Any]], bool]


def _s(args: dict[str, Any], key: str) -> str:
    return str(args.get(key, "")).lower()


SCENARIOS: list[dict[str, Any]] = [
    {"id": "weather", "category": "selection", "tool": "get_weather",
     "prompt": "What is the weather in Paris in celsius? Use the get_weather tool.",
     "args_ok": lambda a: "paris" in _s(a, "city") and _s(a, "unit") in {"celsius", ""}},
    {"id": "calc", "category": "selection", "tool": "calculator",
     "prompt": "What is 1234 * 5678? Use the calculator tool.",
     "args_ok": lambda a: "1234" in _s(a, "expression") and "5678" in _s(a, "expression")},
    {"id": "timer", "category": "arguments", "tool": "set_timer",
     "prompt": "Set a timer for 15 minutes.",
     "args_ok": lambda a: str(a.get("minutes")).strip() in {"15", "15.0"},
     "strict_types": {"minutes": int}},
    {"id": "email", "category": "arguments", "tool": "send_email",
     "prompt": "Email bob@example.com with the subject 'Lunch' and tell him: "
               "Are we still on for noon?",
     "args_ok": lambda a: "bob@example.com" in _s(a, "to") and "lunch" in _s(a, "subject")
     and "noon" in _s(a, "body")},
    {"id": "search", "category": "selection", "tool": "search_web",
     "prompt": "Look up the latest news about the Mars Sample Return mission.",
     "args_ok": lambda a: "mars" in _s(a, "query")},
    {"id": "no_tool", "category": "restraint", "tool": None,
     "prompt": "Hi! How are you today? Just say hello back, no tools needed.",
     "args_ok": lambda a: True},
]


def _extract_calls(tool_calls: list[dict[str, Any]]) -> list[tuple[str, dict[str, Any], bool]]:
    """Normalise provider tool calls -> [(name, args, args_were_valid_json)]."""
    out = []
    for tc in tool_calls:
        fn = tc.get("function") or tc
        name = fn.get("name") or tc.get("name")
        args = fn.get("arguments", tc.get("arguments", {}))
        valid = True
        if isinstance(args, str):
            try:
                args = json.loads(args) if args.strip() else {}
            except json.JSONDecodeError:
                args, valid = {}, False
        if not isinstance(args, dict):
            args, valid = {}, False
        if name:
            out.append((str(name), args, valid))
    return out


def _text_fallback(text: str, known: set[str]) -> list[tuple[str, dict[str, Any], bool]]:
    """Models without native tool support often print the call as JSON in text."""
    for m in re.finditer(r"\{.*\}", text, re.S):
        try:
            data = json.loads(m.group(0))
        except json.JSONDecodeError:
            continue
        if isinstance(data, dict):
            name = data.get("name") or data.get("tool") or data.get("function")
            args = data.get("arguments") or data.get("parameters") or data.get("args") or {}
            if isinstance(name, str) and name in known and isinstance(args, dict):
                return [(name, args, True)]
    return []


class ToolCallingTask(BenchTask):
    name = "tool_calling"
    aliases = ("tools",)
    requires_tools = True
    description = "tool selection, argument accuracy (types!), and not calling tools when unneeded"
    kind = "capability"
    scenarios = SCENARIOS

    def sample_count(self, config: BenchConfig) -> int:
        n = len(self.scenarios)
        return min(n, config.max_items) if config.max_items else n

    async def run(
        self,
        provider: BaseProvider,
        model: ModelRef,
        config: BenchConfig,
        *,
        run_index: int,
    ) -> SampleResult:
        sc = self.scenarios[run_index % len(self.scenarios)]
        cat = sc["category"]
        max_tokens = 256
        req = ChatRequest(
            model=model,
            messages=[ChatMessage(role="user", content=sc["prompt"])],
            max_tokens=max_tokens,
            temperature=0.0,
            seed=config.seed,
            tools=TOOLS,
            system=config.system or "You are a helpful assistant that uses tools when appropriate.",
            timeout=config.timeout,
        )
        result = await provider.chat(req)
        base_detail = {"scenario": sc["id"], "category": cat}
        if result.error:
            return self.make_sample(
                model, run_index,
                {"score": 0.0, "success_rate": 0.0, "success": False, f"score_{cat}": 0.0},
                error=result.error, detail=base_detail,
            )

        text = result.text or ""
        calls = _extract_calls(result.tool_calls)
        native = bool(calls)
        if not calls:
            calls = _text_fallback(text, {t.name for t in TOOLS})

        expected = sc["tool"]
        if expected is None:
            # restraint: any tool call is a failure
            score = 1.0 if not calls else 0.0
            extra = {"called_tool": bool(calls)}
        else:
            match = next((c for c in calls if c[0] == expected), None)
            selected = match is not None
            args_ok = bool(match and match[2] and sc["args_ok"](match[1]))
            if args_ok and match:
                for key, typ in (sc.get("strict_types") or {}).items():
                    v = match[1].get(key)
                    if isinstance(v, bool) or not isinstance(v, typ):
                        args_ok = False
            # selecting the right tool is half the credit, correct arguments the rest
            score = (0.5 if selected else 0.0) + (0.5 if args_ok else 0.0)
            extra = {
                "tool_selected": selected,
                "args_correct": args_ok,
                "valid_json_args": bool(match[2]) if match else False,
                "native_tool_call": native,
            }

        return self.make_sample(
            model,
            run_index,
            {
                "score": score,
                "success_rate": score,
                "success": score >= 0.999,
                f"score_{cat}": score,
                **response_metrics(result, max_tokens),
                **extra,
            },
            detail={**base_detail, "text_preview": text[:300],
                    "calls": [{"name": n, "args": a} for n, a, _ in calls]},
        )
