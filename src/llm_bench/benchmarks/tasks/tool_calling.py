"""Tool / JSON schema calling task."""

from __future__ import annotations

import json
import re

from llm_bench.benchmarks.tasks.base import BenchTask
from llm_bench.config import BenchConfig
from llm_bench.providers.base import BaseProvider
from llm_bench.schema import ChatMessage, ChatRequest, ModelRef, SampleResult, ToolSpec


class ToolCallingTask(BenchTask):
    name = "tool_calling"
    aliases = ("tools",)
    requires_tools = True

    async def run(
        self,
        provider: BaseProvider,
        model: ModelRef,
        config: BenchConfig,
        *,
        run_index: int,
    ) -> SampleResult:
        tool = ToolSpec(
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
        )
        prompt = (
            "What is the weather in Paris in celsius? "
            "Use the get_weather tool. Do not answer directly."
        )
        req = ChatRequest(
            model=model,
            messages=[ChatMessage(role="user", content=prompt)],
            max_tokens=256,
            temperature=0.0,
            seed=config.seed,
            tools=[tool],
            system=config.system
            or "You are a helpful assistant that uses tools when appropriate.",
            timeout=config.timeout,
        )
        result = await provider.chat(req)
        if result.error:
            return self.make_sample(
                model,
                run_index,
                {"success_rate": 0.0, "success": False},
                error=result.error,
            )

        success = False
        city_ok = False
        # Native tool calls
        for tc in result.tool_calls:
            fn = tc.get("function") or tc
            name = fn.get("name") or tc.get("name")
            if name == "get_weather":
                success = True
                args = fn.get("arguments") or tc.get("arguments") or {}
                if isinstance(args, str):
                    try:
                        args = json.loads(args)
                    except json.JSONDecodeError:
                        args = {}
                if isinstance(args, dict) and "paris" in str(args.get("city", "")).lower():
                    city_ok = True

        # Fallback: model emitted JSON in text
        text = result.text or ""
        if not success:
            if "get_weather" in text and "paris" in text.lower():
                success = True
                city_ok = True
            else:
                match = re.search(r"\{[^{}]*\"city\"[^{}]*\}", text, re.I)
                if match and "paris" in match.group(0).lower():
                    success = True
                    city_ok = True

        rate = 1.0 if success and city_ok else (0.5 if success else 0.0)
        return self.make_sample(
            model,
            run_index,
            {
                "success": success and city_ok,
                "success_rate": rate,
                "ttft_ms": result.ttft_ms,
                "total_ms": result.total_ms,
            },
            detail={"text_preview": text[:300], "tool_calls": result.tool_calls},
        )
