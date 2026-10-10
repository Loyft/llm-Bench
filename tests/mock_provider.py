"""Deterministic mock provider for offline tests."""

from __future__ import annotations

import time

from llm_bench.providers.base import BaseProvider, timed_chat_fields
from llm_bench.schema import ChatRequest, ChatResult, ModelRef, ProviderName


class MockProvider(BaseProvider):
    name = ProviderName.OPENAI

    def __init__(self, **kwargs) -> None:  # type: ignore[no-untyped-def]
        super().__init__(**kwargs)
        self.chat_calls = 0
        self.shutdown_calls: list[str] = []
        self.ready: list[str] = []

    async def ensure_ready(self, model: ModelRef) -> None:
        self.ready.append(model.id)
        self._mark_loaded(model)

    async def list_models(self) -> list[ModelRef]:
        return [ModelRef(id="mock-model", provider=self.name)]

    async def chat(self, req: ChatRequest) -> ChatResult:
        self.chat_calls += 1
        start = time.perf_counter()
        time.sleep(0.01)
        first = time.perf_counter()
        text = "323"
        user = req.messages[-1].content if req.messages else ""
        if "JSON" in user or "json" in user:
            text = '{"name": "Ada", "age": 36}'
        elif "add(a, b)" in user or "Python function" in user:
            text = "def add(a, b):\n    return a + b\n"
        elif "Photosynthesis" in user:
            text = "Photosynthesis converts light into chemical energy in plants."
        elif "weather" in user.lower() or req.tools:
            text = 'Calling get_weather with {"city": "Paris", "unit": "celsius"}'
            return timed_chat_fields(
                text=text,
                start=start,
                first_token_at=first,
                completion_tokens=12,
                prompt_tokens=20,
                tool_calls=[
                    {
                        "function": {
                            "name": "get_weather",
                            "arguments": '{"city": "Paris", "unit": "celsius"}',
                        }
                    }
                ],
            )
        elif "SECRET_CODE" in user:
            # echo needle if present
            import re

            m = re.search(r"SECRET_CODE_\d+_ALPHA", user)
            text = m.group(0) if m else "unknown"
        elif "Say OK" in user:
            text = "OK"
        else:
            text = "A CPU fetches, decodes, executes, and writes back instructions. " * 20

        time.sleep(0.01)
        prompt_tokens = max(1, len(user) // 4)
        completion_tokens = max(1, len(text.split()))
        # Simulate provider-native prefill timing (~2ms/token)
        prefill_ms = max(1.0, prompt_tokens * 2.0)
        return timed_chat_fields(
            text=text,
            start=start,
            first_token_at=first,
            completion_tokens=completion_tokens,
            prompt_tokens=prompt_tokens,
            prefill_ms=prefill_ms,
            prefill_tokens_per_sec=(prompt_tokens / prefill_ms) * 1000.0,
        )

    async def shutdown(self, model: ModelRef) -> None:
        self.shutdown_calls.append(model.id)
        await super().shutdown(model)

    async def aclose(self) -> None:
        return None


class OracleProvider(MockProvider):
    """Answers every capability task with its gold response (grader/harness self-test)."""

    TOOL_ARGS = {
        "weather": ("get_weather", {"city": "Paris", "unit": "celsius"}),
        "calc": ("calculator", {"expression": "1234 * 5678"}),
        "timer": ("set_timer", {"minutes": 15}),
        "email": ("send_email", {"to": "bob@example.com", "subject": "Lunch",
                                 "body": "Are we still on for noon?"}),
        "search": ("search_web", {"query": "Mars Sample Return news"}),
    }

    def __init__(self, **kwargs) -> None:  # type: ignore[no-untyped-def]
        super().__init__(**kwargs)
        from llm_bench.benchmarks.registry import task_catalog
        from llm_bench.benchmarks.tasks.items import ItemTask

        self.gold: dict[tuple, str] = {}
        for task in task_catalog():
            if isinstance(task, ItemTask):
                for item in task.items:
                    key = tuple((m.role, m.content) for m in task.build_messages(item))
                    self.gold[key] = task.gold_response(item)

    async def chat(self, req: ChatRequest) -> ChatResult:
        import json
        import re

        from llm_bench.benchmarks.tasks.tool_calling import SCENARIOS

        key = tuple((m.role, m.content) for m in req.messages)
        user = req.messages[-1].content
        if key in self.gold:
            return self._reply(self.gold[key])
        if req.tools:
            for sc in SCENARIOS:
                if sc["prompt"] == user:
                    if sc["tool"] is None:
                        return self._reply("Hello! I'm doing well.")
                    name, args = self.TOOL_ARGS[sc["id"]]
                    return timed_chat_fields(
                        text="",
                        start=time.perf_counter(),
                        first_token_at=time.perf_counter(),
                        completion_tokens=10,
                        prompt_tokens=30,
                        tool_calls=[{"function": {"name": name, "arguments": json.dumps(args)}}],
                    )
        m = re.search(r"vault access code is ([A-Z]+-\d+)", user)
        if m:
            return self._reply(m.group(1))
        return await super().chat(req)

    def _reply(self, text: str) -> ChatResult:
        start = time.perf_counter()
        return timed_chat_fields(
            text=text,
            start=start,
            first_token_at=time.perf_counter(),
            completion_tokens=max(1, len(text.split())),
            prompt_tokens=20,
        )
