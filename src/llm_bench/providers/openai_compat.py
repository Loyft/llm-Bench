"""Generic OpenAI-compatible provider (LM Studio, vLLM, llama.cpp server, etc.)."""

from __future__ import annotations

import json
import time

import httpx

from llm_bench.providers.base import BaseProvider, timed_chat_fields
from llm_bench.schema import ChatRequest, ChatResult, ModelRef, ProviderName


class OpenAICompatProvider(BaseProvider):
    name = ProviderName.OPENAI

    def __init__(self, **kwargs) -> None:  # type: ignore[no-untyped-def]
        super().__init__(**kwargs)
        self.base_url = (self.base_url or "http://127.0.0.1:1234/v1").rstrip("/")
        if not self.base_url.endswith("/v1"):
            # Allow bare host; append /v1 if missing
            if not self.base_url.endswith("v1"):
                self.base_url = self.base_url + "/v1"
        self._client = httpx.AsyncClient(
            base_url=self.base_url,
            timeout=self.timeout,
            headers={"Authorization": "Bearer llm-bench"},
        )

    def _base_for(self, model: ModelRef) -> str:
        base = (model.base_url or self.base_url).rstrip("/")
        if base.endswith("/v1"):
            return base
        # llama.cpp often serves at /v1
        if base.endswith(":8080") or "localhost" in base or "127.0.0.1" in base:
            return base + "/v1" if not base.endswith("/v1") else base
        return base if base.endswith("/v1") else base + "/v1"

    def _model_id(self, model: ModelRef) -> str:
        if model.is_url:
            return "default"
        return model.id

    async def ensure_ready(self, model: ModelRef) -> None:
        base = self._base_for(model)
        try:
            async with httpx.AsyncClient(
                base_url=base,
                timeout=min(10.0, self.timeout),
                headers={"Authorization": "Bearer llm-bench"},
            ) as client:
                r = await client.get("/models")
                r.raise_for_status()
        except httpx.ConnectError as exc:
            raise RuntimeError(
                f"Cannot reach OpenAI-compatible server at {base}. "
                "Is it running (LM Studio / vLLM / llama-server)?"
            ) from exc
        except httpx.TimeoutException as exc:
            raise RuntimeError(f"Timed out connecting to {base}") from exc
        self._mark_loaded(model)

    async def list_models(self) -> list[ModelRef]:
        r = await self._client.get("/models")
        r.raise_for_status()
        data = r.json()
        items = data.get("data") or data.get("models") or []
        out: list[ModelRef] = []
        for m in items:
            mid = m.get("id") or m.get("name")
            if mid:
                out.append(ModelRef(id=mid, provider=self.name, display_name=mid))
        return out

    async def chat(self, req: ChatRequest) -> ChatResult:
        base = self._base_for(req.model)
        model_id = self._model_id(req.model)

        messages = []
        if req.system:
            messages.append({"role": "system", "content": req.system})
        for m in req.messages:
            messages.append({"role": m.role, "content": m.content})

        payload: dict = {
            "model": model_id,
            "messages": messages,
            "stream": True,
            "temperature": req.temperature,
            "max_tokens": req.max_tokens,
        }
        if req.seed is not None:
            payload["seed"] = req.seed
        if req.tools:
            payload["tools"] = [
                {
                    "type": "function",
                    "function": {
                        "name": t.name,
                        "description": t.description,
                        "parameters": t.parameters,
                    },
                }
                for t in req.tools
            ]

        start = time.perf_counter()
        first_token_at: float | None = None
        text_parts: list[str] = []
        tool_calls: list[dict] = []
        completion_tokens = None
        prompt_tokens = None

        try:
            async with httpx.AsyncClient(
                base_url=base,
                timeout=req.timeout,
                headers={"Authorization": "Bearer llm-bench"},
            ) as client:
                async with client.stream("POST", "/chat/completions", json=payload) as resp:
                    resp.raise_for_status()
                    async for line in resp.aiter_lines():
                        if not line or not line.startswith("data:"):
                            continue
                        data_str = line[len("data:") :].strip()
                        if data_str == "[DONE]":
                            break
                        data = json.loads(data_str)
                        choice = (data.get("choices") or [{}])[0]
                        delta = choice.get("delta") or {}
                        chunk = delta.get("content") or ""
                        if chunk:
                            if first_token_at is None:
                                first_token_at = time.perf_counter()
                            text_parts.append(chunk)
                        if delta.get("tool_calls"):
                            tool_calls.extend(delta["tool_calls"])
                        usage = data.get("usage") or {}
                        if usage:
                            completion_tokens = usage.get("completion_tokens", completion_tokens)
                            prompt_tokens = usage.get("prompt_tokens", prompt_tokens)

            # Estimate tokens if provider omitted usage
            text = "".join(text_parts)
            if completion_tokens is None and text:
                completion_tokens = max(1, len(text.split()))
            return timed_chat_fields(
                text=text,
                start=start,
                first_token_at=first_token_at,
                completion_tokens=completion_tokens,
                prompt_tokens=prompt_tokens,
                tool_calls=tool_calls,
            )
        except Exception as exc:
            return timed_chat_fields(
                text="".join(text_parts),
                start=start,
                first_token_at=first_token_at,
                completion_tokens=completion_tokens,
                prompt_tokens=prompt_tokens,
                tool_calls=tool_calls,
                error=str(exc),
            )

    async def shutdown(self, model: ModelRef) -> None:
        # External servers are not killed; we only drop our tracking.
        await super().shutdown(model)

    async def aclose(self) -> None:
        await self._client.aclose()
