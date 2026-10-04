"""Ollama provider adapter."""

from __future__ import annotations

import json
import time

import httpx

from llm_bench.providers.base import BaseProvider, timed_chat_fields
from llm_bench.schema import ChatRequest, ChatResult, ModelRef, ProviderName


class OllamaProvider(BaseProvider):
    name = ProviderName.OLLAMA

    def __init__(self, **kwargs) -> None:  # type: ignore[no-untyped-def]
        super().__init__(**kwargs)
        self.base_url = (self.base_url or "http://127.0.0.1:11434").rstrip("/")
        self._client = httpx.AsyncClient(base_url=self.base_url, timeout=self.timeout)

    def _endpoint_for(self, model: ModelRef) -> str:
        return (model.base_url or self.base_url).rstrip("/")

    async def ensure_ready(self, model: ModelRef) -> None:
        url = self._endpoint_for(model)
        try:
            async with httpx.AsyncClient(base_url=url, timeout=min(10.0, self.timeout)) as client:
                r = await client.get("/api/tags")
                r.raise_for_status()
                names = {m.get("name") for m in r.json().get("models", [])}
                if model.is_url:
                    # URL-only: just verify server is up
                    self._mark_loaded(model)
                    return
                model_name = model.id
                if model_name not in names and f"{model_name}:latest" not in names:
                    # Try a lightweight show to confirm
                    show = await client.post("/api/show", json={"name": model_name})
                    if show.status_code >= 400:
                        available = ", ".join(sorted(n for n in names if n)[:12]) or "(none)"
                        raise RuntimeError(
                            f"ollama model not found: {model_name}. "
                            f"Available: {available}"
                        )
        except httpx.ConnectError as exc:
            raise RuntimeError(
                f"Cannot reach Ollama at {url}. "
                "Is the server running? Start it with: ollama serve"
            ) from exc
        except httpx.TimeoutException as exc:
            raise RuntimeError(
                f"Timed out connecting to Ollama at {url}. "
                "Is the server starting or overloaded?"
            ) from exc
        self._mark_loaded(model)

    async def list_models(self) -> list[ModelRef]:
        r = await self._client.get("/api/tags")
        r.raise_for_status()
        out: list[ModelRef] = []
        for m in r.json().get("models", []):
            name = m.get("name")
            if name:
                out.append(ModelRef(id=name, provider=self.name, display_name=name))
        return out

    async def chat(self, req: ChatRequest) -> ChatResult:
        url = self._endpoint_for(req.model)
        model_name = req.model.id
        if req.model.is_url:
            raise RuntimeError("ollama chat requires a model name, not just a server URL")

        messages = []
        if req.system:
            messages.append({"role": "system", "content": req.system})
        for m in req.messages:
            messages.append({"role": m.role, "content": m.content})

        payload: dict = {
            "model": model_name,
            "messages": messages,
            "stream": True,
            "options": {
                "temperature": req.temperature,
                "num_predict": req.max_tokens,
            },
        }
        if req.seed is not None:
            payload["options"]["seed"] = req.seed
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
        prompt_tokens = None
        completion_tokens = None
        eval_count = None
        prompt_eval_count = None
        prompt_eval_duration_ns = None
        eval_duration_ns = None
        done_raw: dict = {}

        try:
            async with httpx.AsyncClient(base_url=url, timeout=req.timeout) as client:
                async with client.stream("POST", "/api/chat", json=payload) as resp:
                    resp.raise_for_status()
                    async for line in resp.aiter_lines():
                        if not line:
                            continue
                        data = json.loads(line)
                        msg = data.get("message") or {}
                        chunk = msg.get("content") or ""
                        if chunk:
                            if first_token_at is None:
                                first_token_at = time.perf_counter()
                            text_parts.append(chunk)
                        if msg.get("tool_calls"):
                            tool_calls.extend(msg["tool_calls"])
                        if data.get("done"):
                            done_raw = data
                            eval_count = data.get("eval_count")
                            prompt_eval_count = data.get("prompt_eval_count")
                            prompt_eval_duration_ns = data.get("prompt_eval_duration")
                            eval_duration_ns = data.get("eval_duration")
            completion_tokens = eval_count
            prompt_tokens = prompt_eval_count
            prefill_ms = None
            decode_ms = None
            prefill_tps = None
            decode_tps = None
            if prompt_eval_duration_ns and prompt_eval_duration_ns > 0:
                prefill_ms = prompt_eval_duration_ns / 1_000_000.0
                if prompt_tokens:
                    prefill_tps = prompt_tokens / (prompt_eval_duration_ns / 1e9)
            if eval_duration_ns and eval_duration_ns > 0:
                decode_ms = eval_duration_ns / 1_000_000.0
                if completion_tokens:
                    decode_tps = completion_tokens / (eval_duration_ns / 1e9)
            return timed_chat_fields(
                text="".join(text_parts),
                start=start,
                first_token_at=first_token_at,
                completion_tokens=completion_tokens,
                prompt_tokens=prompt_tokens,
                tool_calls=tool_calls,
                prefill_ms=prefill_ms,
                decode_ms=decode_ms,
                prefill_tokens_per_sec=prefill_tps,
                decode_tokens_per_sec=decode_tps,
                raw={
                    "prompt_eval_duration_ns": prompt_eval_duration_ns,
                    "eval_duration_ns": eval_duration_ns,
                    "done": {
                        k: done_raw.get(k)
                        for k in (
                            "prompt_eval_count",
                            "eval_count",
                            "prompt_eval_duration",
                            "eval_duration",
                            "total_duration",
                        )
                    },
                },
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
        """Ask Ollama to unload the model we loaded."""
        if model.is_url:
            await super().shutdown(model)
            return
        url = self._endpoint_for(model)
        try:
            async with httpx.AsyncClient(base_url=url, timeout=30.0) as client:
                # keep_alive=0 unloads from memory
                await client.post(
                    "/api/generate",
                    json={"model": model.id, "keep_alive": 0, "prompt": ""},
                )
        except Exception:
            pass
        await super().shutdown(model)

    async def aclose(self) -> None:
        await self._client.aclose()
