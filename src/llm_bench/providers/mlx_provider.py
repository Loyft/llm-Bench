"""Apple Silicon MLX provider via mlx-lm (optional extra)."""

from __future__ import annotations

import asyncio
import time
from pathlib import Path

from llm_bench.providers.base import BaseProvider, timed_chat_fields
from llm_bench.schema import ChatRequest, ChatResult, ModelRef, ProviderName


class MLXProvider(BaseProvider):
    name = ProviderName.MLX

    def __init__(self, **kwargs) -> None:  # type: ignore[no-untyped-def]
        super().__init__(**kwargs)
        self._models: dict[str, tuple[object, object, object]] = {}

    def _import_mlx(self):
        try:
            from mlx_lm import generate, load  # type: ignore
            from mlx_lm.sample_utils import make_sampler  # type: ignore
        except ImportError as exc:
            raise RuntimeError(
                "mlx-lm is not installed. Install with: pip install 'llm-bench[mlx]'"
            ) from exc
        return load, generate, make_sampler

    async def ensure_ready(self, model: ModelRef) -> None:
        if model.id in self._models:
            return
        load, _generate, _sampler = self._import_mlx()
        path = model.id

        def _load():
            return load(path)

        mlx_model, tokenizer = await asyncio.to_thread(_load)
        self._models[model.id] = (mlx_model, tokenizer, _generate)
        self._mark_loaded(model)

    async def list_models(self) -> list[ModelRef]:
        # MLX has no central registry; return empty or workspace hints
        return []

    async def chat(self, req: ChatRequest) -> ChatResult:
        if req.model.id not in self._models:
            await self.ensure_ready(req.model)
        mlx_model, tokenizer, generate = self._models[req.model.id]
        _, _, make_sampler = self._import_mlx()

        messages = []
        if req.system:
            messages.append({"role": "system", "content": req.system})
        for m in req.messages:
            messages.append({"role": m.role, "content": m.content})

        if hasattr(tokenizer, "apply_chat_template"):
            prompt = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        else:
            prompt = "\n".join(f"{m['role']}: {m['content']}" for m in messages) + "\nassistant:"

        start = time.perf_counter()
        first_token_at: float | None = None
        try:
            sampler = make_sampler(temp=req.temperature)

            def _gen():
                nonlocal first_token_at
                # mlx-lm generate is blocking; approximate TTFT as start of generation
                first_token_at = time.perf_counter()
                return generate(
                    mlx_model,
                    tokenizer,
                    prompt=prompt,
                    max_tokens=req.max_tokens,
                    sampler=sampler,
                    verbose=False,
                )

            text = await asyncio.to_thread(_gen)
            # rough token count
            completion_tokens = max(1, len(text.split()))
            if hasattr(tokenizer, "encode"):
                try:
                    completion_tokens = len(tokenizer.encode(text))
                except Exception:
                    pass
            return timed_chat_fields(
                text=text,
                start=start,
                first_token_at=first_token_at,
                completion_tokens=completion_tokens,
            )
        except Exception as exc:
            return timed_chat_fields(
                text="",
                start=start,
                first_token_at=first_token_at,
                completion_tokens=None,
                error=str(exc),
            )

    async def shutdown(self, model: ModelRef) -> None:
        self._models.pop(model.id, None)
        # Encourage GC of large weights
        try:
            import gc

            gc.collect()
        except Exception:
            pass
        await super().shutdown(model)

    async def aclose(self) -> None:
        for mid in list(self._models.keys()):
            await self.shutdown(
                ModelRef(
                    id=mid,
                    provider=self.name,
                    is_path=Path(mid).exists(),
                )
            )
