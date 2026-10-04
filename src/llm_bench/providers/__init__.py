"""LLM provider adapters."""

from __future__ import annotations

from llm_bench.providers.base import BaseProvider, resolve_model_ref
from llm_bench.providers.llamacpp import LlamaCppProvider
from llm_bench.providers.mlx_provider import MLXProvider
from llm_bench.providers.ollama import OllamaProvider
from llm_bench.providers.openai_compat import OpenAICompatProvider
from llm_bench.schema import ProviderName
from llm_bench.workspace import Workspace


def get_provider(
    name: ProviderName | str,
    *,
    base_url: str | None = None,
    workspace: Workspace | None = None,
    timeout: float = 120.0,
) -> BaseProvider:
    key = ProviderName(name) if not isinstance(name, ProviderName) else name
    if key == ProviderName.OLLAMA:
        return OllamaProvider(base_url=base_url, workspace=workspace, timeout=timeout)
    if key == ProviderName.OPENAI:
        return OpenAICompatProvider(base_url=base_url, workspace=workspace, timeout=timeout)
    if key == ProviderName.LLAMACPP:
        return LlamaCppProvider(base_url=base_url, workspace=workspace, timeout=timeout)
    if key == ProviderName.MLX:
        return MLXProvider(workspace=workspace, timeout=timeout)
    raise ValueError(f"unknown provider: {name}")


__all__ = [
    "BaseProvider",
    "get_provider",
    "resolve_model_ref",
    "OllamaProvider",
    "OpenAICompatProvider",
    "LlamaCppProvider",
    "MLXProvider",
]
