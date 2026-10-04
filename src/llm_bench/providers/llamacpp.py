"""llama.cpp provider — GGUF path (spawn llama-server) or existing server URL."""

from __future__ import annotations

import asyncio
import os
import shutil
import socket
import subprocess
import time
from pathlib import Path

import httpx

from llm_bench.providers.base import BaseProvider
from llm_bench.providers.openai_compat import OpenAICompatProvider
from llm_bench.schema import ChatRequest, ChatResult, ModelRef, ProviderName


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


class LlamaCppProvider(BaseProvider):
    name = ProviderName.LLAMACPP

    def __init__(self, **kwargs) -> None:  # type: ignore[no-untyped-def]
        super().__init__(**kwargs)
        self._servers: dict[str, tuple[subprocess.Popen, str]] = {}
        self._compat = OpenAICompatProvider(
            base_url=self.base_url or "http://127.0.0.1:8080/v1",
            workspace=self.workspace,
            timeout=self.timeout,
        )

    def _find_llama_server(self) -> str | None:
        for candidate in ("llama-server", "llama.cpp-server", "server"):
            path = shutil.which(candidate)
            if path:
                return path
        # Homebrew / common locations
        for path in (
            "/opt/homebrew/bin/llama-server",
            "/usr/local/bin/llama-server",
        ):
            if Path(path).exists():
                return path
        return None

    async def _wait_ready(self, base: str, timeout: float = 60.0) -> None:
        deadline = time.time() + timeout
        url = base.rstrip("/") + "/models"
        async with httpx.AsyncClient(timeout=5.0) as client:
            while time.time() < deadline:
                try:
                    r = await client.get(url)
                    if r.status_code < 500:
                        return
                except Exception:
                    pass
                await asyncio.sleep(0.3)
        raise TimeoutError(f"llama-server did not become ready at {base}")

    async def ensure_ready(self, model: ModelRef) -> None:
        if model.is_url:
            model.base_url = model.id.rstrip("/")
            await self._compat.ensure_ready(model)
            self._mark_loaded(model)
            return

        if model.is_path or Path(model.id).exists():
            gguf = Path(model.id).expanduser().resolve()
            if not gguf.exists():
                raise FileNotFoundError(f"GGUF not found: {gguf}")
            if model.id in self._servers:
                return
            binary = self._find_llama_server()
            if not binary:
                raise RuntimeError(
                    "llama-server binary not found on PATH; "
                    "start a server yourself and pass its URL, or install llama.cpp"
                )
            port = _free_port()
            host = "127.0.0.1"
            base = f"http://{host}:{port}"
            log_path = None
            if self.workspace is not None:
                log_path = self.workspace.tmp_path(f"llama-server-{port}.log")
                self.workspace.register_port(port)

            stdout = open(log_path, "w") if log_path else subprocess.DEVNULL  # noqa: SIM115
            proc = subprocess.Popen(
                [
                    binary,
                    "-m",
                    str(gguf),
                    "--host",
                    host,
                    "--port",
                    str(port),
                    "-c",
                    "8192",
                ],
                stdout=stdout,
                stderr=subprocess.STDOUT,
                start_new_session=True,
                env=os.environ.copy(),
            )
            if self.workspace is not None:
                self.workspace.register_process(proc, label=f"llama-server:{gguf.name}", owned=True)

                def _close_log() -> None:
                    try:
                        if hasattr(stdout, "close"):
                            stdout.close()
                    except Exception:
                        pass

                self.workspace.register_cleanup(_close_log)

            self._servers[model.id] = (proc, base)
            model.base_url = base
            await self._wait_ready(base + "/v1")
            self._mark_loaded(model)
            return

        # Treat as model name on an existing server
        await self._compat.ensure_ready(model)
        self._mark_loaded(model)

    async def list_models(self) -> list[ModelRef]:
        try:
            return await self._compat.list_models()
        except Exception:
            return []

    async def chat(self, req: ChatRequest) -> ChatResult:
        if req.model.id in self._servers:
            _, base = self._servers[req.model.id]
            req.model.base_url = base
        elif req.model.is_url:
            req.model.base_url = req.model.id.rstrip("/")
        return await self._compat.chat(req)

    async def shutdown(self, model: ModelRef) -> None:
        entry = self._servers.pop(model.id, None)
        if entry is not None:
            proc, _ = entry
            if proc.poll() is None:
                try:
                    if self.workspace is not None:
                        self.workspace._kill_process_tree(proc)
                    else:
                        proc.terminate()
                        try:
                            proc.wait(timeout=3)
                        except subprocess.TimeoutExpired:
                            proc.kill()
                except Exception:
                    pass
        await super().shutdown(model)

    async def aclose(self) -> None:
        for mid in list(self._servers.keys()):
            await self.shutdown(ModelRef(id=mid, provider=self.name, is_path=True))
        await self._compat.aclose()
