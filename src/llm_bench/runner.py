"""Benchmark orchestration with workspace lifecycle."""

from __future__ import annotations

import asyncio
import shutil
from collections.abc import Callable
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from llm_bench.benchmarks.registry import resolve_tasks
from llm_bench.config import BenchConfig, RunMode
from llm_bench.hostinfo import collect_host_info
from llm_bench.memory import MemoryMonitor
from llm_bench.metrics import aggregate_samples, score_models
from llm_bench.providers import get_provider, resolve_model_ref
from llm_bench.providers.base import BaseProvider
from llm_bench.schema import (
    BenchReport,
    MemoryTimeline,
    ModelRef,
    RunManifest,
    SampleResult,
)
from llm_bench.workspace import Workspace, create_workspace

ProgressCallback = Callable[[str, dict[str, Any]], None]


class BenchRunner:
    def __init__(
        self,
        config: BenchConfig,
        *,
        workspace: Workspace | None = None,
        on_progress: ProgressCallback | None = None,
    ) -> None:
        self.config = config
        self.workspace = workspace or create_workspace(
            root=config.workspace,
            keep=config.keep_workspace,
        )
        self.on_progress = on_progress
        self._provider: BaseProvider | None = None

    def _emit(self, event: str, **payload: Any) -> None:
        if self.on_progress:
            self.on_progress(event, payload)

    def _model_refs(self) -> list[ModelRef]:
        return [
            resolve_model_ref(m, self.config.provider, base_url=self.config.base_url)
            for m in self.config.models
        ]

    async def _warmup(self, provider: BaseProvider, model: ModelRef) -> None:
        from llm_bench.schema import ChatMessage, ChatRequest

        for i in range(self.config.warmup):
            self._emit("warmup", model=model.label, index=i)
            req = ChatRequest(
                model=model,
                messages=[ChatMessage(role="user", content="Say OK.")],
                max_tokens=8,
                temperature=0.0,
                seed=self.config.seed,
                timeout=self.config.timeout,
            )
            await provider.chat(req)

    async def _run_model_tasks(
        self,
        provider: BaseProvider,
        model: ModelRef,
        tasks: list,
        memory: MemoryMonitor | None = None,
    ) -> list[SampleResult]:
        samples: list[SampleResult] = []
        try:
            self._emit("ensure_ready", model=model.label)
            if memory is not None:
                memory.model_name = model.id if not model.is_url else memory.model_name
                memory.set_label(f"load:{model.label}")
            await provider.ensure_ready(model)
            if memory is not None:
                await memory.mark_baseline()
            await self._warmup(provider, model)

            for task in tasks:
                for run_index in range(task.sample_count(self.config)):
                    if memory is not None:
                        memory.set_label(f"{model.label}/{task.name}#{run_index}")
                    self._emit(
                        "sample_start",
                        model=model.label,
                        task=task.name,
                        run_index=run_index,
                    )
                    sample = await task.run(
                        provider, model, self.config, run_index=run_index
                    )
                    samples.append(sample)
                    self._append_jsonl(sample)
                    self._emit(
                        "sample_done",
                        model=model.label,
                        task=task.name,
                        run_index=run_index,
                        error=sample.error,
                        metrics=sample.metrics,
                    )
        except Exception as exc:
            self._emit("model_error", model=model.label, error=str(exc))
            samples.append(
                SampleResult(
                    task="setup",
                    model=model.label,
                    provider=model.provider.value,
                    run_index=0,
                    error=str(exc),
                )
            )
        finally:
            try:
                await provider.shutdown(model)
            except Exception:
                pass
        return samples

    def _append_jsonl(self, sample: SampleResult) -> None:
        path = self.workspace.raw_dir / "samples.jsonl"
        with path.open("a", encoding="utf-8") as f:
            f.write(sample.model_dump_json() + "\n")

    async def run(self) -> BenchReport:
        self.workspace.ensure_dirs()
        self.workspace.install_signal_handlers()
        started = datetime.now(timezone.utc)
        models = self._model_refs()
        tasks = resolve_tasks(
            self.config.tasks,
            tools_enabled=self.config.tools,
            code_exec_enabled=self.config.code_exec,
        )

        provider = get_provider(
            self.config.provider,
            base_url=self.config.base_url,
            workspace=self.workspace,
            timeout=self.config.timeout,
        )
        self._provider = provider

        # Register provider close
        def _close_provider() -> None:
            try:
                loop = asyncio.get_event_loop()
                if loop.is_running():
                    loop.create_task(provider.aclose())
                else:
                    loop.run_until_complete(provider.aclose())
            except Exception:
                try:
                    asyncio.run(provider.aclose())
                except Exception:
                    pass

        self.workspace.register_cleanup(_close_provider)

        manifest = RunManifest(
            run_id=self.workspace.run_id,
            started_at=started,
            provider=self.config.provider,
            models=[m.label for m in models],
            model_count=len(models),
            tasks=[t.name for t in tasks],
            context=self.config.context,
            tools=self.config.tools,
            mode=self.config.mode.value,
            runs=self.config.runs,
            warmup=self.config.warmup,
            host=collect_host_info(),
            config=self.config.model_dump(mode="json"),
        )

        if not tasks:
            raise RuntimeError(
                "No tasks selected to run. Check --tasks / --tools / --no-code-exec "
                "(tool-calling is skipped when --tools off, coding when code exec is off)."
            )

        total_samples = len(models) * sum(t.sample_count(self.config) for t in tasks)
        self._emit(
            "run_start",
            run_id=manifest.run_id,
            models=manifest.models,
            total_samples=total_samples,
            plan=[(t.name, t.sample_count(self.config)) for t in tasks],
        )

        all_samples: list[SampleResult] = []
        memory = MemoryMonitor(
            provider=self.config.provider,
            base_url=self.config.base_url
            or (models[0].base_url if models else None)
            or "http://127.0.0.1:11434",
            model_name=models[0].id if models else None,
            context_tokens=self.config.context,
            interval_s=0.5,
        )
        mem_summary = None
        try:
            await memory.start()
            if self.config.mode == RunMode.PARALLEL and len(models) > 1:
                sem = asyncio.Semaphore(max(1, self.config.concurrency))

                async def _guarded(m: ModelRef) -> list[SampleResult]:
                    async with sem:
                        return await self._run_model_tasks(
                            provider, m, tasks, memory=memory
                        )

                results = await asyncio.gather(
                    *[_guarded(m) for m in models], return_exceptions=True
                )
                for r in results:
                    if isinstance(r, Exception):
                        self._emit("model_error", error=str(r))
                        all_samples.append(
                            SampleResult(
                                task="setup",
                                model="?",
                                provider=self.config.provider.value,
                                run_index=0,
                                error=str(r),
                            )
                        )
                    else:
                        all_samples.extend(r)
            else:
                for m in models:
                    all_samples.extend(
                        await self._run_model_tasks(provider, m, tasks, memory=memory)
                    )
        finally:
            mem_summary = await memory.stop()
            try:
                await provider.aclose()
            except Exception:
                pass

        finished = datetime.now(timezone.utc)
        manifest.finished_at = finished

        summaries = aggregate_samples(all_samples)
        scores = score_models(summaries)
        mode = "profile" if len(models) == 1 else "compare"
        memory_timeline = MemoryTimeline.model_validate(
            (mem_summary or memory.summary()).as_dict()
        )

        # Persist raw memory timeline alongside samples
        (self.workspace.raw_dir / "memory.json").write_text(
            memory_timeline.model_dump_json(indent=2), encoding="utf-8"
        )

        report = BenchReport(
            manifest=manifest,
            samples=all_samples,
            summaries=summaries,
            scores=scores,
            mode=mode,
            memory=memory_timeline,
        )

        if not report.ok:
            # Keep structured report for debugging, but make failure obvious upstream.
            errs = "; ".join(
                dict.fromkeys(s.error for s in report.errors if s.error)
            ) or "no successful samples"
            self._emit("run_failed", run_id=manifest.run_id, error=errs)

        # Write draft report into workspace
        draft = self.workspace.report_dir / "report.json"
        draft.write_text(report.model_dump_json(indent=2), encoding="utf-8")
        (self.workspace.raw_dir / "manifest.json").write_text(
            manifest.model_dump_json(indent=2), encoding="utf-8"
        )

        self._emit("run_done", run_id=manifest.run_id, mode=mode)
        return report

    def export_report_files(self, report: BenchReport, html_path: Path | None = None) -> Path:
        """Export a single durable HTML report. Returns the HTML path.

        Machine-readable JSON stays in the ephemeral workspace only (wiped on
        cleanup unless --keep-workspace), so each run produces one report file.
        """
        out_dir = Path(self.config.report_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        run_out = out_dir / report.manifest.run_id
        run_out.mkdir(parents=True, exist_ok=True)

        dest = run_out / "report.html"
        if html_path and html_path.exists():
            shutil.copy2(html_path, dest)
        else:
            # Fallback: still emit HTML even if draft path missing
            from llm_bench.report.html import write_html_report

            write_html_report(report, dest)
        return dest


async def run_benchmark(
    config: BenchConfig,
    *,
    on_progress: ProgressCallback | None = None,
) -> tuple[BenchReport, Workspace]:
    ws = create_workspace(root=config.workspace, keep=config.keep_workspace)
    runner = BenchRunner(config, workspace=ws, on_progress=on_progress)
    try:
        report = await runner.run()
        return report, ws
    except Exception:
        ws.cleanup()
        raise
