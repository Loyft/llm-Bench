"""Interactive pre-run dialog for configuring a benchmark."""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

from rich.prompt import Confirm, IntPrompt, Prompt
from rich.table import Table

from llm_bench.benchmarks.registry import list_tasks
from llm_bench.config import BenchConfig, RunMode
from llm_bench.providers import get_provider
from llm_bench.schema import ProviderName
from llm_bench.ui.banner import dialog, render_banner
from llm_bench.ui.theme import console

TASK_CHOICES = [
    ("speed", "generation_speed — decode tok/s, TTFT, inter-token latency"),
    ("latency", "prompt_latency — prefill across prompt sizes"),
    ("throughput", "throughput — aggregate tok/s under 1/2/4 concurrent requests"),
    ("context", "long_context — needle retrieval across context sizes and depths"),
    ("tools", "tool_calling — tool selection, argument accuracy, restraint"),
    ("quality", "instruction_quality — 4-prompt sanity smoke test"),
    ("reasoning", "reasoning — word problems, arithmetic, logic"),
    ("knowledge", "knowledge — multiple-choice general knowledge"),
    ("coding", "coding — write Python, graded by running unit tests"),
    ("instructions", "instruction_following — IFEval-style verifiable constraints"),
    ("hallucination", "hallucination — abstain / correct false premises / answer controls"),
    ("multi_turn", "multi_turn — memory, corrections, system-prompt persistence"),
    ("consistency", "consistency — answer stability at temperature 0.8"),
]

CONTEXT_PRESETS = [2048, 4096, 8192, 16384, 32768, 65536]


def _print_numbered(title: str, items: list[str]) -> None:
    table = Table(title=title, border_style="keygen.border", header_style="keygen.accent")
    table.add_column("#", style="keygen.amber", width=4)
    table.add_column("Option", style="keygen.label")
    for i, item in enumerate(items, start=1):
        table.add_row(str(i), item)
    console.print(table)


def _parse_multi_indices(raw: str, n: int) -> list[int]:
    """Parse '1,3' or '1 3' or 'all' into 0-based indices."""
    raw = raw.strip().lower()
    if raw in {"all", "*"}:
        return list(range(n))
    parts = [p for p in raw.replace(" ", ",").split(",") if p]
    out: list[int] = []
    for p in parts:
        idx = int(p) - 1
        if idx < 0 or idx >= n:
            raise ValueError(f"choice out of range: {p}")
        out.append(idx)
    return out


async def _discover_models(provider_name: str, base_url: str | None) -> list[str]:
    try:
        provider = get_provider(provider_name, base_url=base_url, timeout=5.0)
        try:
            refs = await provider.list_models()
            return [r.label for r in refs]
        finally:
            await provider.aclose()
    except Exception as exc:
        console.print(f"[keygen.dim]could not list models ({exc})[/]")
        return []


def run_wizard(defaults: dict[str, Any] | None = None) -> BenchConfig:
    """Prompt for provider / models / tasks / settings. Returns a BenchConfig."""
    d = defaults or {}
    console.print(render_banner())
    console.print(
        dialog(
            "SETUP",
            "[keygen.title]Configure your run[/]\n"
            "[keygen.dim]Pick from the lists, or type a custom value when offered. "
            "Music starts only once the run does.[/]",
        )
    )

    providers = [p.value for p in ProviderName]
    _print_numbered("Providers", providers)
    default_provider = str(d.get("provider") or "ollama")
    default_p_idx = (
        str(providers.index(default_provider) + 1) if default_provider in providers else "1"
    )
    p_raw = Prompt.ask("Provider # (or name)", default=default_p_idx)
    if p_raw.isdigit() and 1 <= int(p_raw) <= len(providers):
        provider = providers[int(p_raw) - 1]
    elif p_raw in providers:
        provider = p_raw
    else:
        raise ValueError(f"unknown provider: {p_raw}")

    base_url = d.get("base_url")
    if Confirm.ask("Set custom base URL?", default=bool(base_url)):
        base_url = Prompt.ask("Base URL", default=base_url or "http://127.0.0.1:11434")
    else:
        base_url = base_url

    console.print("\n[keygen.label]Discovering models…[/]")
    discovered = asyncio.run(_discover_models(provider, base_url))
    model_options = list(discovered)
    model_options.append("⇢ custom name / path / URL…")
    _print_numbered("Models", model_options)

    default_models = d.get("models") or []
    hint = (
        f"default from flags: {', '.join(default_models)}"
        if default_models
        else "e.g. 1 or 1,2 or custom"
    )
    console.print(f"[keygen.dim]{hint}[/]")
    m_raw = Prompt.ask(
        "Model #s (comma-separated), or type a custom model id/path/url",
        default=(
            str(model_options.index(default_models[0]) + 1)
            if len(default_models) == 1 and default_models[0] in model_options
            else (",".join(default_models) if default_models else "1")
        ),
    )

    models: list[str] = []
    # If input looks like indices relative to list
    try:
        if any(ch.isdigit() for ch in m_raw) and all(
            part.strip().isdigit() or part.strip() == ""
            for part in m_raw.replace(" ", ",").split(",")
        ):
            idxs = _parse_multi_indices(m_raw, len(model_options))
            for idx in idxs:
                choice = model_options[idx]
                if choice.startswith("⇢"):
                    custom = Prompt.ask("Custom model name / path / URL")
                    models.append(custom.strip())
                else:
                    models.append(choice)
        else:
            # treat as custom comma-separated names/paths
            models = [m.strip() for m in m_raw.split(",") if m.strip()]
    except ValueError:
        models = [m.strip() for m in m_raw.split(",") if m.strip()]

    if not models:
        raise ValueError("at least one model is required")

    # Tasks
    task_labels = [f"{alias} — {desc}" for alias, desc in TASK_CHOICES]
    _print_numbered("Tasks", task_labels + ["all"])
    default_tasks = d.get("tasks")
    alias_to_idx = {alias: i + 1 for i, (alias, _) in enumerate(TASK_CHOICES)}
    if isinstance(default_tasks, list):
        nums = [str(alias_to_idx[x]) for x in default_tasks if x in alias_to_idx]
        default_task_str = ",".join(nums) or "all"
    elif isinstance(default_tasks, str) and default_tasks.strip().lower() == "all":
        default_task_str = "all"
    elif isinstance(default_tasks, str):
        # Allow comma aliases from CLI to map to numbers when possible
        parts = [p.strip() for p in default_tasks.split(",") if p.strip()]
        if all(p in alias_to_idx for p in parts):
            default_task_str = ",".join(str(alias_to_idx[p]) for p in parts)
        else:
            default_task_str = default_tasks
    else:
        default_task_str = "all"
    t_raw = Prompt.ask(
        "Task #s (comma-separated) or 'all'",
        default=default_task_str,
    )
    if t_raw.strip().lower() in {"all", str(len(TASK_CHOICES) + 1)}:
        tasks = [alias for alias, _ in TASK_CHOICES]
    else:
        try:
            idxs = _parse_multi_indices(t_raw, len(TASK_CHOICES))
            tasks = [TASK_CHOICES[i][0] for i in idxs]
        except ValueError:
            tasks = [t.strip() for t in t_raw.split(",") if t.strip()]

    # Context
    ctx_labels = [str(c) for c in CONTEXT_PRESETS] + ["custom"]
    _print_numbered("Context window", ctx_labels)
    default_ctx = int(d.get("context") or 8192)
    default_ctx_idx = (
        str(CONTEXT_PRESETS.index(default_ctx) + 1)
        if default_ctx in CONTEXT_PRESETS
        else str(len(CONTEXT_PRESETS) + 1)
    )
    c_raw = Prompt.ask("Context # (or enter tokens)", default=default_ctx_idx)
    if c_raw.isdigit() and 1 <= int(c_raw) <= len(CONTEXT_PRESETS):
        context = CONTEXT_PRESETS[int(c_raw) - 1]
    elif c_raw.isdigit() and int(c_raw) == len(CONTEXT_PRESETS) + 1:
        context = IntPrompt.ask("Custom context tokens", default=default_ctx)
    elif c_raw.isdigit():
        context = int(c_raw)
    else:
        context = default_ctx

    tools_default = d.get("tools")
    if isinstance(tools_default, str):
        tools_default = tools_default.lower() in {"on", "true", "1", "yes"}
    if tools_default is None:
        tools_default = "tools" in tasks
    tools = Confirm.ask("Enable tool-calling task?", default=bool(tools_default))

    mode_default = str(d.get("mode") or "sequential")
    mode = Prompt.ask("Mode", choices=["sequential", "parallel"], default=mode_default)

    runs = IntPrompt.ask("Runs for speed/latency tasks", default=int(d.get("runs") or 3))
    max_items = IntPrompt.ask(
        "Max items per capability task (0 = full bank, e.g. 5 for a quick pass)",
        default=int(d.get("max_items") or 0),
    )
    warmup = IntPrompt.ask("Warmup runs", default=int(d.get("warmup") or 1))

    music = Confirm.ask(
        "Play background music during the run? (never while you configure or after it ends)",
        default=bool(d.get("music", True)),
    )

    report_dir = Prompt.ask(
        "Report directory",
        default=str(d.get("report_dir") or d.get("report") or "./reports"),
    )

    console.print()
    summary = Table(title="Run plan", border_style="keygen.border", header_style="keygen.accent")
    summary.add_column("Key", style="keygen.label")
    summary.add_column("Value", style="keygen.amber")
    summary.add_row("provider", provider)
    summary.add_row("base_url", base_url or "(default)")
    summary.add_row("models", ", ".join(models))
    summary.add_row("tasks", ", ".join(tasks))
    summary.add_row("context", str(context))
    summary.add_row("tools", "on" if tools else "off")
    summary.add_row("mode", mode)
    summary.add_row("runs / warmup", f"{runs} / {warmup}")
    summary.add_row("max items", str(max_items) if max_items else "full bank")
    summary.add_row("music", "on" if music else "off")
    summary.add_row("report", report_dir)
    console.print(summary)

    if not Confirm.ask("Start benchmark?", default=True):
        raise SystemExit(0)

    return BenchConfig(
        provider=ProviderName(provider),
        models=models,
        base_url=base_url,
        context=context,
        tools=tools,
        tasks=tasks,
        mode=RunMode(mode),
        runs=runs,
        warmup=warmup,
        max_items=max_items or None,
        code_exec=bool(d.get("code_exec", True)),
        music=music,
        music_path=Path(d["music_path"]).expanduser() if d.get("music_path") else None,
        report_dir=Path(report_dir),
        keep_workspace=bool(d.get("keep_workspace") or False),
        workspace=Path(d["workspace"]).expanduser() if d.get("workspace") else None,
        timeout=float(d.get("timeout") or 120.0),
        seed=int(d.get("seed") or 42),
        system=d.get("system"),
        interactive=True,
        concurrency=int(d.get("concurrency") or 2),
    )


# Silence unused import warning for list_tasks in static checkers
_ = list_tasks
