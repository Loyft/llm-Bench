"""Typer CLI entry point for llm-bench."""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any, Optional

import typer

from llm_bench import __version__
from llm_bench.benchmarks.registry import aliases, task_catalog
from llm_bench.config import BenchConfig, load_yaml_config, merge_config
from llm_bench.report.html import write_html_report
from llm_bench.report.terminal import render_terminal_report
from llm_bench.runner import BenchRunner
from llm_bench.schema import ProviderName
from llm_bench.ui.banner import boot_sequence
from llm_bench.ui.live import BenchLiveDisplay
from llm_bench.ui.theme import console
from llm_bench.ui.wizard import run_wizard
from llm_bench.workspace import create_workspace

app = typer.Typer(
    name="llm-bench",
    help="Retro CLI to benchmark local LLMs across providers.",
    no_args_is_help=False,
    add_completion=False,
)


def _version_callback(value: bool) -> None:
    if value:
        console.print(f"llm-bench {__version__}")
        raise typer.Exit()


@app.callback(invoke_without_command=True)
def main(
    ctx: typer.Context,
    version: bool = typer.Option(
        False, "--version", "-V", callback=_version_callback, is_eager=True
    ),
) -> None:
    if ctx.invoked_subcommand is None:
        config = run_wizard()
        _execute(config)


@app.command("list-providers")
def list_providers() -> None:
    """List supported providers."""
    for p in ProviderName:
        console.print(f"  [keygen.label]{p.value}[/]")


@app.command("list-tasks")
def list_tasks_cmd() -> None:
    """List benchmark tasks and aliases."""
    console.print("[keygen.title]Tasks[/]")
    alias_for: dict[str, list[str]] = {}
    for alias, target in aliases().items():
        alias_for.setdefault(target, []).append(alias)
    for kind in ("performance", "capability"):
        console.print(f"\n[keygen.accent]{kind}[/]")
        for t in (t for t in task_catalog() if t.kind == kind):
            al = ", ".join(sorted(alias_for.get(t.name, [])))
            console.print(f"  [keygen.label]{t.name:<22}[/] {t.description}")
            if al:
                console.print(f"  {'':<22} [keygen.dim]aliases: {al}[/]")
    console.print("\n[keygen.dim]Presets: all, performance, capability[/]")


@app.command("run")
def run(
    config_file: Optional[Path] = typer.Option(
        None, "--config", "-c", help="YAML config file"
    ),
    provider: Optional[str] = typer.Option(None, "--provider", "-p"),
    model: Optional[list[str]] = typer.Option(
        None, "--model", "-m", help="Model name, path, or URL (repeatable)"
    ),
    base_url: Optional[str] = typer.Option(None, "--base-url"),
    context: Optional[int] = typer.Option(None, "--context"),
    tools: Optional[str] = typer.Option(None, "--tools", help="on|off"),
    tasks: Optional[str] = typer.Option(
        None, "--tasks", help="comma list, or preset: all | performance | capability"
    ),
    max_items: Optional[int] = typer.Option(
        None, "--max-items", help="cap items per capability task (quick runs); 0 = full bank"
    ),
    no_music: bool = typer.Option(
        False, "--no-music", help="no background music / visuals sync during the run"
    ),
    music: Optional[Path] = typer.Option(
        None, "--music", help="audio file or folder to play (default: bundled playlist)"
    ),
    volume: Optional[float] = typer.Option(
        None, "--volume", min=0.0, max=1.0, help="starting music volume 0.0-1.0 (default 0.7)"
    ),
    no_code_exec: bool = typer.Option(
        False, "--no-code-exec", help="skip the coding task (never execute model-written code)"
    ),
    mode: Optional[str] = typer.Option(None, "--mode", help="sequential|parallel"),
    concurrency: Optional[int] = typer.Option(None, "--concurrency"),
    runs: Optional[int] = typer.Option(None, "--runs"),
    warmup: Optional[int] = typer.Option(None, "--warmup"),
    report: Optional[Path] = typer.Option(None, "--report", help="Durable report output dir"),
    workspace: Optional[Path] = typer.Option(None, "--workspace", help="Sandbox root override"),
    keep_workspace: bool = typer.Option(
        False, "--keep-workspace", help="Retain sandbox after run"
    ),
    timeout: Optional[float] = typer.Option(None, "--timeout"),
    seed: Optional[int] = typer.Option(None, "--seed"),
    system: Optional[str] = typer.Option(None, "--system"),
    yes: bool = typer.Option(
        False,
        "--yes",
        "-y",
        help="Skip setup wizard and run with provided flags/config only",
    ),
) -> None:
    """Run benchmarks (opens a setup wizard unless --yes is passed)."""
    yaml_data = load_yaml_config(config_file) if config_file else {}
    defaults: dict[str, Any] = dict(yaml_data or {})
    if "report" in defaults and "report_dir" not in defaults:
        defaults["report_dir"] = defaults.pop("report")

    # CLI overrides become wizard defaults
    for key, value in {
        "provider": provider,
        "models": model,
        "base_url": base_url,
        "context": context,
        "tools": tools,
        "tasks": tasks,
        "mode": mode,
        "concurrency": concurrency,
        "runs": runs,
        "warmup": warmup,
        "report_dir": report,
        "workspace": workspace,
        "keep_workspace": keep_workspace or None,
        "timeout": timeout,
        "seed": seed,
        "system": system,
        "max_items": max_items,
        "code_exec": False if no_code_exec else None,
        "music": False if no_music else None,
        "music_path": music,
        "music_volume": volume,
    }.items():
        if value is not None:
            defaults[key] = value

    if yes:
        try:
            if not defaults.get("models"):
                raise ValueError("at least one --model is required with --yes")
            config = merge_config(
                defaults,
                provider=None,
                models=None,
            )
        except Exception as exc:
            console.print(f"[keygen.fail]config error:[/] {exc}")
            raise typer.Exit(2) from exc
    else:
        try:
            config = run_wizard(defaults)
        except SystemExit:
            raise
        except Exception as exc:
            console.print(f"[keygen.fail]wizard error:[/] {exc}")
            raise typer.Exit(2) from exc

    _execute(config)


def _prepare_music(config: BenchConfig):  # type: ignore[no-untyped-def]
    """Resolve the playlist and analyse each track for the visuals.

    Returns (player | None, fallback visuals, label for the boot screen).
    """
    from llm_bench.audio import MusicPlayer, Track, find_playlist, load_visuals
    from llm_bench.audio.player import player_backend

    if not config.music:
        return None, load_visuals(None), None
    paths = find_playlist(config.music_path)
    if not paths:
        return None, load_visuals(None), "no track found (visuals only)"
    with console.status("[keygen.label]analysing module(s)…[/]", spinner="dots"):
        tracks = []
        for path in paths:
            vis = load_visuals(path)
            tracks.append(Track(path, vis.duration, vis))
    label = f"{len(tracks)} track{'s' if len(tracks) != 1 else ''}"
    if player_backend() is None:
        return None, tracks[0].visuals, f"{label} (no audio player; visuals only)"
    return MusicPlayer(tracks, volume=config.music_volume), tracks[0].visuals, label


def _execute(config: BenchConfig) -> None:
    from llm_bench.benchmarks.registry import resolve_tasks

    task_objs = resolve_tasks(
        config.tasks, tools_enabled=config.tools, code_exec_enabled=config.code_exec
    )
    task_names = [t.name for t in task_objs]

    player, visuals, music_label = _prepare_music(config)
    boot_sequence(
        provider=config.provider.value,
        models=config.models,
        tasks=task_names,
        music=music_label,
    )

    ws = create_workspace(root=config.workspace, keep=config.keep_workspace)
    live = BenchLiveDisplay(
        models=config.models,
        tasks=task_names,
        runs=max(1, sum(t.sample_count(config) for t in task_objs) // max(1, len(task_names))),
        player=player,
        visuals=visuals,
    )

    def on_progress(event: str, payload: dict) -> None:
        live.handle(event, payload)

    runner = BenchRunner(config, workspace=ws, on_progress=on_progress)
    exit_code = 0

    try:
        with live:
            report = asyncio.run(runner.run())

        draft_html = ws.report_dir / "report.html"
        write_html_report(report, draft_html)
        runner.export_report_files(report, html_path=draft_html)
        out_dir = Path(config.report_dir) / report.manifest.run_id
        html_path = out_dir / "report.html"

        render_terminal_report(report)
        console.print(
            f"\n[keygen.ok]REPORT[/] → [keygen.amber]{html_path.resolve()}[/]"
        )

        if not report.ok:
            uniq = list(dict.fromkeys(s.error for s in report.errors if s.error))
            console.print(
                "\n[keygen.fail]BENCHMARK FAILED[/] — no successful samples.\n"
                + "\n".join(f"  • {e}" for e in uniq)
            )
            exit_code = 1
    except KeyboardInterrupt:
        console.print("\n[keygen.fail]ABORT — cleaning up…[/]")
        exit_code = 130
    except Exception as exc:
        console.print(f"\n[keygen.fail]RUN FAILED:[/] {exc}")
        exit_code = 1
    finally:
        # Guaranteed cleanup of procs/models/tmp; wipe workspace unless kept
        ws.cleanup()

    if exit_code:
        raise typer.Exit(exit_code)


if __name__ == "__main__":
    app()
