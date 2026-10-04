"""Live progress display — VU-meter style bars."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Any

from rich.console import Group
from rich.live import Live
from rich.panel import Panel
from rich.progress import BarColumn, Progress, SpinnerColumn, TextColumn
from rich.table import Table
from rich.text import Text

from llm_bench.ui.theme import KEYGEN_THEME, console


def _vu(level: float, width: int = 20) -> str:
    level = max(0.0, min(1.0, level))
    filled = int(level * width)
    return "█" * filled + "░" * (width - filled)


@dataclass
class BenchLiveDisplay:
    models: list[str]
    tasks: list[str]
    runs: int
    _logs: deque[str] = field(default_factory=lambda: deque(maxlen=8))
    _status: str = "IDLE"
    _current: dict[str, str] = field(default_factory=dict)
    _done: int = 0
    _total: int = 0
    _live: Live | None = None
    _tick: int = 0

    def __post_init__(self) -> None:
        self._total = max(1, len(self.models) * len(self.tasks) * self.runs)
        self._progress = Progress(
            SpinnerColumn(style="keygen.accent"),
            TextColumn("[keygen.label]{task.description}"),
            BarColumn(bar_width=28, complete_style="magenta", finished_style="green"),
            TextColumn("[keygen.amber]{task.percentage:>3.0f}%"),
            console=console,
        )
        self._task_id = self._progress.add_task("CRACKING…", total=self._total)

    def __enter__(self) -> BenchLiveDisplay:
        self._live = Live(
            self._render(),
            console=console,
            refresh_per_second=12,
            transient=False,
        )
        self._live.__enter__()
        return self

    def __exit__(self, *args: Any) -> None:
        if self._live:
            self._live.__exit__(*args)

    def handle(self, event: str, payload: dict[str, Any]) -> None:
        self._tick += 1
        if event == "run_start":
            self._status = "BOOT"
            self._logs.appendleft(f"run {payload.get('run_id')} locked")
        elif event == "warmup":
            self._status = "WARMUP"
            self._current[payload.get("model", "?")] = "warmup"
            self._logs.appendleft(f"warmup {payload.get('model')}")
        elif event == "ensure_ready":
            self._status = "LOAD"
            self._logs.appendleft(f"loading {payload.get('model')}")
        elif event == "sample_start":
            self._status = "CRACKING"
            m = payload.get("model", "?")
            t = payload.get("task", "?")
            self._current[m] = f"{t}#{payload.get('run_index', 0)}"
            self._logs.appendleft(f"→ {m} / {t}")
        elif event == "sample_done":
            self._done += 1
            self._progress.update(self._task_id, completed=self._done)
            err = payload.get("error")
            if err:
                self._logs.appendleft(f"! error: {err[:60]}")
            else:
                metrics = payload.get("metrics") or {}
                tps = metrics.get("tokens_per_sec")
                if tps is not None:
                    self._logs.appendleft(f"✓ {tps:.1f} tok/s")
                else:
                    self._logs.appendleft("✓ sample ok")
        elif event == "model_error":
            self._logs.appendleft(f"!! {payload.get('error', '')[:70]}")
        elif event == "run_done":
            self._status = "UNLOCKED"
            self._logs.appendleft("sequence complete")
        elif event == "run_failed":
            self._status = "FAILED"
            self._logs.appendleft(f"!! {payload.get('error', 'run failed')[:70]}")

        if self._live:
            self._live.update(self._render())

    def _render(self) -> Panel:
        blink = "█" if (self._tick % 4) < 2 else " "
        header = Text.assemble(
            ("STATUS ", "keygen.dim"),
            (self._status, "keygen.accent"),
            (f" {blink}", "keygen.amber"),
        )

        meter_level = self._done / self._total
        meter = Text(
            f"SIGNAL [{_vu(meter_level)}] {self._done}/{self._total}",
            style="keygen.meter",
        )

        table = Table.grid(padding=(0, 2))
        table.add_column(style="keygen.label")
        table.add_column()
        for m in self.models:
            cur = self._current.get(m, "—")
            table.add_row(m[:28], Text(cur, style="keygen.amber"))

        log_lines = Text("\n").join(
            Text(f"  {line}", style="keygen.dim") for line in list(self._logs)[:6]
        )

        body = Group(
            header,
            Text(""),
            meter,
            Text(""),
            self._progress,
            Text(""),
            table,
            Text(""),
            Panel(log_lines or Text("  …", style="keygen.dim"), title="TRACE", border_style="magenta"),
        )
        return Panel(
            body,
            title="[keygen.title]LLM-BENCH // LIVE[/]",
            border_style="keygen.border",
            subtitle="[keygen.dim]ctrl+c aborts & cleans up[/]",
        )
