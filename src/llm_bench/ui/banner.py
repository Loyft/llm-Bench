"""Keygen-style banner, dialog frames and the boot sequence."""

from __future__ import annotations

import hashlib
import random
import time

from rich import box
from rich.align import Align
from rich.console import Group, RenderableType
from rich.panel import Panel
from rich.text import Text

from llm_bench import __version__
from llm_bench.ui.theme import CREAM, TAN_MID, YELLOW, console

BANNER = r"""
██╗     ██╗     ███╗   ███╗      ██████╗ ███████╗███╗   ██╗ ██████╗██╗  ██╗
██║     ██║     ████╗ ████║      ██╔══██╗██╔════╝████╗  ██║██╔════╝██║  ██║
██║     ██║     ██╔████╔██║█████╗██████╔╝█████╗  ██╔██╗ ██║██║     ███████║
██║     ██║     ██║╚██╔╝██║╚════╝██╔══██╗██╔══╝  ██║╚██╗██║██║     ██╔══██║
███████╗███████╗██║ ╚═╝ ██║      ██████╔╝███████╗██║ ╚████║╚██████╗██║  ██║
╚══════╝╚══════╝╚═╝     ╚═╝      ╚═════╝ ╚══════╝╚═╝  ╚═══╝ ╚═════╝╚═╝  ╚═╝
""".strip("\n")

# top -> bottom fade of the lit faces; the extruded shadow stays dark
_FACE = ["#fff6d2", "#f7e9bd", "#efdcab", "#e4cd9b", "#d8be8c", "#c9ae7e"]
_SHADOW = TAN_MID
GREETS = "greetz to  ollama · llama.cpp · mlx · every GPU that ever got hot"


def render_logo() -> Text:
    out = Text(no_wrap=True)
    for i, line in enumerate(BANNER.splitlines()):
        face = _FACE[min(i, len(_FACE) - 1)]
        for ch in line:
            out.append(ch, face if ch == "█" else _SHADOW)
        out.append("\n")
    out.rstrip()
    return out


def raster_bar(width: int = 76) -> Text:
    """Copper-bar style horizontal gradient (brown -> yellow -> cream -> yellow -> brown)."""
    stops = [(0x7D, 0x60, 0x46), (0xFF, 0xE9, 0x33), (0xF1, 0xE2, 0xBB), (0xFF, 0xE9, 0x33),
             (0x7D, 0x60, 0x46)]
    bar = Text(no_wrap=True)
    for x in range(width):
        pos = x / max(1, width - 1) * (len(stops) - 1)
        i = min(int(pos), len(stops) - 2)
        f = pos - i
        rgb = tuple(int(a + (b - a) * f) for a, b in zip(stops[i], stops[i + 1]))
        bar.append("▀", "#%02x%02x%02x" % rgb)  # noqa: UP031
    return bar


def render_banner() -> Panel:
    subtitle = Text.assemble(
        (f"v{__version__}", f"bold {YELLOW}"),
        ("  ::  local LLM benchmark  ::  speed · reasoning · code · honesty", CREAM),
    )
    body = Group(
        Align.center(render_logo()),
        Align.center(raster_bar(74)),
        Align.center(subtitle),
        Align.center(Text(GREETS, style=TAN_MID)),
    )
    return Panel(
        body,
        box=box.DOUBLE,
        border_style="keygen.border",
        title=f"[bold {CREAM}] LLM-BENCH KEYGEN [/]",
        subtitle=f"[{TAN_MID}] tracker edition [/]",
        padding=(1, 2),
    )


def dialog(title: str, body: RenderableType) -> Panel:
    """A cream dialog window, like the settings boxes in old setup programs."""
    return Panel(body, box=box.DOUBLE, border_style="keygen.border",
                 title=f"[bold {CREAM}] {title} [/]", padding=(0, 2))


def serial_for(*parts: str) -> str:
    digest = hashlib.sha1("|".join(parts).encode()).hexdigest().upper()
    return "-".join(digest[i : i + 5] for i in range(0, 20, 5))


def _reveal_serial(serial: str) -> None:
    """Keygen-style serial that scrambles, then locks in left to right."""
    if not console.is_terminal:
        console.print(f"  [keygen.label]Serial[/] : [keygen.title]{serial}[/]")
        return
    rng = random.Random(serial)
    chars = "0123456789ABCDEF"
    from rich.live import Live

    with Live(console=console, refresh_per_second=40, transient=False) as live:
        for locked in range(len(serial) + 1):
            shown = "".join(
                c if (c == "-" or i < locked) else rng.choice(chars) for i, c in enumerate(serial)
            )
            live.update(
                Text.assemble(
                    ("  Serial", "keygen.label"), (" : ", ""), (shown, f"bold {YELLOW}")
                )
            )
            time.sleep(0.02)


def boot_sequence(
    *,
    provider: str,
    models: list[str],
    tasks: list[str],
    music: str | None = None,
) -> None:
    console.print(render_banner())
    name = ", ".join(models)
    console.print(f"  [keygen.label]Name  [/] : [keygen.accent]{name[:60]}[/]")
    _reveal_serial(serial_for(provider, *models, *tasks))
    console.print(
        f"  [{TAN_MID}]\\[[/][keygen.accent] Generate [/][{TAN_MID}]][/]  "
        f"[{TAN_MID}]\\[[/][keygen.label] About [/][{TAN_MID}]][/]  "
        f"[{TAN_MID}]\\[[/][keygen.label] Exit [/][{TAN_MID}]][/]"
    )
    console.print()
    steps = [
        ("provider", provider),
        ("targets", f"{len(models)} model(s)"),
        ("tasks", f"{len(tasks)} loaded"),
    ]
    if music:
        steps.append(("module", music))
    for label, value in steps:
        console.print(f"  [keygen.label]{label:<9}[/] [keygen.ok]{value}[/]")
        if console.is_terminal:
            time.sleep(0.05)
    console.print("\n  [keygen.title]Press Generate...[/] [keygen.dim](running benchmarks)[/]\n")
