"""ASCII banner + boot sequence."""

from __future__ import annotations

import time

from rich.align import Align
from rich.panel import Panel
from rich.text import Text

from llm_bench import __version__
from llm_bench.ui.theme import console

BANNER = r"""
██╗     ██╗     ███╗   ███╗      ██████╗ ███████╗███╗   ██╗ ██████╗██╗  ██╗
██║     ██║     ████╗ ████║      ██╔══██╗██╔════╝████╗  ██║██╔════╝██║  ██║
██║     ██║     ██╔████╔██║█████╗██████╔╝█████╗  ██╔██╗ ██║██║     ███████║
██║     ██║     ██║╚██╔╝██║╚════╝██╔══██╗██╔══╝  ██║╚██╗██║██║     ██╔══██║
███████╗███████╗██║ ╚═╝ ██║      ██████╔╝███████╗██║ ╚████║╚██████╗██║  ██║
╚══════╝╚══════╝╚═╝     ╚═╝      ╚═════╝ ╚══════╝╚═╝  ╚═══╝ ╚═════╝╚═╝  ╚═╝
"""


def render_banner() -> Panel:
    text = Text(BANNER, style="keygen.title")
    subtitle = Text(
        f"  v{__version__}  //  LOCAL MODEL CRACKER  //  SYS.READY",
        style="keygen.amber",
    )
    body = Text.assemble(text, "\n", subtitle)
    return Panel(
        Align.center(body),
        border_style="keygen.border",
        title="[keygen.accent]★ KEYGEN ★[/]",
        subtitle="[keygen.dim]authorized personnel only[/]",
    )


def boot_sequence(*, provider: str, models: list[str], tasks: list[str]) -> None:
    console.print(render_banner())
    steps = [
        ("MEM CHECK", "OK"),
        ("PROVIDER PROBE", provider.upper()),
        ("MODEL LOCK", f"{len(models)} TARGET(S)"),
        ("TASK MATRIX", ",".join(tasks)[:40]),
        ("CHECKSUM", "0xDEADBEEF"),
    ]
    for label, value in steps:
        console.print(
            f"  [keygen.dim]>[/] [keygen.label]{label:<16}[/] "
            f"[keygen.ok][{value}][/]"
        )
        time.sleep(0.05)
    console.print(
        "\n  [keygen.accent]CRACKING SEQUENCE INITIATED…[/]\n"
    )
