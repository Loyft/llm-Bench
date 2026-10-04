"""Retro keygen color theme — cyan / magenta / amber on black."""

from __future__ import annotations

from rich.console import Console
from rich.theme import Theme

KEYGEN_THEME = Theme(
    {
        "keygen.title": "bold bright_cyan",
        "keygen.accent": "bold magenta",
        "keygen.amber": "bold yellow",
        "keygen.ok": "bold green",
        "keygen.fail": "bold red",
        "keygen.dim": "bright_black",
        "keygen.border": "bright_cyan",
        "keygen.meter": "magenta",
        "keygen.label": "cyan",
    }
)

console = Console(theme=KEYGEN_THEME, force_terminal=None)
