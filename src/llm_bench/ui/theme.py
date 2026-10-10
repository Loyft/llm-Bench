"""Tracker / keygen-scene palette: warm tan, cream and yellow with tracker green.

Hex colours are down-converted automatically on 256-colour terminals.
"""

from __future__ import annotations

from rich.console import Console
from rich.theme import Theme

# Palette sampled from classic tracker UIs (tan desktop, black panels, cream bevels).
TAN = "#b59475"
TAN_DARK = "#3b2b1d"  # label text on the tan desktop
TAN_MID = "#7d6046"
CREAM = "#f1e2bb"
YELLOW = "#ffe933"
GREEN = "#43c943"
RED = "#d9453d"
SCOPE_RED = "#a8284a"
BLACK = "#000000"
HILITE = "#4a3b30"  # current-row bar in the pattern view

KEYGEN_THEME = Theme(
    {
        "keygen.title": f"bold {YELLOW}",
        "keygen.accent": f"bold {CREAM}",
        "keygen.amber": "#e6c15a",
        "keygen.ok": f"bold {GREEN}",
        "keygen.fail": f"bold {RED}",
        "keygen.dim": TAN_MID,
        "keygen.border": TAN,
        "keygen.meter": YELLOW,
        "keygen.label": "#d9bc96",
    }
)

console = Console(theme=KEYGEN_THEME, force_terminal=None)
