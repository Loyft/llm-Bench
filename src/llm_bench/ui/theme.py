"""Mars-colony palette: violet night sky, cream ink, gold / mint / coral accents.

Matches the HTML report. Hex colours are down-converted automatically on 256-colour
terminals. Constant names are historical (they began life as a tan tracker theme):
TAN is the desktop background, BLACK the panel fill.
"""

from __future__ import annotations

from rich.console import Console
from rich.theme import Theme

TAN = "#2a0f3d"  # desktop: dusk violet
TAN_DARK = "#b9a7c9"  # label text on the desktop (soft lavender)
TAN_MID = "#7a5aa0"  # dim text, frame shadows, header bars
CREAM = "#f6e7c8"
YELLOW = "#ffc857"  # primary highlight (gold)
GREEN = "#7fe0c3"  # ok (mint)
RED = "#ff6b6b"  # failures
SCOPE_RED = "#ff8fb8"  # scope trace (rose)
BLACK = "#0d0620"  # panel fill (deep space)
HILITE = "#3a1c5a"  # current-row bar in the pattern view
CORAL = "#ff7a59"
PERI = "#9bb0ff"
SOFT = "#b9a7c9"
# One colour per model, reused across bars and tables.
MODEL_COLORS = [YELLOW, GREEN, CORAL, PERI, SCOPE_RED, CREAM]

KEYGEN_THEME = Theme(
    {
        "keygen.title": f"bold {YELLOW}",
        "keygen.accent": f"bold {CREAM}",
        "keygen.amber": CORAL,
        "keygen.ok": f"bold {GREEN}",
        "keygen.fail": f"bold {RED}",
        "keygen.dim": SOFT,
        "keygen.border": CORAL,
        "keygen.meter": YELLOW,
        "keygen.label": GREEN,
    }
)

console = Console(theme=KEYGEN_THEME, force_terminal=None)
