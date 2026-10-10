"""A character-cell canvas: paint text/boxes with per-cell styles, then render as Rich Text."""

from __future__ import annotations

from rich.text import Text


class Canvas:
    def __init__(self, width: int, height: int, style: str = "") -> None:
        self.w = max(1, width)
        self.h = max(1, height)
        self._ch = [[" "] * self.w for _ in range(self.h)]
        self._st = [[style] * self.w for _ in range(self.h)]

    def put(self, x: int, y: int, text: str, style: str | None = None) -> None:
        """Write ``text`` at (x, y), clipped. ``style=None`` keeps each cell's existing style."""
        if not 0 <= y < self.h:
            return
        for i, c in enumerate(text):
            cx = x + i
            if 0 <= cx < self.w:
                self._ch[y][cx] = c
                if style is not None:
                    self._st[y][cx] = style

    def fill(self, x: int, y: int, w: int, h: int, style: str, char: str = " ") -> None:
        for cy in range(max(0, y), min(self.h, y + h)):
            for cx in range(max(0, x), min(self.w, x + w)):
                self._ch[cy][cx] = char
                self._st[cy][cx] = style

    def hline(self, x: int, y: int, w: int, char: str, style: str) -> None:
        self.put(x, y, char * w, style)

    def bevel_box(
        self,
        x: int,
        y: int,
        w: int,
        h: int,
        *,
        fill: str,
        dark: str,
        light: str,
    ) -> None:
        """Panel with a beveled frame: top/left edges in ``dark``, bottom/right in ``light``."""
        if w < 2 or h < 2:
            return
        self.fill(x, y, w, h, fill)
        bg = _bg(fill)
        dk = f"{dark} on {bg}"
        lt = f"{light} on {bg}"
        self.put(x, y, "┌" + "─" * (w - 2), dk)
        self.put(x + w - 1, y, "┐", lt)
        for cy in range(y + 1, y + h - 1):
            self.put(x, cy, "│", dk)
            self.put(x + w - 1, cy, "│", lt)
        self.put(x, y + h - 1, "└", dk)
        self.put(x + 1, y + h - 1, "─" * (w - 2) + "┘", lt)

    def to_text(self) -> Text:
        out = Text(no_wrap=True, overflow="crop")
        for y in range(self.h):
            run_style = self._st[y][0]
            run: list[str] = []
            for x in range(self.w):
                st = self._st[y][x]
                if st != run_style:
                    out.append("".join(run), run_style)
                    run, run_style = [], st
                run.append(self._ch[y][x])
            out.append("".join(run), run_style)
            if y < self.h - 1:
                out.append("\n")
        return out

    def plain_lines(self) -> list[str]:
        return ["".join(row) for row in self._ch]


def _bg(style: str) -> str:
    """Background colour named in a Rich style string ('fg on bg' or 'on bg')."""
    parts = style.split()
    return parts[parts.index("on") + 1] if "on" in parts else style
