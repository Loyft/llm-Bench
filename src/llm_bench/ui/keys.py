"""Non-blocking single-key input for the live screen (no Enter needed)."""

from __future__ import annotations

import os
import select
import sys
import threading
from collections.abc import Callable

try:  # POSIX only; on Windows the reader is simply inactive
    import termios
    import tty
except ImportError:  # pragma: no cover
    termios = None  # type: ignore[assignment]
    tty = None  # type: ignore[assignment]

ARROWS = {"A": "up", "B": "down", "C": "right", "D": "left"}


def parse_keys(data: str) -> list[str]:
    """Split raw input into key names; arrow escape sequences become 'up'/'down'/..."""
    keys: list[str] = []
    i = 0
    while i < len(data):
        ch = data[i]
        if ch == "\x1b" and data[i + 1 : i + 2] in ("[", "O") and i + 2 < len(data):
            keys.append(ARROWS.get(data[i + 2], "esc"))
            i += 3
            continue
        keys.append("space" if ch == " " else ch)
        i += 1
    return keys


class KeyReader:
    """Reads keys on a daemon thread and calls ``on_key(name)``.

    The terminal is switched to cbreak mode (Ctrl+C still raises KeyboardInterrupt) and always
    restored by :meth:`stop`. If ``fd`` is not a TTY the reader still works for pipes (tests),
    it just has no terminal state to change (only with ``allow_non_tty``).
    """

    def __init__(
        self,
        on_key: Callable[[str], None],
        fd: int | None = None,
        *,
        allow_non_tty: bool = False,
    ) -> None:
        self.on_key = on_key
        self.allow_non_tty = allow_non_tty  # tests only: a pipe stands in for the keyboard
        if fd is None:
            try:
                fd = sys.stdin.fileno()
            except (AttributeError, ValueError, OSError):
                fd = -1  # no usable stdin (captured / closed)
        self.fd = fd
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._saved: list | None = None

    @property
    def usable(self) -> bool:
        try:
            return os.isatty(self.fd) and termios is not None
        except (OSError, ValueError):
            return False

    def start(self) -> bool:
        if self._thread is not None:
            return True
        if self.fd < 0:
            return False
        if not os.isatty(self.fd) and not self.allow_non_tty:
            return False  # never consume piped input that belongs to someone else
        if os.isatty(self.fd):
            if termios is None:
                return False
            try:
                self._saved = termios.tcgetattr(self.fd)
                tty.setcbreak(self.fd)
            except (termios.error, OSError):
                self._saved = None
                return False
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        return True

    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                ready, _, _ = select.select([self.fd], [], [], 0.1)
                if not ready:
                    continue
                data = os.read(self.fd, 32)
            except (OSError, ValueError):
                return
            if not data:
                return
            for key in parse_keys(data.decode("utf-8", "ignore")):
                try:
                    self.on_key(key)
                except Exception:
                    pass  # a bad handler must never kill input or the run

    def stop(self) -> None:
        self._stop.set()
        thread = self._thread
        if thread is not None and thread is not threading.current_thread():
            thread.join(timeout=1.0)
        self._thread = None
        if self._saved is not None and termios is not None:
            try:
                termios.tcsetattr(self.fd, termios.TCSADRAIN, self._saved)
            except (termios.error, OSError):
                pass
            self._saved = None
