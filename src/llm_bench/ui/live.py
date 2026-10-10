"""Live run screen: a mission-control take on a classic tracker info page.

Dusk-violet desktop, deep-space panels with beveled frames, gold channel VU bars, a
scrolling mint pattern view, a rose oscilloscope. Everything on screen is driven either by the
song that is playing (spectrum bands) or by the benchmark's real progress.
"""

from __future__ import annotations

import threading
import time
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from rich.console import Console, ConsoleOptions, RenderResult

from llm_bench import __version__
from llm_bench.audio.analysis import BANDS, Visuals
from llm_bench.audio.player import VOLUME_STEP, MusicPlayer
from llm_bench.ui.canvas import Canvas
from llm_bench.ui.keys import KeyReader
from llm_bench.ui.theme import (
    BLACK,
    CREAM,
    GREEN,
    HILITE,
    RED,
    SCOPE_RED,
    SOFT,
    TAN,
    TAN_DARK,
    TAN_MID,
    YELLOW,
    console,
)

ROWS_PER_SEC = 8.0
CELL_W = 11  # "C-5 01 v48" + spacer
LABEL = f"bold {TAN_DARK} on {TAN}"
VALUE = f"bold {CREAM} on {TAN}"
DESK = f"{TAN_DARK} on {TAN}"
FIELD = f"bold {YELLOW} on {BLACK}"
PANEL = f"on {BLACK}"
SCOPE_CHARS = " ▁▂▃▄▅▆▇█"


@dataclass
class LogLine:
    text: str
    kind: str  # "ok" | "fail" | "info"


@dataclass
class LiveState:
    run_id: str = "-"
    models: list[str] = field(default_factory=list)
    plan: list[tuple[str, int]] = field(default_factory=list)
    total: int = 1
    done: int = 0
    errors: int = 0
    graded: int = 0
    passed: int = 0
    status: str = "BOOT"
    model: str = ""
    task: str = ""
    sample: int = 0
    last_tps: float | None = None
    started: float = field(default_factory=time.monotonic)
    started_wall: datetime = field(default_factory=datetime.now)
    log: deque[LogLine] = field(default_factory=lambda: deque(maxlen=200))
    dots: deque[tuple[float, str]] = field(default_factory=lambda: deque(maxlen=40))
    max_tps: float = 1.0
    sample_t0: float | None = None
    task_done: dict[str, int] = field(default_factory=dict)
    task_secs: dict[str, float] = field(default_factory=dict)
    levels: list[float] = field(default_factory=lambda: [0.0] * BANDS)
    scope: deque[float] = field(default_factory=lambda: deque(maxlen=64))

    @property
    def task_index(self) -> int:
        for i, (name, _) in enumerate(self.plan):
            if name == self.task:
                return i + 1
        return 0

    @property
    def task_samples(self) -> int:
        for name, n in self.plan:
            if name == self.task:
                return n
        return 0

    def eta_seconds(self) -> float | None:
        """Estimated time left, or None until a first sample has finished.

        Each remaining sample is priced at its own task's observed mean duration (a coding
        sample is not a knowledge question); tasks not seen yet use the overall mean.
        """
        if self.status == "DONE":
            return 0.0
        n_done = sum(self.task_done.values())
        if n_done == 0:
            return None
        fallback = sum(self.task_secs.values()) / n_done
        total = 0.0
        for name, per_model in self.plan:
            remaining = max(0, per_model * max(1, len(self.models)) - self.task_done.get(name, 0))
            done = self.task_done.get(name, 0)
            mean = self.task_secs[name] / done if done else fallback
            total += remaining * mean
        return total

    @property
    def model_index(self) -> int:
        return self.models.index(self.model) + 1 if self.model in self.models else 0


def _fit(text: str, width: int) -> str:
    return text if len(text) <= width else text[: max(0, width - 1)] + "…"


def _clock(seconds: float) -> str:
    s = int(seconds)
    return f"{s // 3600}:{s % 3600 // 60:02d}:{s % 60:02d}"


class BenchLiveDisplay:
    """Rich renderable + context manager; feed it runner events via :meth:`handle`."""

    def __init__(
        self,
        models: list[str],
        tasks: list[str],
        runs: int = 1,
        *,
        player: MusicPlayer | None = None,
        visuals: Visuals | None = None,
    ) -> None:
        self.state = LiveState(models=list(models), plan=[(t, runs) for t in tasks])
        self.state.total = max(1, len(models) * len(tasks) * runs)
        self.player = player
        self.visuals = visuals or Visuals(synthetic=True)
        self._lock = threading.RLock()
        self._live: Any = None
        self._t0 = time.monotonic()
        self._console: Console = console
        self._keys: KeyReader | None = None

    # -- lifecycle -----------------------------------------------------------------------

    def __enter__(self) -> BenchLiveDisplay:
        from rich.live import Live

        self._t0 = time.monotonic()
        if self.player is not None:
            self.player.start()
        self._live = Live(
            self, console=self._console, screen=True, refresh_per_second=15, transient=True
        )
        self._live.__enter__()
        if self.player is not None:
            self._keys = KeyReader(self.handle_key)
            self._keys.start()
        return self

    def __exit__(self, *args: Any) -> None:
        # Restore the terminal's input mode, then silence the music, so neither outlasts the run.
        if self._keys is not None:
            self._keys.stop()
            self._keys = None
        if self.player is not None:
            self.player.stop()
        if self._live is not None:
            self._live.__exit__(*args)

    # -- keys ----------------------------------------------------------------------------

    def handle_key(self, key: str) -> None:
        """p / space: pause; + / - (or up / down): volume; n / b (or right / left): next / prev."""
        player = self.player
        if player is None or player.failed:
            return
        if key in ("p", "P", "space"):
            player.toggle_pause()
        elif key in ("+", "=", "]", "up"):
            player.adjust_volume(VOLUME_STEP)
        elif key in ("-", "_", "[", "down"):
            player.adjust_volume(-VOLUME_STEP)
        elif key in ("n", "N", "right"):
            player.next()
        elif key in ("b", "B", "left"):
            player.prev()

    # -- events --------------------------------------------------------------------------

    def handle(self, event: str, payload: dict[str, Any]) -> None:
        with self._lock:
            st = self.state
            if event == "run_start":
                st.run_id = str(payload.get("run_id", "-"))
                if payload.get("models"):
                    st.models = list(payload["models"])
                if payload.get("plan"):
                    st.plan = [(str(n), int(c)) for n, c in payload["plan"]]
                if payload.get("total_samples"):
                    st.total = max(1, int(payload["total_samples"]))
                st.status = "BOOT"
                st.log.append(LogLine(f"run {st.run_id} locked", "info"))
            elif event == "ensure_ready":
                st.status = "LOAD"
                st.model = payload.get("model", st.model)
                st.log.append(LogLine(f"loading {st.model}", "info"))
            elif event == "warmup":
                st.status = "WARMUP"
                st.model = payload.get("model", st.model)
            elif event == "sample_start":
                st.status = "RUN"
                st.model = payload.get("model", st.model)
                st.task = payload.get("task", st.task)
                st.sample = int(payload.get("run_index", 0)) + 1
                st.sample_t0 = time.monotonic()
            elif event == "sample_done":
                self._sample_done(payload)
            elif event == "model_error":
                st.errors += 1
                st.log.append(LogLine(f"ERR {payload.get('error', '')}", "fail"))
            elif event == "run_done":
                st.status = "DONE"
                st.log.append(LogLine("sequence complete", "info"))
            elif event == "run_failed":
                st.status = "FAILED"
                st.log.append(LogLine(f"FAILED {payload.get('error', '')}", "fail"))

    def _sample_done(self, payload: dict[str, Any]) -> None:
        st = self.state
        st.done += 1
        task = payload.get("task", "?")
        if st.sample_t0 is not None:
            st.task_done[task] = st.task_done.get(task, 0) + 1
            st.task_secs[task] = st.task_secs.get(task, 0.0) + (time.monotonic() - st.sample_t0)
            st.sample_t0 = None
        idx = int(payload.get("run_index", 0)) + 1
        head = f"{task[:14]:<14} {idx:02d}"
        err = payload.get("error")
        metrics = payload.get("metrics") or {}
        if err:
            st.errors += 1
            st.log.append(LogLine(f"{head} ERR  {err}", "fail"))
            st.dots.append((0.02, "fail"))
            return
        tps = metrics.get("tokens_per_sec")
        if isinstance(tps, (int, float)) and not isinstance(tps, bool):
            st.last_tps = float(tps)
            st.max_tps = max(st.max_tps, float(tps))
        passed = metrics.get("passed")
        if isinstance(passed, bool):
            st.graded += 1
            st.passed += int(passed)
            kind = "ok" if passed else "fail"
            st.log.append(LogLine(f"{head} {'PASS' if passed else 'FAIL'}", kind))
            st.dots.append((1.0 if passed else 0.1, kind))
        else:
            detail = f"{tps:.1f} t/s" if isinstance(tps, (int, float)) else "ok"
            st.log.append(LogLine(f"{head} {detail}", "ok"))
            frac = (float(tps) / st.max_tps) if isinstance(tps, (int, float)) else 0.5
            st.dots.append((max(0.05, min(1.0, frac)), "ok"))

    # -- time / visuals ------------------------------------------------------------------

    def _now(self) -> float:
        """Playback position when the song is audible, else wall-clock so visuals still move."""
        if self.player is not None:
            pos = self.player.position()
            if pos is not None:
                return pos
        return time.monotonic() - self._t0

    def _current_visuals(self) -> Visuals:
        """The playing track's analysis, else the display-wide fallback."""
        cur = getattr(self.player, "current", None)
        vis = getattr(cur, "visuals", None)
        return vis if vis is not None else self.visuals

    def _advance_levels(self, t: float) -> None:
        st = self.state
        paused = bool(getattr(self.player, "paused", False))
        target = [0.0] * BANDS if paused else self._current_visuals().levels_at(t)
        st.levels = [max(v, old * 0.80) for v, old in zip(target, st.levels)]
        st.scope.extend([0.0] if paused else self._current_visuals().wave(t, 1))

    # -- rendering -----------------------------------------------------------------------

    def __rich_console__(self, console: Console, options: ConsoleOptions) -> RenderResult:
        size = options.size
        width = size.width
        height = options.height or size.height
        with self._lock:
            t = self._now()
            self._advance_levels(t)
            canvas = self.draw(width, height, t)
        yield canvas.to_text()

    def draw(self, width: int, height: int, t: float) -> Canvas:
        cv = Canvas(width, height, DESK)
        st = self.state
        if width < 64 or height < 20:
            self._draw_compact(cv, st, t)
            return cv

        elapsed = time.monotonic() - st.started
        title = f"LLM-BENCH v{__version__} started {st.started_wall:%b %e %Y %H:%M:%S}"
        cv.put((width - len(title)) // 2, 0, title, LABEL)

        self._draw_header(cv, st, t)
        tight = height < 30  # drop the blank spacer rows to leave room for the pattern view
        status_y = 6 if tight else 7
        self._draw_status(cv, st, t, status_y, elapsed)
        info_y = status_y + (1 if tight else 2)
        info = "Benchmark Info (live)"
        playing_now = self._now_playing()
        if playing_now:
            info = _fit(f"{info}  ::  {playing_now}", width - 6)
        cv.put((width - len(info)) // 2, info_y, info, LABEL)

        panel_h = 10 if height >= 34 else (6 if height >= 26 else 5)
        py = info_y + 1
        self._draw_panels(cv, st, py, panel_h)
        iy = py + panel_h
        self._draw_info(cv, st, iy)
        pat_y = iy + 3
        self._draw_pattern(cv, pat_y, height - pat_y, t)
        return cv

    # header block -------------------------------------------------------------------------

    def _draw_header(self, cv: Canvas, st: LiveState, t: float) -> None:
        w = cv.w
        fw = min(30, max(14, w // 4))
        rows = [
            ("Run ID", _fit(st.run_id, fw)),
            ("Models", _fit(", ".join(st.models), fw)),
            ("Order", f"{st.model_index:03d}/{len(st.models):03d}"),
            ("Pattern", f"{st.task_index:03d}/{len(st.plan):03d}"),
            ("Row", f"{st.sample:03d}/{st.task_samples:03d}"),
        ]
        for i, (label, value) in enumerate(rows):
            y = 1 + i
            cv.put(2, y, f"{label:>7}", LABEL)
            vw = fw if i < 2 else 9
            cv.put(10, y, " " * vw, FIELD)
            cv.put(10, y, f" {value}"[:vw], FIELD)

        mx = 10 + fw + 3
        if w >= 84:
            self._draw_music_controls(cv, mx)

        # right block: current task / speed / pass rate, plus the oscilloscope
        scope_w = 26 if w >= 112 else 0
        right_x = mx + 22
        if w >= 100 and right_x + 30 <= w - scope_w - 2:
            pass_rate = f"{st.passed / st.graded:.0%}" if st.graded else "-"
            speed = f"{st.last_tps:.1f} tok/s" if st.last_tps is not None else "-"
            info = [("Task", _fit(st.task or "-", 20)), ("Speed", speed), ("Passing", pass_rate)]
            for i, (label, value) in enumerate(info):
                y = 1 + i
                cv.put(right_x, y, f"{label:>9}", LABEL)
                cv.put(right_x + 10, y, " " * 21, FIELD)
                cv.put(right_x + 10, y, f" {value}"[:21], FIELD)
        if scope_w:
            self._draw_scope(cv, w - scope_w - 2, 1, scope_w, 4)

    def _now_playing(self) -> str:
        pl = self.player
        cur = getattr(pl, "current", None)
        if pl is None or cur is None or pl.failed:
            return ""
        return f"Now playing {pl.index % len(pl.tracks) + 1}/{len(pl.tracks)}: {cur.title}"

    def _draw_music_controls(self, cv: Canvas, x: int) -> None:
        pl = self.player
        if pl is None:
            cv.put(x, 1, "MUSIC...off", LABEL)
            cv.put(x, 5, "CTRL+C..Abort", LABEL)
            return
        if pl.failed:
            state = "failed"
        elif getattr(pl, "paused", False):
            state = "paused"
        else:
            state = "playing" if pl.playing else "..."
        cv.put(x, 1, f"MUSIC...{state}", LABEL)
        vol = float(getattr(pl, "volume", 1.0))
        filled = int(round(vol * 10))
        cv.put(x, 2, "VOL ", LABEL)
        cv.put(x + 4, 2, "█" * filled + "░" * (10 - filled), f"{YELLOW} on {BLACK}")
        cv.put(x + 15, 2, f"{int(round(vol * 100)):>3}%", VALUE)
        if not pl.failed:
            cv.put(x, 3, "P..Pause  +/-..Vol", LABEL)
            if len(getattr(pl, "tracks", [])) > 1:
                cv.put(x, 4, "B/N..Prev/Next", LABEL)
        cv.put(x, 5, "CTRL+C..Abort", LABEL)

    def _draw_scope(self, cv: Canvas, x: int, y: int, w: int, h: int) -> None:
        cv.bevel_box(x, y, w, h, fill=PANEL, dark=TAN_MID, light=CREAM)
        data = list(self.state.scope)[-(w - 2):]
        inner_h = h - 2
        for i, v in enumerate(data):
            level = max(0.0, min(1.0, v)) * inner_h
            for row in range(inner_h):
                from_bottom = inner_h - 1 - row
                frac = max(0.0, min(1.0, level - from_bottom))
                ch = SCOPE_CHARS[int(frac * 8)]
                cv.put(x + 1 + i, y + 1 + row, ch, f"{SCOPE_RED} on {BLACK}")

    # status line --------------------------------------------------------------------------

    def _draw_status(self, cv: Canvas, st: LiveState, t: float, y: int, elapsed: float) -> None:
        active = sum(1 for v in st.levels if v > 0.15)
        state_word = {"RUN": "Running", "BOOT": "Starting", "LOAD": "Loading model",
                      "WARMUP": "Warming up", "DONE": "Finished", "FAILED": "Failed"}.get(
            st.status, st.status)
        parts: list[tuple[str, str]] = [
            (f"{state_word}, ", LABEL),
            ("Model: ", LABEL),
            (f"{st.model_index}/{len(st.models)}", VALUE),
            (", Task: ", LABEL),
            (_fit(st.task or "-", 22), VALUE),
            (", Row: ", LABEL),
            (f"{st.sample}/{st.task_samples}", VALUE),
            (", ", LABEL),
            (str(active), VALUE),
            (" Channels", LABEL),
        ]
        x = 2
        for text, style in parts:
            cv.put(x, y, text, style)
            x += len(text)
        clock = f"Time  {_clock(elapsed)}"
        cv.put(cv.w - len(clock) - 3, y, clock, LABEL)

    # three panels -------------------------------------------------------------------------

    def _panel_geometry(self, w: int) -> tuple[int, int, int, int, int, int]:
        wr = 10 if w >= 100 else 8
        x_left = 4
        wl = max(24, int((w - 8 - wr) * 0.36))
        x_mid = x_left + wl + 2
        wm = w - 1 - wr - 2 - x_mid
        if wm < 20:
            wl = max(20, wl - (20 - wm))
            x_mid = x_left + wl + 2
            wm = max(8, w - 1 - wr - 2 - x_mid)
        x_right = w - 1 - wr
        return x_left, wl, x_mid, wm, x_right, wr

    def _draw_panels(self, cv: Canvas, st: LiveState, y: int, h: int) -> None:
        x_left, wl, x_mid, wm, x_right, wr = self._panel_geometry(cv.w)
        inner = h - 2
        for x, wid in ((x_left, wl), (x_mid, wm), (x_right, wr)):
            cv.bevel_box(x, y, wid, h, fill=PANEL, dark=TAN_MID, light=CREAM)

        # VU bars: one row per channel (or averaged groups when short on rows)
        nrows = 8 if inner >= 8 else (4 if inner >= 4 else 2)
        group = BANDS // nrows
        for r in range(nrows):
            level = max(st.levels[r * group:(r + 1) * group])
            row_y = y + 1 + r
            cv.put(x_left - 3, row_y, f"{r + 1:02d}", f"{TAN_MID} on {TAN}")
            n = int(round(level * (wl - 2)))
            cv.put(x_left + 1, row_y, "▍" * n, f"{YELLOW} on {BLACK}")
            if n >= int((wl - 2) * 0.9):
                cv.put(x_left + n, row_y, "▍", f"{RED} on {BLACK}")

        # result log: newest at the bottom
        lines = list(st.log)[-inner:]
        for i, line in enumerate(lines):
            colour = {"ok": GREEN, "fail": RED}.get(line.kind, CREAM)
            cv.put(x_mid + 1, y + 1 + inner - len(lines) + i,
                   _fit(line.text, wm - 2), f"{colour} on {BLACK}")

        # dots: newest at the bottom, x position = value
        dots = list(st.dots)[-inner:]
        for i, (value, kind) in enumerate(dots):
            dx = int(value * (wr - 3))
            colour = {"fail": RED}.get(kind, SOFT)
            cv.put(x_right + 1 + dx, y + 1 + inner - len(dots) + i, "██", f"{colour} on {BLACK}")

    # progress rows ------------------------------------------------------------------------

    def _draw_info(self, cv: Canvas, st: LiveState, y: int) -> None:
        pass_rate = f"{st.passed / st.graded:.0%}" if st.graded else "-"
        cv.put(4, y, f"Samples done: {st.done} ({st.total})", LABEL)
        cv.put(4 + 24, y, f"Errors: {st.errors}", LABEL)
        cv.put(4 + 36, y, f"Pass rate: {pass_rate}", LABEL)
        eta = st.eta_seconds()
        eta_text = "ETA --:--" if eta is None else f"ETA {_clock(eta)}"
        cv.put(cv.w - 4 - len(eta_text), y, eta_text, VALUE)
        bar_x = 4
        bar_w = cv.w - 8 - 8
        frac = min(1.0, st.done / st.total)
        filled = int(frac * bar_w)
        cv.put(bar_x, y + 1, "█" * filled + "░" * (bar_w - filled), f"{YELLOW} on {BLACK}")
        cv.put(bar_x + bar_w + 1, y + 1, f"{frac:>4.0%}", LABEL)

    # pattern view -------------------------------------------------------------------------

    def _draw_pattern(self, cv: Canvas, y: int, h: int, t: float) -> None:
        if h < 5:
            return
        x0 = 4
        wid = cv.w - 1 - x0
        nch = max(1, min(BANDS, (wid - 2) // CELL_W))
        cw = max(CELL_W, (wid - 2) // nch)  # spread channels across the full width
        # channel header strip above the box
        cv.fill(x0, y, wid, 1, f"{CREAM} on {TAN_MID}")
        for c in range(nch):
            cv.put(x0 + 1 + c * cw, y, f"{c + 1:02d}".center(cw - 1), f"bold {CREAM} on {TAN_MID}")
        cv.bevel_box(x0, y + 1, wid, h - 1, fill=PANEL, dark=TAN_MID, light=CREAM)

        inner = h - 3
        vis = self._current_visuals()
        cur = int(t * ROWS_PER_SEC)
        mid = inner // 2
        for i in range(inner):
            row = cur - mid + i
            row_y = y + 2 + i
            hl = i == mid
            bg = HILITE if hl else BLACK
            if hl:
                cv.fill(x0 + 1, row_y, wid - 2, 1, f"on {HILITE}")
            if row < 0:
                continue
            cv.put(0, row_y, f"{row % 1000:03d}", f"{CREAM if hl else TAN_MID} on {TAN}")
            for c in range(nch):
                x = x0 + 1 + c * cw
                note = vis.note(row, c)
                if note is None:
                    cv.put(x, row_y, "··· ·· ···", f"#2d6b5a on {bg}")
                else:
                    name, ins, vol = note
                    cv.put(x, row_y, name, f"bold {GREEN} on {bg}")
                    cv.put(x + 4, row_y, f"{ins:02d}", f"{CREAM} on {bg}")
                    cv.put(x + 7, row_y, f"v{vol:02d}", f"#b5f0dd on {bg}")

    # compact fallback ---------------------------------------------------------------------

    def _draw_compact(self, cv: Canvas, st: LiveState, t: float) -> None:
        title = f"LLM-BENCH v{__version__}"
        cv.put(max(0, (cv.w - len(title)) // 2), 0, title, LABEL)
        eta = st.eta_seconds()
        line = (
            f"{st.status}  {st.model_index}/{len(st.models)}  {st.task or '-'}  "
            f"{st.done}/{st.total}  ETA {'--:--' if eta is None else _clock(eta)}"
        )
        cv.put(1, 2, _fit(line, cv.w - 2), VALUE)
        bar_w = max(4, cv.w - 8)
        filled = int(min(1.0, st.done / st.total) * bar_w)
        cv.put(1, 3, "█" * filled + "░" * (bar_w - filled), f"{YELLOW} on {BLACK}")
        vu = "".join(" ▁▂▃▄▅▆▇█"[int(min(1.0, v) * 8)] for v in st.levels)
        cv.put(1, 5, vu, f"{YELLOW} on {BLACK}")
        for i, entry in enumerate(list(st.log)[-(cv.h - 7):]):
            cv.put(1, 7 + i, _fit(entry.text, cv.w - 2), DESK)

