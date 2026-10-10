"""Tracker-style live screen, canvas, and music lifecycle around a run."""

from __future__ import annotations

import io
import time
from pathlib import Path
from unittest.mock import patch

import pytest
from mock_provider import OracleProvider
from rich.console import Console

from llm_bench.audio.analysis import Visuals
from llm_bench.config import BenchConfig, merge_config
from llm_bench.runner import BenchRunner
from llm_bench.schema import ProviderName
from llm_bench.ui.canvas import Canvas
from llm_bench.ui.live import BenchLiveDisplay

MODELS = ["llama3.2:3b", "qwen2.5:7b"]
PLAN = [("generation_speed", 3), ("reasoning", 18)]


def _display(**kw) -> BenchLiveDisplay:  # type: ignore[no-untyped-def]
    vis = Visuals(synthetic=True)
    d = BenchLiveDisplay(MODELS, [p[0] for p in PLAN], runs=3, visuals=vis, **kw)
    d.handle("run_start", {"run_id": "r1", "models": MODELS, "total_samples": 42, "plan": PLAN})
    return d


# ---- canvas ---------------------------------------------------------------------------


def test_canvas_clips_and_keeps_style() -> None:
    cv = Canvas(5, 2, "red")
    cv.put(3, 0, "abcdef", "blue")  # clipped at the right edge
    cv.put(-2, 1, "xyz", None)  # clipped at the left, style untouched
    cv.put(0, 9, "off-canvas")
    assert cv.plain_lines() == ["   ab", "z    "]
    text = cv.to_text()
    assert text.plain == "   ab\nz    "
    styles = {str(s.style) for s in text.spans}
    assert styles == {"red", "blue"}


def test_bevel_box_frames_and_fills() -> None:
    cv = Canvas(6, 4, "white on #111111")
    cv.bevel_box(0, 0, 6, 4, fill="on #000000", dark="#333333", light="#eeeeee")
    assert cv.plain_lines() == ["┌────┐", "│    │", "│    │", "└────┘"]


# ---- drawing --------------------------------------------------------------------------


@pytest.mark.parametrize("size", [(200, 60), (130, 40), (100, 30), (80, 24), (64, 20), (60, 18),
                                  (40, 10), (20, 5), (1, 1)])
def test_draw_fits_every_terminal_size(size: tuple[int, int]) -> None:
    w, h = size
    d = _display()
    d.handle("sample_start", {"model": MODELS[0], "task": "reasoning", "run_index": 3})
    for i in range(30):  # more log lines than any panel can show
        d.handle("sample_done", {"model": MODELS[0], "task": "reasoning", "run_index": i,
                                 "metrics": {"passed": i % 2 == 0, "tokens_per_sec": 20.0 + i}})
    for t in (0.0, 12.3, 99.9):
        cv = d.draw(w, h, t)
        lines = cv.plain_lines()
        assert len(lines) == h and all(len(line) == w for line in lines)
        assert cv.to_text().plain.count("\n") == h - 1


def test_full_screen_shows_run_state() -> None:
    d = _display()
    d.handle("ensure_ready", {"model": MODELS[0]})
    d.handle("sample_start", {"model": MODELS[0], "task": "reasoning", "run_index": 6})
    d.handle("sample_done", {"model": MODELS[0], "task": "reasoning", "run_index": 5,
                             "metrics": {"passed": True}})
    d.handle("sample_done", {"model": MODELS[0], "task": "reasoning", "run_index": 6,
                             "metrics": {"passed": False}})
    d.handle("sample_done", {"model": MODELS[0], "task": "generation_speed", "run_index": 0,
                             "metrics": {"tokens_per_sec": 42.5}})
    d.handle("sample_done", {"model": MODELS[0], "task": "reasoning", "run_index": 7,
                             "error": "boom"})
    screen = "\n".join(d.draw(130, 40, 20.0).plain_lines())
    for needle in ("LLM-BENCH", "Run ID", "r1", "Order", "001/002", "Pattern", "002/002",
                   "Row", "007/018", "Running", "reasoning", "PASS", "FAIL", "ERR  boom",
                   "42.5 t/s", "Samples done: 4 (42)", "Errors: 1", "Pass rate: 50%",
                   "Benchmark Info"):
        assert needle in screen, needle


def test_status_words_and_terminal_states() -> None:
    d = _display()
    d.handle("run_done", {})
    assert "Finished" in "\n".join(d.draw(130, 40, 0).plain_lines())
    d.handle("run_failed", {"error": "nope"})
    assert "Failed" in "\n".join(d.draw(130, 40, 0).plain_lines())


def test_vu_bars_follow_the_music_and_decay() -> None:
    vis = Visuals(frames=[[1.0] * 8] * 8 + [[0.0] * 8] * 80, fps=16)
    d = BenchLiveDisplay(MODELS, ["reasoning"], visuals=vis)
    d._advance_levels(0.1)
    assert all(v == 1.0 for v in d.state.levels)
    d._advance_levels(0.75)  # silence: bars fall, they do not snap to zero
    assert all(0.0 < v < 1.0 for v in d.state.levels)
    for k in range(40):
        d._advance_levels(0.75 + k * 0.01)
    assert all(v < 0.01 for v in d.state.levels)


def test_pattern_view_scrolls_with_time() -> None:
    d = _display()
    a = "\n".join(d.draw(130, 40, 5.0).plain_lines())
    b = "\n".join(d.draw(130, 40, 7.0).plain_lines())
    assert a != b


# ---- music lifecycle ------------------------------------------------------------------


class FakePlayer:
    def __init__(self, order: list[str]) -> None:
        self.order = order
        self.failed = False
        self.playing = False
        self.paused = False
        self.volume = 0.5
        self.calls: list[str] = []

    def toggle_pause(self) -> None:
        self.paused = not self.paused
        self.calls.append("toggle")

    def next(self) -> None:
        self.calls.append("next")

    def prev(self) -> None:
        self.calls.append("prev")

    def adjust_volume(self, delta: float) -> None:
        self.volume = round(self.volume + delta, 2)
        self.calls.append(f"vol{delta:+.1f}")

    def start(self) -> bool:
        self.order.append("music-start")
        self.playing = True
        return True

    def stop(self) -> None:
        self.order.append("music-stop")
        self.playing = False

    def position(self) -> float | None:
        return 1.5 if self.playing else None


def _console(order: list[str]) -> Console:
    return Console(file=io.StringIO(), force_terminal=True, width=100, height=30,
                   color_system="truecolor")


def test_music_plays_only_inside_the_live_screen_and_stops_first() -> None:
    order: list[str] = []
    fake = FakePlayer(order)
    d = _display(player=fake)  # type: ignore[arg-type]
    d._console = _console(order)
    order.append("before")
    with d:
        order.append("running")
        d.handle("sample_start", {"model": MODELS[0], "task": "reasoning", "run_index": 0})
        time.sleep(0.15)  # let the refresh thread draw a few frames
    order.append("after")
    assert order == ["before", "music-start", "running", "music-stop", "after"]


def test_music_stops_when_the_run_is_aborted() -> None:
    order: list[str] = []
    d = _display(player=FakePlayer(order))  # type: ignore[arg-type]
    d._console = _console(order)
    with pytest.raises(KeyboardInterrupt):
        with d:
            raise KeyboardInterrupt
    assert order == ["music-start", "music-stop"]


def test_visuals_run_on_wall_clock_without_a_player() -> None:
    d = _display()
    t0 = d._now()
    time.sleep(0.05)
    assert d._now() > t0


def test_header_reports_music_state() -> None:
    d = _display(player=FakePlayer([]))  # type: ignore[arg-type]
    d.player.playing = True  # type: ignore[union-attr]
    assert "MUSIC...playing" in "\n".join(d.draw(130, 40, 1.0).plain_lines())
    d.player.failed = True  # type: ignore[union-attr]
    assert "MUSIC...failed" in "\n".join(d.draw(130, 40, 1.0).plain_lines())


# ---- wiring ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_runner_announces_plan_for_the_screen(tmp_path: Path) -> None:
    events: list[tuple[str, dict]] = []
    config = BenchConfig(provider=ProviderName.OPENAI, models=["m"], tasks=["reasoning", "speed"],
                         runs=2, warmup=0, max_items=4, report_dir=tmp_path / "r",
                         workspace=tmp_path / "ws")
    with patch("llm_bench.runner.get_provider", return_value=OracleProvider()):
        runner = BenchRunner(config, on_progress=lambda e, p: events.append((e, p)))
        await runner.run()
    start = next(p for e, p in events if e == "run_start")
    assert start["plan"] == [("reasoning", 4), ("generation_speed", 2)]
    assert start["total_samples"] == 6
    runner.workspace.cleanup()


def test_music_config_defaults_and_flags() -> None:
    cfg = BenchConfig(models=["m"])
    assert cfg.music is True and cfg.music_path is None
    off = merge_config({"models": ["m"]}, music=False, music_path="~/song.mp3")
    assert off.music is False and off.music_path == Path("~/song.mp3").expanduser()


# ---- keys and music controls ----------------------------------------------------------


def test_keys_map_to_player_actions() -> None:
    fake = FakePlayer([])
    d = _display(player=fake)  # type: ignore[arg-type]
    for key in ("p", "space", "+", "=", "up", "-", "down", "x", "enter"):
        d.handle_key(key)
    assert fake.calls == ["toggle", "toggle", "vol+0.1", "vol+0.1", "vol+0.1", "vol-0.1",
                          "vol-0.1"]


def test_keys_are_ignored_without_a_working_player() -> None:
    _display().handle_key("p")  # no player at all: no crash
    fake = FakePlayer([])
    fake.failed = True
    d = _display(player=fake)  # type: ignore[arg-type]
    d.handle_key("p")
    d.handle_key("+")
    assert fake.calls == []


def test_header_shows_controls_volume_and_pause_state() -> None:
    fake = FakePlayer([])
    fake.playing = True
    d = _display(player=fake)  # type: ignore[arg-type]
    screen = "\n".join(d.draw(130, 40, 1.0).plain_lines())
    assert "MUSIC...playing" in screen and "VOL" in screen and "50%" in screen
    assert "P..Pause" in screen and "+/-..Vol" in screen
    fake.paused = True
    fake.volume = 0.9
    screen = "\n".join(d.draw(130, 40, 1.0).plain_lines())
    assert "MUSIC...paused" in screen and "90%" in screen
    fake.failed = True
    screen = "\n".join(d.draw(130, 40, 1.0).plain_lines())
    assert "MUSIC...failed" in screen and "P..Pause" not in screen


def test_no_player_shows_music_off_without_controls() -> None:
    screen = "\n".join(_display().draw(130, 40, 1.0).plain_lines())
    assert "MUSIC...off" in screen and "P..Pause" not in screen


def test_visuals_go_quiet_while_paused() -> None:
    vis = Visuals(frames=[[1.0] * 8] * 400, fps=16)
    fake = FakePlayer([])
    fake.playing = True
    d = BenchLiveDisplay(MODELS, ["reasoning"], visuals=vis, player=fake)  # type: ignore[arg-type]
    d._advance_levels(1.0)
    assert all(v == 1.0 for v in d.state.levels)
    fake.paused = True
    for k in range(40):
        d._advance_levels(1.0)
    assert all(v < 0.01 for v in d.state.levels)  # bars fall to silence, not frozen at max


def test_keyreader_only_runs_on_a_real_tty_inside_the_live_screen() -> None:
    order: list[str] = []
    d = _display(player=FakePlayer(order))  # type: ignore[arg-type]
    d._console = _console(order)
    with d:  # pytest's stdin is not a TTY: must not start reading it
        assert d._keys is not None and d._keys._thread is None
    assert d._keys is None


# ---- playlist UI ----------------------------------------------------------------------


def test_skip_keys_map_to_next_and_prev() -> None:
    fake = FakePlayer([])
    d = _display(player=fake)  # type: ignore[arg-type]
    for key in ("n", "N", "right", "b", "B", "left"):
        d.handle_key(key)
    assert fake.calls == ["next", "next", "next", "prev", "prev", "prev"]


class PlaylistPlayer(FakePlayer):
    def __init__(self) -> None:
        super().__init__([])
        from llm_bench.audio.player import Track

        self.tracks = [Track(Path("keygen-01.mp3"), 10.0, Visuals(frames=[[1.0] * 8] * 160)),
                       Track(Path("keygen-02.mp3"), 10.0, Visuals(frames=[[0.0] * 8] * 160))]
        self.index = 0
        self.playing = True

    @property
    def current(self):  # type: ignore[no-untyped-def]
        return self.tracks[self.index]


def test_now_playing_and_prev_next_hint_are_shown() -> None:
    pl = PlaylistPlayer()
    d = _display(player=pl)  # type: ignore[arg-type]
    screen = "\n".join(d.draw(130, 40, 1.0).plain_lines())
    assert "Now playing 1/2: Keygen 01" in screen and "B/N..Prev/Next" in screen
    pl.index = 1
    assert "Now playing 2/2: Keygen 02" in "\n".join(d.draw(130, 40, 1.0).plain_lines())
    pl.failed = True
    assert "Now playing" not in "\n".join(d.draw(130, 40, 1.0).plain_lines())


def test_visuals_come_from_the_current_track() -> None:
    pl = PlaylistPlayer()
    d = BenchLiveDisplay(MODELS, ["reasoning"], visuals=Visuals(synthetic=True),
                         player=pl)  # type: ignore[arg-type]
    d._advance_levels(1.0)
    assert all(v == 1.0 for v in d.state.levels)  # track 1 is all-loud
    pl.index = 1
    for _ in range(40):
        d._advance_levels(1.0)
    assert all(v < 0.01 for v in d.state.levels)  # track 2 is silent: bars follow it


# ---- ETA ------------------------------------------------------------------------------


class Clock:
    def __init__(self) -> None:
        self.now = 1000.0

    def __call__(self) -> float:
        return self.now


def _timed(d: BenchLiveDisplay, clock: Clock, task: str, idx: int, secs: float) -> None:
    d.handle("sample_start", {"model": MODELS[0], "task": task, "run_index": idx})
    clock.now += secs
    d.handle("sample_done", {"model": MODELS[0], "task": task, "run_index": idx,
                             "metrics": {"passed": True}})


def test_eta_prices_each_task_by_its_own_observed_duration(monkeypatch) -> None:
    from llm_bench.ui import live

    clock = Clock()
    monkeypatch.setattr(live.time, "monotonic", clock)
    d = BenchLiveDisplay(["m"], ["a", "b"], runs=1, visuals=Visuals(synthetic=True))
    d.handle("run_start", {"run_id": "r", "models": ["m"], "total_samples": 6,
                           "plan": [("a", 2), ("b", 4)]})
    assert d.state.eta_seconds() is None  # nothing finished yet: no guess
    assert "ETA --:--" in "\n".join(d.draw(130, 40, 0).plain_lines())

    _timed(d, clock, "a", 0, 10.0)
    # 1 more 'a' at 10 s + 4 unseen 'b' at the overall mean (10 s)
    assert d.state.eta_seconds() == 50.0
    _timed(d, clock, "a", 1, 20.0)
    assert d.state.eta_seconds() == 60.0  # 'a' done; 4 'b' at overall mean 15 s
    _timed(d, clock, "b", 0, 2.0)
    assert d.state.eta_seconds() == 6.0  # 3 'b' left at b's own 2 s: cheap tasks stay cheap
    assert "ETA 0:00:06" in "\n".join(d.draw(130, 40, 0).plain_lines())
    for i in (1, 2, 3):
        _timed(d, clock, "b", i, 2.0)
    assert d.state.eta_seconds() == 0.0


def test_eta_scales_with_number_of_models(monkeypatch) -> None:
    from llm_bench.ui import live

    clock = Clock()
    monkeypatch.setattr(live.time, "monotonic", clock)
    d = BenchLiveDisplay(["m1", "m2"], ["a"], runs=1, visuals=Visuals(synthetic=True))
    d.handle("run_start", {"run_id": "r", "models": ["m1", "m2"], "total_samples": 6,
                           "plan": [("a", 3)]})  # 3 samples per model, 2 models
    _timed(d, clock, "a", 0, 5.0)
    assert d.state.eta_seconds() == 25.0  # 5 left at 5 s each


def test_eta_is_zero_when_finished_and_shown_in_compact_mode() -> None:
    d = _display()
    d.handle("run_done", {})
    assert d.state.eta_seconds() == 0.0
    assert "ETA 0:00:00" in "\n".join(d.draw(60, 18, 0).plain_lines())
    assert "ETA" in "\n".join(d.draw(80, 24, 0).plain_lines())
