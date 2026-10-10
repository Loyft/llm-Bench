"""Spectrum analysis, music discovery, and the background player lifecycle."""

from __future__ import annotations

import math
import time
from array import array
from pathlib import Path

import pytest

from llm_bench.audio import analysis, player
from llm_bench.audio.analysis import BANDS, CENTRES, SAMPLE_RATE, Visuals, band_energies, normalise


def _tone(freq: float, seconds: float = 1.0) -> array:
    n = int(SAMPLE_RATE * seconds)
    wave = (12000 * math.sin(2 * math.pi * freq * i / SAMPLE_RATE) for i in range(n))
    return array("h", (int(v) for v in wave))


@pytest.mark.parametrize("band", [0, 3, 5, 7])
def test_tone_lights_up_its_own_band(band: int) -> None:
    raw = band_energies(_tone(CENTRES[band]))
    mean = [sum(col) / len(col) for col in zip(*raw)]
    assert max(range(BANDS), key=lambda b: mean[b]) == band


def test_normalise_bounds_and_silence() -> None:
    frames = normalise(band_energies(_tone(800)))
    assert frames and all(0.0 <= v <= 1.0 for row in frames for v in row)
    assert normalise([]) == []
    silent = normalise(band_energies(array("h", [0] * SAMPLE_RATE)))
    assert all(v == 0.0 for row in silent for v in row)


def test_visuals_wrap_and_duration() -> None:
    v = Visuals(frames=[[0.1] * BANDS, [0.9] * BANDS], fps=2)
    assert v.duration == 1.0
    assert v.level(0, 0) == 0.1 and v.level(3, 0) == 0.9  # wraps around the loop
    assert v.levels_at(0.6) == [0.9] * BANDS


def test_note_needs_an_onset() -> None:
    steady = Visuals(frames=[[0.8] * BANDS] * 40)
    assert steady.note(10, 0) is None  # loud but not new
    hit = Visuals(frames=([[0.0] * BANDS] * 10) + ([[0.9] * BANDS] * 10))
    name, ins, vol = hit.note(5, 2)  # row 5 = frames 10-11: silence -> loud
    assert ins == 3 and 16 <= vol <= 64 and len(name) == 3


def test_synthetic_visuals_are_bounded_and_deterministic() -> None:
    v = Visuals(synthetic=True)
    vals = [v.level(f, b) for f in range(200) for b in range(BANDS)]
    assert all(0.0 <= x <= 1.0 for x in vals) and max(vals) > 0.6
    assert v.level(37, 2) == v.level(37, 2)
    assert v.duration is None
    assert len(v.wave(5.0, 12)) == 12


def test_load_visuals_without_file_is_synthetic(tmp_path: Path) -> None:
    assert analysis.load_visuals(None).synthetic
    assert analysis.load_visuals(tmp_path / "missing.m4a").synthetic


def test_load_visuals_falls_back_when_decoding_fails(tmp_path: Path, monkeypatch) -> None:
    f = tmp_path / "x.m4a"
    f.write_bytes(b"not audio")
    monkeypatch.setattr(analysis.Path, "home", lambda: tmp_path)
    monkeypatch.setattr(analysis, "_decode", lambda p: None)
    assert analysis.load_visuals(f).synthetic


def test_load_visuals_uses_cache(tmp_path: Path, monkeypatch) -> None:
    f = tmp_path / "x.m4a"
    f.write_bytes(b"123")
    monkeypatch.setattr(analysis.Path, "home", lambda: tmp_path)
    calls = []

    def fake_decode(p):  # type: ignore[no-untyped-def]
        calls.append(p)
        return _tone(800, 2.0)

    monkeypatch.setattr(analysis, "_decode", fake_decode)
    first = analysis.load_visuals(f)
    second = analysis.load_visuals(f)
    assert not first.synthetic and len(second.frames) == len(first.frames)
    assert len(calls) == 1


# ---- discovery ------------------------------------------------------------------------


def _touch(d: Path, *names: str) -> list[Path]:
    d.mkdir(parents=True, exist_ok=True)
    out = []
    for n in names:
        f = d / n
        f.write_bytes(b"x")
        out.append(f)
    return out


def test_find_playlist_resolution_order(tmp_path: Path, monkeypatch) -> None:
    song = _touch(tmp_path, "a.mp3")[0]
    folder = tmp_path / "set"
    b, a = _touch(folder, "b.wav", "a.flac")
    _touch(folder, "notes.txt")
    monkeypatch.delenv(player.ENV_VAR, raising=False)
    assert player.find_playlist(song) == [song]
    assert player.find_playlist(folder) == [a, b]  # a folder is a playlist, sorted, audio only
    assert player.find_playlist(tmp_path / "nope.mp3") == []  # explicit but missing: no fallback
    monkeypatch.setenv(player.ENV_VAR, str(folder))
    assert player.find_playlist() == [a, b]
    assert player.find_playlist(song) == [song]  # explicit beats env


def test_find_playlist_falls_back_to_user_dir(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.delenv(player.ENV_VAR, raising=False)
    monkeypatch.setattr(player.Path, "home", lambda: tmp_path)
    monkeypatch.setattr(player, "_audio_in",
                        lambda d: [] if "assets" in str(d) else _real_audio_in(d))
    user = tmp_path / ".config" / "llm-bench" / "music"
    _touch(user, "readme.txt", "z.flac", "y.mp3")
    assert [p.name for p in player.find_playlist()] == ["y.mp3", "z.flac"]


_real_audio_in = player._audio_in


def test_bundled_playlist_ships_mp3_only_in_order() -> None:
    bundled = Path(player.__file__).resolve().parent.parent / "assets" / "music"
    tracks = player._audio_in(bundled)
    assert len(tracks) >= 8
    assert {t.suffix for t in tracks} == {".mp3"}  # no video / m4a sources in the repo
    assert [t.name for t in tracks] == sorted(t.name for t in tracks)
    assert not list(bundled.glob("*.mp4"))


def test_track_title() -> None:
    assert player.track_title(Path("keygen-03.mp3")) == "Keygen 03"
    assert player.track_title(Path("/x/my_cool-song.m4a")) == "My Cool Song"


def test_player_command_prefers_afplay_and_carries_volume(monkeypatch) -> None:
    avail = {"afplay": "/usr/bin/afplay", "ffplay": "/x/ffplay", "mpv": "/x/mpv"}
    monkeypatch.setattr(player.shutil, "which", lambda n: avail.get(n))
    cmd, can_seek = player.player_command(Path("s.m4a"), volume=0.5)  # type: ignore[misc]
    assert cmd[:3] == ["afplay", "-v", "0.50"] and not can_seek
    del avail["afplay"]
    cmd, can_seek = player.player_command(Path("s.m4a"), volume=0.5, start=12.5)  # type: ignore[misc]
    assert cmd[0] == "ffplay" and can_seek
    assert cmd[cmd.index("-volume") + 1] == "50" and cmd[cmd.index("-ss") + 1] == "12.50"
    del avail["ffplay"]
    cmd, _ = player.player_command(Path("s.m4a"), volume=0.3, start=2)  # type: ignore[misc]
    assert "--volume=30" in cmd and "--start=2.00" in cmd
    avail.clear()
    assert player.player_command(Path("s.m4a")) is None


# ---- player lifecycle -----------------------------------------------------------------


def _fake_backend(monkeypatch, cmd=None):  # type: ignore[no-untyped-def]
    """Pretend a backend exists; the 'player' is a harmless long sleep. Records spawn args."""
    class Spawns(list):
        paths: list[Path]

    spawns = Spawns()
    spawns.paths = []
    monkeypatch.setattr(player, "player_backend", lambda: "fake")

    def fake_command(path, volume=1.0, start=0.0):  # type: ignore[no-untyped-def]
        spawns.append((volume, start))
        spawns.paths.append(path)
        chosen = cmd(path) if callable(cmd) else cmd
        return (chosen or ["sleep", "30"]), True

    monkeypatch.setattr(player, "player_command", fake_command)
    return spawns


def _wait(cond, timeout: float = 3.0) -> bool:  # type: ignore[no-untyped-def]
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        if cond():
            return True
        time.sleep(0.02)
    return False


def test_player_start_position_and_stop_kills_process(tmp_path: Path, monkeypatch) -> None:
    _fake_backend(monkeypatch)
    mp = player.MusicPlayer(tmp_path / "x.m4a", duration=10.0)
    assert mp.position() is None  # not started
    assert mp.start()
    assert _wait(lambda: mp._proc is not None and mp.playing)
    proc = mp._proc
    assert proc is not None and proc.poll() is None
    time.sleep(0.25)
    pos = mp.position()
    assert pos is not None and 0.0 <= pos < 10.0
    mp.stop()
    assert proc.poll() is not None  # music really stopped
    assert not mp.playing and mp.position() is None
    mp.stop()  # idempotent


def test_player_marks_failure_when_backend_exits_immediately(tmp_path: Path, monkeypatch) -> None:
    _fake_backend(monkeypatch, cmd=["true"])
    mp = player.MusicPlayer(tmp_path / "x.m4a")
    mp.start()
    assert _wait(lambda: mp.failed)  # no respawn loop, no crash
    assert not mp.playing
    mp.stop()


def test_player_without_backend_or_file(tmp_path: Path, monkeypatch) -> None:
    assert not player.MusicPlayer(None).start()
    monkeypatch.setattr(player, "player_backend", lambda: None)
    mp = player.MusicPlayer(tmp_path / "x.m4a")
    assert not mp.available and not mp.start() and mp.failed


def test_position_wraps_at_duration(tmp_path: Path, monkeypatch) -> None:
    mp = player.MusicPlayer(tmp_path / "x.m4a", duration=2.0)
    mp._t0 = time.monotonic() - 5.0
    assert 0.0 <= (mp.position() or 0.0) < 2.0
    mp.stop()


# ---- pause / resume / volume ----------------------------------------------------------


def _heartbeat_cmd(path: Path) -> list[str]:
    """A fake 'player' that proves whether it is running: it appends to a file every 50 ms."""
    import sys

    code = ("import sys, time\nwhile True:\n"
            "    open(sys.argv[1], 'a').write('x')\n    time.sleep(0.05)")
    return [sys.executable, "-c", code, str(path)]


def test_pause_freezes_process_and_position_then_resume_continues(tmp_path: Path,
                                                                  monkeypatch) -> None:
    beat = tmp_path / "beat"
    _fake_backend(monkeypatch, cmd=_heartbeat_cmd(beat))
    mp = player.MusicPlayer(tmp_path / "x.m4a", duration=100.0)
    mp.start()
    assert _wait(lambda: mp.playing and beat.exists() and beat.stat().st_size > 3)
    mp.pause()
    assert mp.paused
    time.sleep(0.2)  # let any in-flight write land
    size, frozen = beat.stat().st_size, mp.position()
    time.sleep(0.5)
    assert beat.stat().st_size == size  # the player process is genuinely frozen
    assert mp.position() == frozen  # and the playback clock with it
    mp.resume()
    assert not mp.paused and _wait(lambda: beat.stat().st_size > size)  # running again
    time.sleep(0.3)
    after = mp.position()
    assert after is not None and frozen is not None and after > frozen
    assert after - frozen < 0.9  # the 0.5 s spent paused was not counted as playback
    mp.stop()


def test_toggle_pause_and_noops_when_idle(tmp_path: Path, monkeypatch) -> None:
    _fake_backend(monkeypatch)
    mp = player.MusicPlayer(tmp_path / "x.m4a", duration=100.0)
    mp.pause()
    mp.resume()
    mp.toggle_pause()  # nothing playing yet: all no-ops, no crash
    assert not mp.paused
    mp.start()
    assert _wait(lambda: mp.playing)
    mp.toggle_pause()
    assert mp.paused
    mp.toggle_pause()
    assert not mp.paused
    mp.stop()


def test_stop_while_paused_still_kills_the_process(tmp_path: Path, monkeypatch) -> None:
    _fake_backend(monkeypatch)
    mp = player.MusicPlayer(tmp_path / "x.m4a", duration=100.0)
    mp.start()
    assert _wait(lambda: mp.playing)
    proc = mp._proc
    mp.pause()
    mp.stop()
    assert proc is not None and proc.poll() is not None  # a stopped process still dies


def test_volume_is_clamped_and_stepped() -> None:
    mp = player.MusicPlayer(None, volume=5)
    assert mp.volume == 1.0
    mp.set_volume(-1)
    assert mp.volume == 0.0
    mp.set_volume(0.33)
    assert mp.volume == 0.35
    mp.adjust_volume(player.VOLUME_STEP)
    assert abs(mp.volume - 0.45) < 1e-9


def test_volume_change_restarts_once_at_same_position(tmp_path: Path, monkeypatch) -> None:
    spawns = _fake_backend(monkeypatch)
    monkeypatch.setattr(player, "VOLUME_DEBOUNCE_S", 0.15)
    mp = player.MusicPlayer(tmp_path / "x.m4a", duration=100.0, volume=0.5)
    mp.start()
    assert _wait(lambda: mp.playing)
    first = mp._proc
    time.sleep(0.4)
    for _ in range(3):  # a burst of key presses = a single restart
        mp.adjust_volume(0.1)
    assert _wait(lambda: len(spawns) == 2, timeout=3.0)
    time.sleep(0.4)
    assert len(spawns) == 2
    assert abs(spawns[1][0] - 0.8) < 1e-9
    assert 0.2 < spawns[1][1] < 3.0  # resumed near where it was, not from 0
    assert first is not None and first.poll() is not None
    assert mp._proc is not first and mp.playing and not mp.failed  # restart is not a failure
    pos = mp.position()
    assert pos is not None and pos >= spawns[1][1]
    mp.stop()


def test_volume_change_while_paused_applies_on_resume(tmp_path: Path, monkeypatch) -> None:
    spawns = _fake_backend(monkeypatch)
    mp = player.MusicPlayer(tmp_path / "x.m4a", duration=100.0, volume=0.5)
    mp.start()
    assert _wait(lambda: mp.playing)
    time.sleep(0.3)
    mp.pause()
    frozen = mp.position()
    mp.set_volume(0.9)
    time.sleep(0.3)
    assert len(spawns) == 1 and mp.paused  # no restart (and no sound) while paused
    mp.resume()
    assert _wait(lambda: len(spawns) == 2)
    assert spawns[1][0] == 0.9 and frozen is not None
    assert abs(spawns[1][1] - frozen) < 0.2
    mp.stop()


def test_afplay_seek_uses_trimmed_copy(tmp_path: Path, monkeypatch) -> None:
    song = tmp_path / "s.m4a"
    song.write_bytes(b"x")
    monkeypatch.setattr(player, "player_backend", lambda: "afplay")
    monkeypatch.setattr(player.shutil, "which", lambda n: "/bin/" + n)
    mp = player.MusicPlayer(song, duration=100.0)
    monkeypatch.setattr(mp, "_trim", lambda start: tmp_path / "seek.m4a")
    cmd, eff = mp._command(40.0)  # type: ignore[misc]
    assert cmd[-1].endswith("seek.m4a") and eff == 40.0
    monkeypatch.setattr(mp, "_trim", lambda start: None)  # ffmpeg missing: restart from the top
    cmd, eff = mp._command(40.0)  # type: ignore[misc]
    assert cmd[-1] == str(song) and eff == 0.0
    cmd, eff = mp._command(0.0)  # type: ignore[misc]
    assert eff == 0.0
    mp.stop()


# ---- playlist -------------------------------------------------------------------------


def _tracks(tmp_path: Path, n: int = 3, duration: float | None = 100.0) -> list[player.Track]:
    return [player.Track(tmp_path / f"t{i}.mp3", duration) for i in range(n)]


def test_tracks_play_one_after_another_and_wrap(tmp_path: Path, monkeypatch) -> None:
    # duration < 2 s so a short-lived process is a finished song, not a failure
    spawns = _fake_backend(monkeypatch, cmd=["sleep", "0.3"])
    mp = player.MusicPlayer(_tracks(tmp_path, 3, duration=1.5))
    mp.start()
    assert _wait(lambda: len(spawns) >= 5, timeout=6.0)
    mp.stop()
    assert [p.name for p in spawns.paths[:5]] == ["t0.mp3", "t1.mp3", "t2.mp3", "t0.mp3", "t1.mp3"]
    assert not mp.failed
    assert all(start == 0.0 for _, start in spawns[:5])


def test_next_and_prev_switch_tracks_and_wrap(tmp_path: Path, monkeypatch) -> None:
    spawns = _fake_backend(monkeypatch)
    mp = player.MusicPlayer(_tracks(tmp_path, 3))
    mp.start()
    assert _wait(lambda: mp.playing)
    first = mp._proc
    mp.next()
    assert _wait(lambda: mp.index == 1 and mp._proc is not first and mp._proc.poll() is None)
    assert first.poll() is not None and spawns.paths[-1].name == "t1.mp3"
    mp.next()
    assert _wait(lambda: mp.index == 2)
    mp.next()
    assert _wait(lambda: mp.index == 0)  # wraps to the first
    assert _wait(lambda: mp.playing)
    mp.prev()  # just started (< 3 s in): go to the previous track, wrapping to the last
    assert _wait(lambda: mp.index == 2)
    assert not mp.failed
    mp.stop()


def test_prev_restarts_current_track_when_well_into_it(tmp_path: Path, monkeypatch) -> None:
    spawns = _fake_backend(monkeypatch)
    mp = player.MusicPlayer(_tracks(tmp_path, 3))
    mp.start()
    assert _wait(lambda: mp.playing)
    mp.next()
    assert _wait(lambda: mp.index == 1 and len(spawns) == 2)
    time.sleep(0.3)
    mp._t0 -= 10.0  # pretend we are 10 s into the song
    n = len(spawns)
    mp.prev()
    assert _wait(lambda: len(spawns) == n + 1)
    assert mp.index == 1 and spawns.paths[-1].name == "t1.mp3" and spawns[-1][1] == 0.0
    mp.stop()


def test_quick_skips_accumulate_from_the_pending_target(tmp_path: Path, monkeypatch) -> None:
    _fake_backend(monkeypatch)
    mp = player.MusicPlayer(_tracks(tmp_path, 5))
    mp.start()
    assert _wait(lambda: mp.playing)
    mp.next(); mp.next(); mp.next()  # noqa: E702 - three presses before the loop reacts
    assert _wait(lambda: mp.index == 3 and mp._proc is not None and mp._proc.poll() is None)
    mp.stop()


def test_skip_while_paused_plays_the_new_track(tmp_path: Path, monkeypatch) -> None:
    _fake_backend(monkeypatch)
    mp = player.MusicPlayer(_tracks(tmp_path, 3))
    mp.start()
    assert _wait(lambda: mp.playing)
    mp.pause()
    assert mp.paused
    mp.next()
    assert _wait(lambda: mp.index == 1 and not mp.paused and mp.playing)
    mp.stop()


def test_skip_is_a_noop_without_playback(tmp_path: Path) -> None:
    mp = player.MusicPlayer(_tracks(tmp_path, 3))
    mp.next()
    mp.prev()
    assert mp.index == 0
    assert player.MusicPlayer([]).current is None and not player.MusicPlayer([]).start()


def test_bad_track_is_skipped_but_all_bad_fails(tmp_path: Path, monkeypatch) -> None:
    spawns = _fake_backend(monkeypatch, cmd=lambda p: ["true"] if p.name == "t0.mp3" else None)
    mp = player.MusicPlayer(_tracks(tmp_path, 3))
    mp.start()
    assert _wait(lambda: len(spawns) >= 2 and mp.index == 1)
    assert mp.playing and not mp.failed  # t0 died instantly; t1 is playing
    mp.stop()
    spawns = _fake_backend(monkeypatch, cmd=["true"])
    mp = player.MusicPlayer(_tracks(tmp_path, 3))
    mp.start()
    assert _wait(lambda: mp.failed)  # every track failed in a row: stop trying
    assert len(spawns) == 3
    mp.stop()


def test_position_and_duration_follow_the_current_track(tmp_path: Path) -> None:
    mp = player.MusicPlayer([player.Track(tmp_path / "a.mp3", 10.0),
                             player.Track(tmp_path / "b.mp3", 50.0)])
    assert mp.duration == 10.0
    mp.index = 1
    assert mp.duration == 50.0 and mp.path == tmp_path / "b.mp3"
    mp._t0 = time.monotonic() - 70.0
    assert 0.0 <= (mp.position() or 0.0) < 50.0  # wraps at the *current* track's length
    mp.stop()
