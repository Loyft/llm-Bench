"""Play a track in the background by driving a system player; stop it on demand.

No audio library is required: macOS ``afplay`` is used if present, otherwise ``ffplay``,
``mpv`` or ``cvlc``. If none exists (or the player exits straight away) playback is
reported as failed and the UI keeps running silently.
"""

from __future__ import annotations

import atexit
import os
import shutil
import signal
import subprocess
import tempfile
import threading
import time
from dataclasses import dataclass
from pathlib import Path

from llm_bench.audio.analysis import Visuals

AUDIO_SUFFIXES = {".m4a", ".mp3", ".wav", ".ogg", ".flac", ".aac", ".mp4", ".xm", ".mod", ".it"}
ENV_VAR = "LLM_BENCH_MUSIC"
# afplay spends a moment starting up; shift the visuals so they line up with what is heard.
START_LATENCY_S = 0.12


def _audio_in(directory: Path) -> list[Path]:
    if not directory.is_dir():
        return []
    return sorted(
        p for p in directory.iterdir() if p.is_file() and p.suffix.lower() in AUDIO_SUFFIXES
    )


def find_playlist(explicit: str | os.PathLike[str] | None = None) -> list[Path]:
    """Resolve the playlist (in play order).

    ``--music`` (file or folder) > ``$LLM_BENCH_MUSIC`` (file or folder) > bundled tracks >
    ``~/.config/llm-bench/music/``. An explicit choice that does not exist yields an empty
    list rather than silently falling back.
    """
    for candidate in (explicit, os.environ.get(ENV_VAR)):
        if candidate:
            p = Path(candidate).expanduser()
            if p.is_dir():
                return _audio_in(p)
            return [p] if p.is_file() else []
    bundled = Path(__file__).resolve().parent.parent / "assets" / "music"
    user = Path.home() / ".config" / "llm-bench" / "music"
    return _audio_in(bundled) or _audio_in(user)


def track_title(path: Path) -> str:
    """'keygen-03.mp3' -> 'Keygen 03'."""
    return path.stem.replace("_", " ").replace("-", " ").strip().title() or path.name


@dataclass
class Track:
    path: Path
    duration: float | None = None
    visuals: Visuals | None = None

    @property
    def title(self) -> str:
        return track_title(self.path)


VOLUME_STEP = 0.1
VOLUME_DEBOUNCE_S = 0.4  # coalesce rapid key presses into one player restart


def player_backend() -> str | None:
    """Name of the first available backend, or None."""
    for name in ("afplay", "ffplay", "mpv", "cvlc"):
        if shutil.which(name):
            return name
    return None


def player_command(
    path: Path, volume: float = 1.0, start: float = 0.0
) -> tuple[list[str], bool] | None:
    """(argv, supports_seek) for the first available backend, or None.

    ``afplay`` cannot seek, so callers feed it a trimmed copy when ``start`` > 0.
    """
    backend = player_backend()
    p = str(path)
    if backend == "afplay":
        return ["afplay", "-v", f"{volume:.2f}", p], False
    if backend == "ffplay":
        return (
            ["ffplay", "-nodisp", "-autoexit", "-loglevel", "quiet",
             "-volume", str(int(volume * 100)), "-ss", f"{start:.2f}", p],
            True,
        )
    if backend == "mpv":
        return (
            ["mpv", "--no-video", "--really-quiet", f"--volume={int(volume * 100)}",
             f"--start={start:.2f}", p],
            True,
        )
    if backend == "cvlc":
        return (
            ["cvlc", "--play-and-exit", "--quiet", f"--gain={volume:.2f}",
             f"--start-time={start:.2f}", p],
            True,
        )
    return None


RESTART_IF_PAST_S = 3.0  # "previous" restarts the current song when it is further in than this


class MusicPlayer:
    """Plays ``tracks`` one after another (wrapping around) until :meth:`stop`.

    Pause freezes the player process (SIGSTOP) so resume continues mid-note. Volume is a
    start-up flag of the backends, so changing it restarts playback at the same position
    (debounced, so holding a key does not stutter). :meth:`next` / :meth:`prev` skip tracks.
    """

    def __init__(
        self,
        tracks: list[Track] | Path | None,
        *,
        duration: float | None = None,
        volume: float = 0.7,
    ) -> None:
        if tracks is None:
            tracks = []
        elif isinstance(tracks, Path):  # single-file convenience
            tracks = [Track(tracks, duration)]
        self.tracks: list[Track] = list(tracks)
        self.index = 0
        self.volume = _clamp_volume(volume)
        self.failed = False
        self.paused = False
        self._proc: subprocess.Popen[bytes] | None = None
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
        self._lock = threading.RLock()
        self._t0: float | None = None  # monotonic time the current process started
        self._base = 0.0  # track offset the current process started at
        self._paused_total = 0.0
        self._pause_at: float | None = None
        self._restart_at: float | None = None  # set => loop respawns this track from here
        self._jump_to: int | None = None  # set => loop respawns this track index from 0
        self._volume_dirty = False
        self._timer: threading.Timer | None = None
        self._tmpdir: str | None = None
        atexit.register(self.stop)

    # -- state ---------------------------------------------------------------------------

    @property
    def current(self) -> Track | None:
        return self.tracks[self.index % len(self.tracks)] if self.tracks else None

    @property
    def path(self) -> Path | None:
        cur = self.current
        return cur.path if cur else None

    @property
    def duration(self) -> float | None:
        cur = self.current
        return cur.duration if cur else None

    @property
    def available(self) -> bool:
        return bool(self.tracks) and player_backend() is not None

    @property
    def playing(self) -> bool:
        return self._t0 is not None and not self._stop.is_set() and not self.failed

    def position(self) -> float | None:
        """Seconds into the current track, frozen while paused; None when nothing plays."""
        t0 = self._t0
        if t0 is None or not self.playing:
            return None
        ref = self._pause_at if self._pause_at is not None else time.monotonic()
        pos = self._base + max(0.0, ref - t0 - self._paused_total - START_LATENCY_S)
        dur = self.duration
        return pos % dur if dur else pos

    # -- controls ------------------------------------------------------------------------

    def start(self) -> bool:
        if not self.tracks or self._thread is not None:
            return False
        if player_backend() is None:
            self.failed = True
            return False
        self._stop.clear()
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()
        return True

    def pause(self) -> None:
        with self._lock:
            proc = self._proc
            if self.paused or not self.playing or proc is None or proc.poll() is not None:
                return
            self._signal(proc, signal.SIGSTOP)
            self.paused = True
            self._pause_at = time.monotonic()

    def resume(self) -> None:
        with self._lock:
            proc = self._proc
            if not self.paused or proc is None:
                return
            now = time.monotonic()
            if self._volume_dirty:
                # volume changed while paused: respawn at the frozen position
                self._restart_at = self.position() or 0.0
                self._volume_dirty = False
                self.paused, self._pause_at = False, None
                self._kill(proc)
                return
            self._paused_total += now - (self._pause_at or now)
            self.paused, self._pause_at = False, None
            self._signal(proc, signal.SIGCONT)

    def toggle_pause(self) -> None:
        if self.paused:
            self.resume()
        else:
            self.pause()

    def next(self) -> None:
        """Skip to the next track (wraps to the first)."""
        self._skip(+1)

    def prev(self) -> None:
        """Restart the current track if it is well under way, else go to the previous one."""
        pos = self.position()
        if pos is not None and pos > RESTART_IF_PAST_S:
            self._skip(0)
        else:
            self._skip(-1)

    def _skip(self, step: int) -> None:
        with self._lock:
            proc = self._proc
            if not self.tracks or proc is None or not self.playing:
                return
            # several quick presses accumulate from the pending target, not the old track
            base = self._jump_to if self._jump_to is not None else self.index
            self._jump_to = (base + step) % len(self.tracks)
            self._restart_at = None
            self._volume_dirty = False  # the new process starts at the current volume anyway
            self.paused, self._pause_at = False, None
            self._kill(proc)

    def set_volume(self, volume: float) -> None:
        volume = _clamp_volume(volume)
        with self._lock:
            if volume == self.volume:
                return
            self.volume = volume
            self._volume_dirty = True
            if self.paused or not self.playing:
                return  # applied on resume / next spawn
            if self._timer is not None:
                self._timer.cancel()
            self._timer = threading.Timer(VOLUME_DEBOUNCE_S, self._apply_volume)
            self._timer.daemon = True
            self._timer.start()

    def adjust_volume(self, delta: float) -> None:
        self.set_volume(self.volume + delta)

    def _apply_volume(self) -> None:
        with self._lock:
            proc = self._proc
            if not self._volume_dirty or self.paused or proc is None or self._stop.is_set():
                return
            if self._jump_to is not None:
                return  # a skip is already respawning at the new volume
            self._volume_dirty = False
            self._restart_at = self.position() or 0.0
            self._kill(proc)

    # -- process management ----------------------------------------------------------------

    def _command(self, start: float) -> tuple[list[str], float] | None:
        """argv plus the track offset it will actually start from."""
        path = self.path
        if path is None:
            return None
        spec = player_command(path, self.volume, start)
        if spec is None:
            return None
        cmd, can_seek = spec
        if start <= 0.5 or can_seek:
            return cmd, (start if can_seek else 0.0)
        trimmed = self._trim(start)
        if trimmed is None:
            return cmd, 0.0  # cannot seek: restart the track from the top
        cmd[-1] = str(trimmed)
        return cmd, start

    def _trim(self, start: float) -> Path | None:
        """Stream-copy of the current track from ``start`` (for backends that cannot seek)."""
        ffmpeg = shutil.which("ffmpeg")
        path = self.path
        if not ffmpeg or path is None:
            return None
        try:
            if self._tmpdir is None:
                self._tmpdir = tempfile.mkdtemp(prefix="llm-bench-music-")
            out = Path(self._tmpdir) / f"seek{path.suffix or '.m4a'}"
            subprocess.run(
                [ffmpeg, "-loglevel", "error", "-y", "-ss", f"{start:.2f}", "-i", str(path),
                 "-vn", "-c", "copy", str(out)],
                capture_output=True, timeout=10, check=True,
            )
            return out
        except (subprocess.SubprocessError, OSError):
            return None

    def _loop(self) -> None:
        start = 0.0
        quick_failures = 0
        while not self._stop.is_set():
            spec = self._command(start)
            if spec is None:
                self.failed = True
                return
            cmd, eff = spec
            spawned = time.monotonic()
            try:
                proc = subprocess.Popen(
                    cmd,
                    stdin=subprocess.DEVNULL,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    start_new_session=True,
                )
            except OSError:
                self.failed = True
                return
            with self._lock:
                self._proc = proc
                if self._stop.is_set():  # stop() raced with us
                    self._kill(proc)
                    return
                self._base, self._t0 = eff, spawned
                self._paused_total, self._pause_at = 0.0, None
                self.paused = False
            proc.wait()
            if self._stop.is_set():
                return
            with self._lock:
                restart, self._restart_at = self._restart_at, None
                jump, self._jump_to = self._jump_to, None
                if jump is not None:
                    self.index = jump
            if jump is not None:
                start, quick_failures = 0.0, 0
                continue
            if restart is not None:
                start = restart
                continue
            dur = self.duration
            remaining = (dur - eff) if dur else 99.0
            if time.monotonic() - spawned < 1.0 and remaining > 2.0:
                # exited immediately: bad file / no audio device. Skip a bad track, but give
                # up when every track has failed in a row instead of spinning.
                quick_failures += 1
                if quick_failures >= len(self.tracks):
                    self.failed = True
                    return
            else:
                quick_failures = 0
            with self._lock:
                self.index = (self.index + 1) % len(self.tracks)  # next song, wrapping
            start = 0.0

    @staticmethod
    def _signal(proc: subprocess.Popen[bytes], sig: int) -> None:
        try:
            os.killpg(proc.pid, sig)
        except (ProcessLookupError, PermissionError, OSError):
            try:
                proc.send_signal(sig)
            except OSError:
                pass

    @classmethod
    def _kill(cls, proc: subprocess.Popen[bytes]) -> None:
        # a stopped process only dies after it is continued
        cls._signal(proc, signal.SIGCONT)
        cls._signal(proc, signal.SIGTERM)

    def stop(self) -> None:
        self._stop.set()
        with self._lock:
            if self._timer is not None:
                self._timer.cancel()
            proc = self._proc
            if proc is not None and proc.poll() is None:
                self._kill(proc)
        thread = self._thread
        if thread is not None and thread is not threading.current_thread():
            thread.join(timeout=2.0)
        if proc is not None and proc.poll() is None:
            self._signal(proc, signal.SIGKILL)
        if self._tmpdir:
            shutil.rmtree(self._tmpdir, ignore_errors=True)
            self._tmpdir = None
        self._thread = None
        self._t0 = None
        self.paused = False


def _clamp_volume(v: float) -> float:
    return max(0.0, min(1.0, round(v * 20) / 20))  # steps of 0.05
