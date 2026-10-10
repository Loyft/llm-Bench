"""Turn an audio file into per-band energy over time so the visuals follow the real song.

Decoding uses ``ffmpeg`` (if it is missing, or analysis fails, callers get a synthetic
visualizer instead). The result is cached on disk, so the analysis cost is paid once.
"""

from __future__ import annotations

import hashlib
import json
import math
import shutil
import subprocess
from array import array
from dataclasses import dataclass, field
from pathlib import Path

FPS = 16  # analysis frames per second; the pattern view scrolls at FPS/2 rows per second
BANDS = 8
SAMPLE_RATE = 8000
WINDOW = 320  # 40 ms
# Band centres in Hz (below the 4 kHz Nyquist of the decimated signal)
CENTRES = (90, 180, 320, 520, 800, 1200, 1800, 2700)
NOTE_NAMES = ("C-", "C#", "D-", "D#", "E-", "F-", "F#", "G-", "G#", "A-", "A#", "B-")
CACHE_VERSION = 2


def goertzel_power(samples: list[int] | array, start: int, n: int, freq: float, rate: int) -> float:
    """Power of ``freq`` in ``samples[start:start+n]`` (Goertzel algorithm)."""
    k = 2.0 * math.cos(2.0 * math.pi * freq / rate)
    s1 = s2 = 0.0
    for i in range(start, start + n):
        s0 = samples[i] + k * s1 - s2
        s2 = s1
        s1 = s0
    return max(0.0, s1 * s1 + s2 * s2 - k * s1 * s2)


def band_energies(samples: list[int] | array, rate: int = SAMPLE_RATE) -> list[list[float]]:
    """Raw per-frame, per-band magnitudes (unnormalised)."""
    hop = rate // FPS
    frames = max(0, (len(samples) - WINDOW) // hop + 1)
    out: list[list[float]] = []
    for f in range(frames):
        start = f * hop
        out.append(
            [math.sqrt(goertzel_power(samples, start, WINDOW, c, rate)) for c in CENTRES]
        )
    return out


def normalise(raw: list[list[float]]) -> list[list[float]]:
    """Scale each band to 0..1 by its own 98th percentile, with a mild power curve.

    (A log scale saturates on dense chiptune material and pegs every bar at the top.)
    """
    if not raw:
        return []
    scaled: list[list[float]] = []
    for col in zip(*raw):
        ref = sorted(col)[int(0.98 * (len(col) - 1))] or 1.0
        scaled.append([min(1.0, v / ref) ** 0.6 for v in col])
    return [list(row) for row in zip(*scaled)]


def _decode(path: Path) -> array | None:
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        return None
    try:
        proc = subprocess.run(
            [ffmpeg, "-loglevel", "error", "-i", str(path), "-vn", "-ac", "1",
             "-ar", str(SAMPLE_RATE), "-f", "s16le", "-"],
            capture_output=True,
            timeout=60,
            check=True,
        )
    except (subprocess.SubprocessError, OSError):
        return None
    samples = array("h")
    samples.frombytes(proc.stdout[: len(proc.stdout) // 2 * 2])
    return samples


def _cache_path(path: Path) -> Path:
    st = path.stat()
    key = hashlib.sha1(f"{path.resolve()}|{st.st_size}|{st.st_mtime_ns}".encode()).hexdigest()[:16]
    return Path.home() / ".cache" / "llm-bench" / f"spectrum-{key}.json"


@dataclass
class Visuals:
    """Per-frame band levels (0..1) for a track, or a synthetic stand-in."""

    frames: list[list[float]] = field(default_factory=list)
    fps: int = FPS
    synthetic: bool = False

    @property
    def duration(self) -> float | None:
        return len(self.frames) / self.fps if self.frames else None

    def level(self, frame: int, band: int) -> float:
        if self.frames:
            return self.frames[frame % len(self.frames)][band]
        return _synthetic_level(frame, band)

    def levels_at(self, t: float) -> list[float]:
        f = int(t * self.fps)
        return [self.level(f, b) for b in range(BANDS)]

    def note(self, row: int, band: int) -> tuple[str, int, int] | None:
        """Tracker-style event for a pattern row (2 frames): (note, instrument, volume)."""
        f = row * 2
        cur = max(self.level(f, band), self.level(f + 1, band))
        prev = max(self.level(f - 1, band), self.level(f - 2, band))
        if cur < 0.32 or cur - prev < 0.16:
            return None
        name = NOTE_NAMES[(row * 5 + band * 7) % 12]
        octave = 2 + band // 2 + (row // 16) % 2
        return f"{name}{octave}", band + 1, int(16 + cur * 48)

    def wave(self, t: float, n: int) -> list[float]:
        """Recent overall level, oldest first (for the oscilloscope)."""
        f = int(t * self.fps)
        out = []
        for i in range(n):
            lv = [self.level(f - (n - 1 - i), b) for b in range(BANDS)]
            out.append(sum(lv) / BANDS)
        return out


def _synthetic_level(frame: int, band: int) -> float:
    """Deterministic pseudo-music: a beat plus drifting band wobble."""
    beat = max(0.0, math.cos(frame * math.pi / 4)) ** 3 if band < 3 else 0.0
    hat = 0.5 if (frame % 4 == 2 and band >= 5) else 0.0
    wob = 0.5 + 0.5 * math.sin(frame * (0.11 + band * 0.037) + band)
    h = ((frame * 2654435761 + band * 40503) >> 7) & 0xFF
    jitter = h / 255.0 * 0.25
    return max(0.0, min(1.0, 0.18 + 0.35 * wob + 0.45 * beat + hat + jitter - 0.2))


def analyse(path: Path) -> Visuals | None:
    samples = _decode(path)
    if not samples:
        return None
    frames = normalise(band_energies(samples))
    return Visuals(frames=frames) if frames else None


def load_visuals(path: Path | None) -> Visuals:
    """Analyse ``path`` (cached), falling back to synthetic visuals if that is impossible."""
    if path is None or not path.exists():
        return Visuals(synthetic=True)
    cache = None
    try:
        cache = _cache_path(path)
        if cache.exists():
            data = json.loads(cache.read_text())
            if data.get("version") == CACHE_VERSION:
                return Visuals(frames=data["frames"])
    except (OSError, ValueError, KeyError):
        pass
    vis = analyse(path)
    if vis is None:
        return Visuals(synthetic=True)
    if cache is not None:
        try:
            cache.parent.mkdir(parents=True, exist_ok=True)
            payload = {
                "version": CACHE_VERSION,
                "frames": [[round(v, 3) for v in row] for row in vis.frames],
            }
            cache.write_text(json.dumps(payload))
        except OSError:
            pass
    return vis
