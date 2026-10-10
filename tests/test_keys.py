"""Key parsing and the cbreak reader."""

from __future__ import annotations

import os
import time

from llm_bench.ui.keys import KeyReader, parse_keys


def test_parse_keys_plain_space_and_arrows() -> None:
    assert parse_keys("p +-") == ["p", "space", "+", "-"]
    assert parse_keys("\x1b[A\x1b[B") == ["up", "down"]
    assert parse_keys("\x1bOA") == ["up"]  # application-mode arrows
    assert parse_keys("a\x1b[Cb") == ["a", "right", "b"]


def _wait(cond, timeout: float = 2.0) -> bool:  # type: ignore[no-untyped-def]
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        if cond():
            return True
        time.sleep(0.02)
    return False


def test_reader_delivers_keys_from_a_pipe() -> None:
    r, w = os.pipe()
    got: list[str] = []
    reader = KeyReader(got.append, fd=r, allow_non_tty=True)
    assert reader.start()
    os.write(w, b"p\x1b[A-")
    assert _wait(lambda: got == ["p", "up", "-"])
    reader.stop()
    os.close(r)
    os.close(w)


def test_reader_survives_a_failing_handler() -> None:
    r, w = os.pipe()
    got: list[str] = []

    def handler(key: str) -> None:
        got.append(key)
        if key == "x":
            raise RuntimeError("boom")

    reader = KeyReader(handler, fd=r, allow_non_tty=True)
    reader.start()
    os.write(w, b"xy")
    assert _wait(lambda: got == ["x", "y"])
    reader.stop()
    os.close(r)
    os.close(w)


def test_reader_refuses_non_tty_by_default() -> None:
    r, w = os.pipe()
    reader = KeyReader(lambda k: None, fd=r)
    assert not reader.start()  # would otherwise swallow someone else's piped input
    assert not KeyReader(lambda k: None, fd=-1).start()
    os.close(r)
    os.close(w)
