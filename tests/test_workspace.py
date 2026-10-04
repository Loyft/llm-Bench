"""Workspace sandbox cleanup guarantees."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from llm_bench.workspace import create_workspace


def test_workspace_dirs_and_cleanup(tmp_path: Path) -> None:
    ws = create_workspace(root=tmp_path / "runs", keep=False)
    assert ws.raw_dir.exists()
    assert ws.tmp_dir.exists()
    scratch = ws.tmp_path("foo.txt")
    scratch.write_text("hi", encoding="utf-8")
    assert scratch.exists()
    ws.cleanup()
    assert not ws.root.exists()


def test_workspace_kills_owned_process(tmp_path: Path) -> None:
    ws = create_workspace(root=tmp_path / "runs", keep=False)
    proc = subprocess.Popen(
        [sys.executable, "-c", "import time; time.sleep(60)"],
        start_new_session=True,
    )
    ws.register_process(proc, label="sleeper", owned=True)
    assert proc.poll() is None
    ws.cleanup()
    assert proc.poll() is not None
    assert not ws.root.exists()
    assert ws.alive_owned_pids() == []


def test_keep_workspace_retains_root(tmp_path: Path) -> None:
    ws = create_workspace(root=tmp_path / "runs", keep=True)
    (ws.raw_dir / "x.json").write_text("{}", encoding="utf-8")
    ws.cleanup()
    assert ws.root.exists()
    assert not ws.tmp_dir.exists()
    # manual wipe for fixture hygiene
    import shutil

    shutil.rmtree(ws.root, ignore_errors=True)
