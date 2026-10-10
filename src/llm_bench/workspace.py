"""Per-run sandbox directory + resource registry with guaranteed teardown."""

from __future__ import annotations

import atexit
import os
import shutil
import signal
import subprocess
import time
import uuid
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import psutil

CleanupHook = Callable[[], Any] | Callable[[], Awaitable[Any]]


@dataclass
class TrackedProcess:
    proc: subprocess.Popen[Any]
    label: str
    owned: bool = True  # only kill processes we started


@dataclass
class Workspace:
    """Isolated run directory; tracks procs/files/models for cleanup."""

    run_id: str
    root: Path
    keep: bool = False
    _procs: list[TrackedProcess] = field(default_factory=list)
    _hooks: list[CleanupHook] = field(default_factory=list)
    _ports: list[int] = field(default_factory=list)
    _closed: bool = False
    _signal_installed: bool = False

    @property
    def raw_dir(self) -> Path:
        return self.root / "raw"

    @property
    def tmp_dir(self) -> Path:
        return self.root / "tmp"

    @property
    def report_dir(self) -> Path:
        return self.root / "report"

    def ensure_dirs(self) -> None:
        for d in (self.raw_dir, self.tmp_dir, self.report_dir):
            d.mkdir(parents=True, exist_ok=True)

    def tmp_path(self, name: str) -> Path:
        path = self.tmp_dir / name
        path.parent.mkdir(parents=True, exist_ok=True)
        return path

    def register_process(
        self,
        proc: subprocess.Popen[Any],
        *,
        label: str,
        owned: bool = True,
    ) -> None:
        self._procs.append(TrackedProcess(proc=proc, label=label, owned=owned))

    def register_port(self, port: int) -> None:
        self._ports.append(port)

    def register_cleanup(self, hook: CleanupHook) -> None:
        self._hooks.append(hook)

    def install_signal_handlers(self) -> None:
        if self._signal_installed:
            return
        self._signal_installed = True

        def _handler(signum: int, _frame: Any) -> None:
            self.cleanup()
            # Re-raise default behavior after cleanup
            signal.signal(signum, signal.SIG_DFL)
            os.kill(os.getpid(), signum)

        for sig in (signal.SIGINT, signal.SIGTERM):
            try:
                signal.signal(sig, _handler)
            except (ValueError, OSError):
                # Not in main thread
                pass
        atexit.register(self.cleanup)

    def alive_owned_pids(self) -> list[int]:
        pids: list[int] = []
        for tracked in self._procs:
            if not tracked.owned:
                continue
            if tracked.proc.poll() is None:
                pids.append(tracked.proc.pid)
        return pids

    def _kill_process_tree(self, proc: subprocess.Popen[Any]) -> None:
        if proc.poll() is not None:
            return
        try:
            parent = psutil.Process(proc.pid)
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            try:
                proc.kill()
            except Exception:
                pass
            return

        children = parent.children(recursive=True)
        for child in children:
            try:
                child.terminate()
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                pass
        try:
            parent.terminate()
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass

        _, alive = psutil.wait_procs([parent, *children], timeout=3)
        for p in alive:
            try:
                p.kill()
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                pass
        try:
            proc.wait(timeout=1)
        except Exception:
            pass

    def cleanup(self, *, keep_report: bool | None = None) -> None:
        """Stop owned resources and remove sandbox files.

        Always stops owned child processes and runs cleanup hooks.
        Removes the workspace directory unless keep/keep_report says otherwise.
        """
        if self._closed:
            return
        self._closed = True

        # Hooks first (unload models, close clients) — reverse order
        for hook in reversed(self._hooks):
            try:
                result = hook()
                # Support sync hooks only at cleanup time; async runners
                # should wrap awaitables before registering.
                if hasattr(result, "__await__"):
                    # Best-effort: ignore un-awaited coroutine at atexit
                    try:
                        result.close()  # type: ignore[attr-defined]
                    except Exception:
                        pass
            except Exception:
                pass

        for tracked in reversed(self._procs):
            if tracked.owned:
                try:
                    self._kill_process_tree(tracked.proc)
                except Exception:
                    pass

        # Always wipe tmp
        if self.tmp_dir.exists():
            shutil.rmtree(self.tmp_dir, ignore_errors=True)

        retain = self.keep if keep_report is None else keep_report
        if retain:
            return

        # Remove entire workspace
        if self.root.exists():
            shutil.rmtree(self.root, ignore_errors=True)
            # Remove empty parent run roots we created
            parent = self.root.parent
            try:
                if parent.exists() and not any(parent.iterdir()):
                    parent.rmdir()
            except OSError:
                pass

    def __enter__(self) -> Workspace:
        self.ensure_dirs()
        self.install_signal_handlers()
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        self.cleanup()


def default_workspace_root() -> Path:
    env = os.environ.get("LLM_BENCH_WORKSPACE")
    if env:
        return Path(env).expanduser()
    xdg = os.environ.get("XDG_CACHE_HOME")
    if xdg:
        return Path(xdg) / "llm-bench" / "runs"
    return Path.home() / ".cache" / "llm-bench" / "runs"


def create_workspace(
    *,
    root: Path | None = None,
    run_id: str | None = None,
    keep: bool = False,
) -> Workspace:
    rid = run_id or time.strftime("%Y%m%d-%H%M%S") + "-" + uuid.uuid4().hex[:8]
    base = root or default_workspace_root()
    ws = Workspace(run_id=rid, root=base / rid, keep=keep)
    ws.ensure_dirs()
    return ws
