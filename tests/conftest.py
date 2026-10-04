"""Shared fixtures — workspace cleanup assertions."""

from __future__ import annotations

import pytest

from llm_bench.workspace import Workspace, create_workspace


@pytest.fixture
def workspace(tmp_path) -> Workspace:
    ws = create_workspace(root=tmp_path / "runs", keep=False)
    yield ws
    # Teardown must leave no owned procs and remove sandbox
    leftover = ws.alive_owned_pids()
    ws.cleanup()
    assert leftover == [], f"owned processes still alive after test: {leftover}"
    assert not ws.root.exists(), f"workspace not removed: {ws.root}"
