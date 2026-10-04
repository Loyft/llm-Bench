"""Runner tests with mock provider — no leftover workspace/procs."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest

from llm_bench.config import BenchConfig
from llm_bench.report.html import write_html_report
from llm_bench.runner import BenchRunner
from llm_bench.schema import ProviderName
from mock_provider import MockProvider


@pytest.mark.asyncio
async def test_single_model_profile(tmp_path: Path) -> None:
    report_dir = tmp_path / "reports"
    ws_root = tmp_path / "ws"
    config = BenchConfig(
        provider=ProviderName.OPENAI,
        models=["mock-model"],
        tasks=["speed", "quality"],
        tools=False,
        runs=2,
        warmup=1,
        report_dir=report_dir,
        workspace=ws_root,
        keep_workspace=False,
    )
    mock = MockProvider(workspace=None)

    with patch("llm_bench.runner.get_provider", return_value=mock):
        runner = BenchRunner(config)
        ws = runner.workspace
        report = await runner.run()
        assert report.mode == "profile"
        assert report.manifest.model_count == 1
        assert mock.chat_calls >= 2  # warmup + samples
        assert "mock-model" in mock.shutdown_calls

        draft = ws.report_dir / "report.html"
        write_html_report(report, draft)
        assert draft.exists()
        runner.export_report_files(report, html_path=draft)
        exported = report_dir / report.manifest.run_id / "report.html"
        assert exported.exists()
        # Single durable report file — no sibling report.json
        assert not (report_dir / report.manifest.run_id / "report.json").exists()

        # Cleanup must wipe sandbox
        assert ws.alive_owned_pids() == []
        ws.cleanup()
        assert not ws.root.exists()


@pytest.mark.asyncio
async def test_multi_model_compare(tmp_path: Path) -> None:
    config = BenchConfig(
        provider=ProviderName.OPENAI,
        models=["model-a", "model-b"],
        tasks=["speed"],
        tools=False,
        runs=1,
        warmup=0,
        report_dir=tmp_path / "reports",
        workspace=tmp_path / "ws",
        mode="sequential",
    )
    mock = MockProvider()
    with patch("llm_bench.runner.get_provider", return_value=mock):
        runner = BenchRunner(config)
        report = await runner.run()
        assert report.mode == "compare"
        assert len(report.scores) == 2
        runner.workspace.cleanup()
        assert not runner.workspace.root.exists()


def test_config_requires_model() -> None:
    with pytest.raises(Exception):
        BenchConfig(models=[])
