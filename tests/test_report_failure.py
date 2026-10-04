"""Failed runs must surface errors instead of looking like empty success."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

from llm_bench.hostinfo import collect_host_info
from llm_bench.report.html import write_html_report
from llm_bench.schema import (
    BenchReport,
    ModelScore,
    ProviderName,
    RunManifest,
    SampleResult,
    TaskSummary,
)


def _failed_report() -> BenchReport:
    manifest = RunManifest(
        run_id="fail-test",
        started_at=datetime.now(timezone.utc),
        finished_at=datetime.now(timezone.utc),
        provider=ProviderName.OLLAMA,
        models=["llama3.2:3b"],
        model_count=1,
        tasks=["generation_speed"],
        context=8192,
        tools=False,
        mode="sequential",
        runs=1,
        warmup=0,
        host=collect_host_info(),
    )
    sample = SampleResult(
        task="setup",
        model="llama3.2:3b",
        provider="ollama",
        run_index=0,
        error="Cannot reach Ollama at http://127.0.0.1:11434. Is the server running?",
    )
    return BenchReport(
        manifest=manifest,
        samples=[sample],
        summaries=[
            TaskSummary(
                task="setup",
                model="llama3.2:3b",
                provider="ollama",
                samples=1,
                errors=1,
            )
        ],
        scores=[
            ModelScore(
                model="llama3.2:3b",
                provider="ollama",
                overall=0.0,
            )
        ],
        mode="profile",
    )


def test_report_ok_false_on_setup_only() -> None:
    report = _failed_report()
    assert not report.ok
    assert len(report.errors) == 1


def test_html_includes_error_banner(tmp_path: Path) -> None:
    dest = tmp_path / "report.html"
    write_html_report(_failed_report(), dest)
    html = dest.read_text(encoding="utf-8")
    assert "FAILED" in html
    assert "Cannot reach Ollama" in html
    assert "Empty report" in html
    assert "Test specifications" in html


def test_profile_html_uses_raw_values_not_bars(tmp_path: Path) -> None:
    from llm_bench.schema import MemoryTimeline

    report = _failed_report()
    # Make it look like a successful profile for layout checks
    report.samples = [
        SampleResult(
            task="generation_speed",
            model="llama3.2:3b",
            provider="ollama",
            run_index=0,
            metrics={
                "tokens_per_sec": 80.0,
                "prefill_tokens_per_sec": 1200.0,
                "ttft_ms": 90.0,
            },
        )
    ]
    report.summaries = [
        TaskSummary(
            task="generation_speed",
            model="llama3.2:3b",
            provider="ollama",
            metrics={
                "tokens_per_sec": 80.0,
                "prefill_tokens_per_sec": 1200.0,
                "ttft_ms": 90.0,
            },
            samples=1,
            errors=0,
        )
    ]
    report.scores = [
        ModelScore(
            model="llama3.2:3b",
            provider="ollama",
            overall=0.9,
            breakdown={"speed": 0.9},
            tasks={"generation_speed": report.summaries[0]},
        )
    ]
    report.mode = "profile"
    report.memory = MemoryTimeline(peak_system_used_mb=1000, samples=[{"t_s": 0, "system_used_mb": 1000}])

    dest = tmp_path / "report.html"
    write_html_report(report, dest)
    html = dest.read_text(encoding="utf-8")
    assert "Test specifications" in html
    assert "Decode speed" in html
    assert "tok/s" in html
    assert "TTFT (ms)" in html
    assert "chartTok" not in html  # no bar charts in profile mode
    assert "80.00" in html or "80.0" in html
