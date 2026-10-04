"""Collect host / machine info for run manifests."""

from __future__ import annotations

import platform
import sys

import psutil

from llm_bench.schema import HostInfo


def collect_host_info() -> HostInfo:
    cpu = platform.processor() or platform.machine()
    try:
        freq = psutil.cpu_freq()
        if freq and cpu:
            cpu = f"{cpu} @ {freq.current:.0f}MHz"
    except Exception:
        pass

    ram_gb = psutil.virtual_memory().total / (1024**3)
    gpu = _detect_gpu()

    return HostInfo(
        os=f"{platform.system()} {platform.release()}",
        platform=platform.platform(),
        cpu=cpu or "unknown",
        ram_gb=round(ram_gb, 2),
        python=sys.version.split()[0],
        gpu=gpu,
    )


def _detect_gpu() -> str | None:
    # macOS metal / nvidia-smi best-effort
    try:
        import subprocess

        if platform.system() == "Darwin":
            out = subprocess.check_output(
                ["system_profiler", "SPDisplaysDataType"],
                text=True,
                timeout=5,
                stderr=subprocess.DEVNULL,
            )
            for line in out.splitlines():
                if "Chipset Model" in line or "Chipset" in line:
                    return line.split(":", 1)[-1].strip()
        else:
            out = subprocess.check_output(
                ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
                text=True,
                timeout=5,
                stderr=subprocess.DEVNULL,
            )
            names = [l.strip() for l in out.splitlines() if l.strip()]
            if names:
                return ", ".join(names)
    except Exception:
        pass
    return None
