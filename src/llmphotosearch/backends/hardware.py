"""Hardware detection, used to pick the inference engine build and default model."""

import platform
import shutil
import subprocess
from dataclasses import dataclass

import psutil


@dataclass(frozen=True)
class Hardware:
    os: str
    """"windows", "linux" or "macos"."""
    arch: str
    """"x64" or "arm64"."""
    ram_gb: float
    nvidia_vram_gb: float | None
    """VRAM of the largest NVIDIA GPU, None when there is none."""


def detect() -> Hardware:
    os_name = {"Windows": "windows", "Darwin": "macos"}.get(platform.system(), "linux")
    arch = "arm64" if platform.machine().lower() in ("arm64", "aarch64") else "x64"
    return Hardware(os_name, arch, psutil.virtual_memory().total / 2**30, _nvidia_vram_gb())


def _nvidia_vram_gb() -> float | None:
    smi = shutil.which("nvidia-smi")
    if not smi:
        return None
    try:
        out = subprocess.run(
            [smi, "--query-gpu=memory.total", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            timeout=10,
            check=True,
        ).stdout
        return max(float(line) for line in out.split()) / 1024
    except (OSError, subprocess.SubprocessError, ValueError):
        return None
