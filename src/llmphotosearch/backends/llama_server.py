"""Bundled llama.cpp server: download, start, stop and query.

The official prebuilt `llama-server` matching the machine is downloaded on first
use, so nothing has to be installed besides this package.
"""

import json
import os
import shutil
import socket
import subprocess
import tarfile
import time
import urllib.error
import urllib.request
import zipfile
from collections.abc import Callable
from pathlib import Path
from typing import Literal

from platformdirs import user_data_path

from llmphotosearch.backends.hardware import Hardware

LLAMA_CPP_BUILD = "b11222"
RELEASE_URL = f"https://github.com/ggml-org/llama.cpp/releases/download/{LLAMA_CPP_BUILD}"

Backend = Literal["cuda", "vulkan", "metal", "cpu"]
ProgressCallback = Callable[[int, int], None]
"""Called with (downloaded bytes, total bytes)."""


class EngineError(RuntimeError):
    pass


def default_backend(hw: Hardware) -> Backend:
    if hw.os == "macos":
        return "metal" if hw.arch == "arm64" else "cpu"
    if hw.nvidia_vram_gb is not None and hw.arch == "x64":
        return "cuda"
    # The Vulkan build also covers AMD/Intel GPUs and iGPUs, and falls back to CPU.
    return "vulkan"


def release_assets(hw: Hardware, backend: Backend) -> list[str]:
    """Archive names to download for this platform and backend."""
    b, arch = LLAMA_CPP_BUILD, hw.arch
    if hw.os == "macos":
        return [f"llama-{b}-bin-macos-{arch}.tar.gz"]
    if hw.os == "windows":
        if backend == "cuda":
            return [
                f"llama-{b}-bin-win-cuda-12.4-x64.zip",
                "cudart-llama-bin-win-cuda-12.4-x64.zip",
            ]
        return [f"llama-{b}-bin-win-{'vulkan' if backend == 'vulkan' else 'cpu'}-{arch}.zip"]
    if backend == "cuda":
        return [
            f"llama-{b}-bin-ubuntu-cuda-12.8-x64.tar.gz",
            f"cudart-llama-{b}-bin-ubuntu-cuda-12.8-x64.tar.gz",
        ]
    return [f"llama-{b}-bin-ubuntu-{'vulkan-' if backend == 'vulkan' else ''}{arch}.tar.gz"]


def engine_dir(backend: Backend) -> Path:
    return (
        user_data_path("llmphotosearch", appauthor=False)
        / "engine"
        / f"{LLAMA_CPP_BUILD}-{backend}"
    )


def ensure_server_binary(
    hw: Hardware, backend: Backend, on_progress: ProgressCallback | None = None
) -> Path:
    """Path to `llama-server`, downloading the engine first if needed.

    The `LLMPHOTOSEARCH_LLAMA_SERVER` environment variable overrides it with a
    custom build.
    """
    if custom := os.environ.get("LLMPHOTOSEARCH_LLAMA_SERVER"):
        return Path(custom)
    target = engine_dir(backend)
    name = "llama-server.exe" if hw.os == "windows" else "llama-server"
    if not (target / ".complete").exists():
        shutil.rmtree(target, ignore_errors=True)
        target.mkdir(parents=True)
        main, *runtimes = release_assets(hw, backend)
        _fetch(main, target, on_progress)
        binary = _find(target, name)
        # Runtime libraries (CUDA) must sit next to the executable.
        for asset in runtimes:
            _fetch(asset, binary.parent, on_progress)
        if hw.os != "windows":
            binary.chmod(0o755)
        (target / ".complete").touch()
    return _find(target, name)


def _find(root: Path, name: str) -> Path:
    found = next(root.rglob(name), None)
    if found is None:
        raise EngineError(f"{name} not found in {root}")
    return found


def _fetch(asset: str, dest: Path, on_progress: ProgressCallback | None) -> None:
    archive = dest / asset
    _download(f"{RELEASE_URL}/{asset}", archive, on_progress)
    if archive.name.endswith(".zip"):
        with zipfile.ZipFile(archive) as z:
            z.extractall(dest)
    else:
        with tarfile.open(archive) as t:
            t.extractall(dest, filter="data")
    archive.unlink()


def _download(url: str, dest: Path, on_progress: ProgressCallback | None) -> None:
    try:
        with urllib.request.urlopen(url, timeout=60) as resp, dest.open("wb") as f:
            total = int(resp.headers.get("Content-Length", 0))
            done = 0
            while chunk := resp.read(1 << 20):
                f.write(chunk)
                done += len(chunk)
                if on_progress:
                    on_progress(done, total)
    except urllib.error.URLError as e:
        raise EngineError(f"Could not download {url}: {e}") from e


class LlamaServer:
    """A running `llama-server` process serving one model on localhost."""

    def __init__(
        self,
        binary: Path,
        model: Path,
        mmproj: Path | None = None,
        *,
        ctx_size: int = 4096,
        gpu: bool = True,
        log_file: Path | None = None,
    ):
        self.port = _free_port()
        self.url = f"http://127.0.0.1:{self.port}"
        self.log_file = log_file
        self._log = (log_file or Path(os.devnull)).open("wb")
        # One slot: images are scored one at a time, as parallel slots bring no
        # speedup (image encoding dominates).
        args = [
            str(binary),
            "--model", str(model),
            "--host", "127.0.0.1",
            "--port", str(self.port),
            "--ctx-size", str(ctx_size),
            "--parallel", "1",
            "--no-webui",
            # Answers must come right away, without a thinking phase.
            "--reasoning", "off",
        ]  # fmt: skip
        if mmproj:
            args += ["--mmproj", str(mmproj)]
        if not gpu:
            args += ["--n-gpu-layers", "0", "--no-mmproj-offload"]
        self._proc = subprocess.Popen(
            args,
            cwd=binary.parent,
            stdout=self._log,
            stderr=subprocess.STDOUT,
            stdin=subprocess.DEVNULL,
        )

    def wait_ready(self, timeout: float = 300) -> None:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if self._proc.poll() is not None:
                raise EngineError(
                    f"llama-server exited with code {self._proc.returncode}"
                    + (f", see {self.log_file}" if self.log_file else "")
                )
            try:
                with urllib.request.urlopen(f"{self.url}/health", timeout=2) as r:
                    if r.status == 200:
                        return
            except (urllib.error.URLError, ConnectionError, TimeoutError):
                pass
            time.sleep(0.25)
        raise EngineError("llama-server did not become ready in time")

    def chat(self, messages: list[dict], **params) -> dict:
        """POST an OpenAI-style chat completion and return the JSON response."""
        body = json.dumps({"messages": messages, **params}).encode()
        req = urllib.request.Request(
            f"{self.url}/v1/chat/completions",
            data=body,
            headers={"Content-Type": "application/json"},
        )
        try:
            with urllib.request.urlopen(req, timeout=600) as resp:
                return json.load(resp)
        except urllib.error.HTTPError as e:
            raise EngineError(
                f"llama-server error {e.code}: {e.read().decode(errors='replace')}"
            ) from e

    def stop(self) -> None:
        if self._proc.poll() is None:
            self._proc.terminate()
            try:
                self._proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self._proc.kill()
                self._proc.wait()
        self._log.close()

    def __enter__(self) -> "LlamaServer":
        return self

    def __exit__(self, *exc) -> None:
        self.stop()


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]
