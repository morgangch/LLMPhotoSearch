import pytest

from llmphotosearch.backends.hardware import Hardware
from llmphotosearch.backends.llama_server import LLAMA_CPP_BUILD, default_backend, release_assets
from llmphotosearch.backends.vlm import default_model

WIN_NVIDIA = Hardware("windows", "x64", 32, 6)
WIN_IGPU = Hardware("windows", "x64", 16, None)
MAC = Hardware("macos", "arm64", 16, None)
LINUX_SMALL = Hardware("linux", "x64", 8, None)


@pytest.mark.parametrize(
    ("hw", "backend"),
    [(WIN_NVIDIA, "cuda"), (WIN_IGPU, "vulkan"), (MAC, "metal"), (LINUX_SMALL, "vulkan")],
)
def test_default_backend(hw, backend):
    assert default_backend(hw) == backend


def test_release_assets_include_cuda_runtime():
    assets = release_assets(WIN_NVIDIA, "cuda")
    assert assets[0] == f"llama-{LLAMA_CPP_BUILD}-bin-win-cuda-12.4-x64.zip"
    assert assets[1].startswith("cudart-")


@pytest.mark.parametrize(
    ("hw", "backend", "asset"),
    [
        (WIN_IGPU, "vulkan", "win-vulkan-x64.zip"),
        (WIN_IGPU, "cpu", "win-cpu-x64.zip"),
        (MAC, "metal", "macos-arm64.tar.gz"),
        (LINUX_SMALL, "vulkan", "ubuntu-vulkan-x64.tar.gz"),
        (LINUX_SMALL, "cpu", "ubuntu-x64.tar.gz"),
    ],
)
def test_release_assets_single_archive(hw, backend, asset):
    assert release_assets(hw, backend) == [f"llama-{LLAMA_CPP_BUILD}-bin-{asset}"]


def test_default_model():
    assert default_model(WIN_NVIDIA) == "qwen3.5-4b"
    assert default_model(WIN_IGPU) == "qwen3.5-4b"
    assert default_model(LINUX_SMALL) == "qwen3.5-2b"
