"""Vision-language models runnable by the bundled llama-server."""

from dataclasses import dataclass
from pathlib import Path

from huggingface_hub import hf_hub_download

from llmphotosearch.backends.hardware import Hardware


@dataclass(frozen=True)
class VLMSpec:
    repo: str
    model_file: str
    mmproj_file: str

    def download(self) -> tuple[Path, Path]:
        return hf_hub_download(self.repo, self.model_file), hf_hub_download(
            self.repo, self.mmproj_file
        )


def _qwen35(size: str) -> VLMSpec:
    return VLMSpec(f"unsloth/Qwen3.5-{size}-GGUF", f"Qwen3.5-{size}-Q4_K_M.gguf", "mmproj-F16.gguf")


# All Apache-2.0. Q4 weights: ~1.9 GB (2B), ~3.3 GB (4B), ~6.3 GB (9B), ~4.9 GB (E4B),
# projector included.
MODELS = {
    "qwen3.5-2b": _qwen35("2B"),
    "qwen3.5-4b": _qwen35("4B"),
    "qwen3.5-9b": _qwen35("9B"),
    "gemma4-e4b": VLMSpec(
        "ggml-org/gemma-4-E4B-it-GGUF",
        "gemma-4-E4B-it-Q4_0.gguf",
        "mmproj-gemma-4-E4B-it-Q8_0.gguf",
    ),
}


def default_model(hw: Hardware) -> str:
    # On a labeled sample, the 4B judged matches best (fewer errors than the 9B and
    # Gemma 4 E4B), so it is the default whenever memory allows.
    if hw.ram_gb < 12:
        return "qwen3.5-2b"
    return "qwen3.5-4b"
