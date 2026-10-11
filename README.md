# LLMPhotoSearch

Find the photos matching a natural-language description ("un chat roux sur un canapé",
"sunset over the sea with a boat") across one or more folders, **fully offline**, on a
regular PC.

> Status: early development. The CLI works; the UI is not there yet.

## How it works

1. **Index**: every image found recursively is embedded with
   [SigLIP 2](https://huggingface.co/google/siglip2-base-patch16-224) on ONNX Runtime.
   Embeddings are cached, so later runs only process new or modified images.
2. **Pre-filter**: the description is translated to English by the VLM, embedded, and
   the closest images are kept (30 by default).
3. **Re-rank**: a small vision-language model, run by a bundled `llama.cpp` server, is
   asked *"Is this a photo of: <description>?"* for each candidate. The **match
   percentage is the model's probability of answering yes**, read from its token
   probabilities, so it is not a number the model makes up.

The inference engine and models are downloaded automatically on first run. There is
nothing else to install or configure. GPU acceleration is used when available (CUDA on
NVIDIA, Vulkan on AMD/Intel GPUs and iGPUs, Metal on Apple Silicon), otherwise CPU.

## Installation

Requires [uv](https://docs.astral.sh/uv/).

```sh
uv tool install git+https://github.com/morgangch/LLMPhotoSearch
```

## Usage

```sh
photosearch search "deux chats qui dorment sur un canapé" ~/Pictures /mnt/backup
```

- Up to 3 matching images: their match percentage and path are printed.
- More than 3: a Markdown report (`photosearch-results.md`) is written with the match
  percentage and full path of each image. Use `--out results.json` for JSON.

| Option | Default | |
|---|---|---|
| `--threshold` | 50 | Minimum match percentage |
| `--top-k` | 30 | Candidates analyzed by the VLM (more: slower, better recall) |
| `--model` | from hardware | `qwen3.5-2b`, `qwen3.5-4b`, `qwen3.5-9b`, `gemma4-e4b` |
| `--backend` | from hardware | `cuda`, `vulkan`, `metal`, `cpu` |
| `--fast` | off | Embeddings only: instant, but no match percentage |
| `--out` | | Report path (`.md` or `.json`) |

### Models

All are Apache-2.0, 4-bit quantized. The default is `qwen3.5-4b`, or `qwen3.5-2b` below
12 GB of RAM.

| Model | Download | Time per candidate, RTX 2060 | Time per candidate, CPU only |
|---|---|---|---|
| qwen3.5-2b | 1.9 GB | 0.2 s | 2.2 s |
| qwen3.5-4b | 3.3 GB | 0.35 s | 3.5 s |
| qwen3.5-9b | 6.3 GB | 1.3 s (does not fit in 6 GB VRAM) | |
| gemma4-e4b | 4.9 GB | 0.25 s | |

Indexing runs at about 10 images/s on CPU, only once per image.

### Data locations

- Models: the Hugging Face cache (`~/.cache/huggingface`).
- Inference engine: the user data directory (`%LOCALAPPDATA%\llmphotosearch` on Windows,
  `~/.local/share/llmphotosearch` on Linux, `~/Library/Application Support/llmphotosearch`
  on macOS).
- Embedding index and `llama-server.log`: the user cache directory.

Set `LLMPHOTOSEARCH_LLAMA_SERVER` to the path of your own `llama-server` build to use it
instead of the bundled one.

## Development

```sh
uv sync
uv run pytest
uv run ruff check
```

## License

[Apache-2.0](LICENSE)
