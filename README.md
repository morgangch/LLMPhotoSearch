# LLMPhotoSearch

Find the photos matching a natural-language description ("un chat roux sur un canapé",
"sunset over the sea with a boat") across one or more folders, **fully offline**, on a
regular PC.

> Status: early development — not usable yet.

## How it works

1. **Index** — every image found recursively is embedded with SigLIP 2 (ONNX Runtime).
   Embeddings are cached, so only new or modified images are processed on later runs.
2. **Pre-filter** — the description is embedded too; the closest candidates are kept.
3. **Re-rank** — a small vision-language model (3B–8B, run by a bundled `llama-server`)
   answers *"does this image match the description?"* for each candidate. The match
   percentage is the model's probability of answering *yes*.

The inference engine and models are downloaded automatically on first run — nothing to
install or configure besides this tool. GPU acceleration is used when available
(CUDA, Vulkan, Metal), otherwise CPU.

## Usage (planned)

```sh
uv tool install llmphotosearch

photosearch search "a red-haired cat on a sofa" ~/Pictures /mnt/backup
photosearch ui
```

- CLI: prints up to 3 matching paths; with more matches, writes a Markdown report with
  match percentage and full path.
- UI: browse all matches in a media viewer, export as Markdown or JSON.

## Development

```sh
uv sync
uv run pytest
uv run ruff check
```

## License

[Apache-2.0](LICENSE)
