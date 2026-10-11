"""Embedding of image folders, with caching."""

from collections.abc import Callable, Sequence
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol

import numpy as np
from PIL import Image

from llmphotosearch.core.images import load_image
from llmphotosearch.index.store import EmbeddingStore


class ImageEmbedder(Protocol):
    name: str
    image_size: int

    def preprocess(self, image: Image.Image) -> np.ndarray: ...
    def embed_images(self, pixels: np.ndarray) -> np.ndarray: ...


@dataclass
class IndexResult:
    paths: list[Path]
    vectors: np.ndarray
    failed: list[tuple[Path, str]] = field(default_factory=list)


def build_index(
    images: Sequence[Path],
    embedder: ImageEmbedder,
    store: EmbeddingStore,
    batch_size: int = 16,
    on_progress: Callable[[int], None] | None = None,
) -> IndexResult:
    """Return embeddings for `images`, computing only those missing from `store`.

    `on_progress` is called with the number of images handled since the last call
    (cached ones first, then each batch).
    """
    cached = store.get_fresh(embedder.name, images)
    todo = [p for p in images if p not in cached]
    if on_progress and cached:
        on_progress(len(cached))
    failed: list[tuple[Path, str]] = []

    def prepare(path: Path) -> np.ndarray | None:
        try:
            return embedder.preprocess(load_image(path, embedder.image_size))
        except Exception as e:  # corrupt or unsupported files must not stop indexing
            failed.append((path, type(e).__name__))
            return None

    batches = [todo[i : i + batch_size] for i in range(0, len(todo), batch_size)]
    with ThreadPoolExecutor() as pool:
        # Decode the next batch while the current one runs through the model.
        pending: list[Future] = [pool.submit(prepare, p) for p in batches[0]] if batches else []
        for i, batch in enumerate(batches):
            arrays = [f.result() for f in pending]
            if i + 1 < len(batches):
                pending = [pool.submit(prepare, p) for p in batches[i + 1]]
            ok = [(p, a) for p, a in zip(batch, arrays, strict=True) if a is not None]
            if ok:
                vectors = embedder.embed_images(np.stack([a for _, a in ok]))
                store.put(embedder.name, zip((p for p, _ in ok), vectors, strict=True))
                cached.update(zip((p for p, _ in ok), vectors, strict=True))
            if on_progress:
                on_progress(len(batch))

    paths = [p for p in images if p in cached]
    vectors = np.stack([cached[p] for p in paths]) if paths else np.empty((0, 0), np.float32)
    return IndexResult(paths, vectors, failed)
