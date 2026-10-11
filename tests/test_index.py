import os

import numpy as np
from conftest import make_image

from llmphotosearch.index.indexer import build_index
from llmphotosearch.index.store import EmbeddingStore
from llmphotosearch.ranking.search import prefilter


def test_store_roundtrip_and_invalidation(tmp_path):
    img = make_image(tmp_path / "a.jpg")
    with EmbeddingStore(tmp_path / "db.sqlite") as store:
        store.put("m", [(img, np.array([1, 2, 3], dtype=np.float32))])
        assert list(store.get_fresh("m", [img])[img]) == [1, 2, 3]
        assert store.get_fresh("other-model", [img]) == {}

        stat = img.stat()
        os.utime(img, ns=(stat.st_atime_ns, stat.st_mtime_ns + 1_000_000_000))
        assert store.get_fresh("m", [img]) == {}


def test_build_index_uses_cache_and_skips_broken(tmp_path, embedder):
    red = make_image(tmp_path / "red.jpg", (255, 0, 0))
    blue = make_image(tmp_path / "blue.png", (0, 0, 255))
    broken = tmp_path / "broken.jpg"
    broken.write_text("not an image")

    with EmbeddingStore(tmp_path / "db.sqlite") as store:
        first = build_index([red, broken, blue], embedder, store, batch_size=1)
        assert first.paths == [red, blue]
        assert [p for p, _ in first.failed] == [broken]
        assert embedder.embedded == 2

        second = build_index([red, blue], embedder, store)
        assert embedder.embedded == 2
        np.testing.assert_allclose(second.vectors, first.vectors)


def test_prefilter_ranks_best_first(tmp_path, embedder):
    paths = [
        make_image(tmp_path / f"{name}.png", color)
        for name, color in [("red", (255, 0, 0)), ("green", (0, 255, 0)), ("blue", (0, 0, 255))]
    ]
    with EmbeddingStore(tmp_path / "db.sqlite") as store:
        index = build_index(paths, embedder, store)

    matches = prefilter("blue", index, embedder)
    assert matches[0].path.name == "blue.png"
    assert matches[0].score > matches[1].score
