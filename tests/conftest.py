from pathlib import Path

import numpy as np
import pytest
from PIL import Image


def make_image(path: Path, color: tuple[int, int, int] = (255, 0, 0)) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (64, 48), color).save(path)
    return path


class FakeEmbedder:
    """Embeds an image as its mean RGB color; texts are color names."""

    name = "fake"
    image_size = 8
    COLORS = {"red": (1.0, 0.0, 0.0), "green": (0.0, 1.0, 0.0), "blue": (0.0, 0.0, 1.0)}

    def __init__(self):
        self.embedded = 0

    def preprocess(self, image):
        return np.asarray(image.resize((8, 8)), dtype=np.float32).transpose(2, 0, 1) / 255

    def embed_images(self, pixels):
        self.embedded += len(pixels)
        v = pixels.mean(axis=(2, 3))
        return v / np.linalg.norm(v, axis=1, keepdims=True)

    def embed_text(self, text):
        return np.array(self.COLORS[text], dtype=np.float32)

    def probability(self, similarity):
        return similarity


@pytest.fixture
def embedder():
    return FakeEmbedder()
