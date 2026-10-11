"""Image discovery and loading."""

import os
from collections.abc import Iterable, Iterator
from pathlib import Path

from PIL import Image, ImageOps
from pillow_heif import register_heif_opener

register_heif_opener()

IMAGE_EXTENSIONS = frozenset(
    {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".gif", ".tif", ".tiff", ".heic", ".heif"}
)


def iter_images(folders: Iterable[Path]) -> Iterator[Path]:
    """Yield every image under `folders`, recursively, each absolute path only once.

    Hidden directories (starting with a dot) are skipped.
    """
    seen: set[Path] = set()
    for folder in folders:
        for root, dirs, files in os.walk(Path(folder).resolve()):
            dirs[:] = sorted(d for d in dirs if not d.startswith("."))
            for name in sorted(files):
                path = Path(root, name)
                if path.suffix.lower() in IMAGE_EXTENSIONS and path not in seen:
                    seen.add(path)
                    yield path


def load_image(path: Path, min_size: int = 0) -> Image.Image:
    """Open an image as upright RGB.

    `min_size` lets JPEGs be decoded at reduced scale, as long as both sides stay
    at least that large, which is much faster for big photos.
    """
    with Image.open(path) as img:
        if min_size:
            img.draft("RGB", (min_size, min_size))
        img = ImageOps.exif_transpose(img)
        return img.convert("RGB")
