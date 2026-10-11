from conftest import make_image
from PIL import Image

from llmphotosearch.core.images import iter_images, load_image


def test_iter_images_recursive_and_filtered(tmp_path):
    a = make_image(tmp_path / "a.jpg")
    b = make_image(tmp_path / "sub" / "deep" / "b.PNG")
    make_image(tmp_path / ".hidden" / "c.jpg")
    (tmp_path / "notes.txt").write_text("x")

    assert list(iter_images([tmp_path])) == [a, b]


def test_iter_images_deduplicates_overlapping_folders(tmp_path):
    a = make_image(tmp_path / "sub" / "a.jpg")

    assert list(iter_images([tmp_path, tmp_path / "sub"])) == [a]


def test_load_image_returns_rgb(tmp_path):
    path = tmp_path / "gray.png"
    Image.new("L", (10, 10)).save(path)

    assert load_image(path).mode == "RGB"
