from __future__ import annotations

from pathlib import Path
from PIL import Image, ImageOps


def load_image(path: str | Path) -> Image.Image:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(path)
    with Image.open(path) as im:
        return im.convert("RGB").copy()


def resize_image(image: Image.Image, max_size: int = 768) -> Image.Image:
    image = image.convert("RGB")
    if max(image.size) <= max_size:
        return image.copy()
    scale = max_size / float(max(image.size))
    size = (max(1, round(image.width * scale)), max(1, round(image.height * scale)))
    return image.resize(size, Image.Resampling.LANCZOS)


def prepare_image(image: Image.Image, max_size: int = 768) -> Image.Image:
    return ImageOps.exif_transpose(image).convert("RGB") if max(image.size) <= max_size else resize_image(ImageOps.exif_transpose(image), max_size)
