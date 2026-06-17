from collections.abc import Sequence
from pathlib import Path
from typing import Any

import torch
from PIL import Image


def load_rgb_image(path: str | Path) -> Image.Image:
    with Image.open(path) as image:
        image = image.convert("RGB")
        image.load()
        return image.copy()


def preprocess_crops(
    *,
    images: Sequence[Image.Image],
    processor: Any,
    image_size: int,
) -> torch.Tensor:
    processor_kwargs = {
        "images": list(images),
        "return_tensors": "pt",
        "size": {
            "height": int(image_size),
            "width": int(image_size),
        },
    }
    if getattr(processor, "crop_size", None) is not None:
        processor_kwargs["crop_size"] = {
            "height": int(image_size),
            "width": int(image_size),
        }
    encoded = processor(**processor_kwargs)
    return encoded["pixel_values"]


def preprocess_crop(
    *,
    image: Image.Image,
    processor: Any,
    image_size: int,
) -> torch.Tensor:
    return preprocess_crops(
        images=[image],
        processor=processor,
        image_size=image_size,
    )
