from collections.abc import Sequence
from pathlib import Path
from typing import Any

import torch
from PIL import Image

from model_core.loader import LoadedClassifier
from model_core.preprocess import load_rgb_image, preprocess_crops


def _normalize_images(images: Sequence[Image.Image | str | Path]) -> list[Image.Image]:
    normalized_images = []
    for image in images:
        if isinstance(image, Image.Image):
            normalized_images.append(image.convert("RGB"))
        else:
            normalized_images.append(load_rgb_image(image))
    return normalized_images


@torch.inference_mode()
def predict_crops(
    *,
    loaded: LoadedClassifier,
    images: Sequence[Image.Image | str | Path],
    top_k: int = 5,
    device: torch.device | str | None = None,
) -> list[list[dict[str, Any]]]:
    if top_k < 1:
        raise ValueError("top_k must be positive")
    if not images:
        return []

    model_device = device
    if model_device is None:
        model_device = next(loaded.model.parameters()).device
    image_size = int(loaded.model_config["image_size"])
    normalized_images = _normalize_images(images)
    pixel_values = preprocess_crops(
        images=normalized_images,
        processor=loaded.processor,
        image_size=image_size,
    ).to(model_device)

    logits = loaded.model(pixel_values)
    probabilities = torch.softmax(logits, dim=1)
    max_top_k = min(int(top_k), probabilities.shape[1])
    top_probs, top_ids = probabilities.topk(k=max_top_k, dim=1)

    batch_predictions = []
    for sample_probs, sample_ids in zip(top_probs.cpu(), top_ids.cpu()):
        predictions = []
        for probability, label_id in zip(sample_probs.tolist(), sample_ids.tolist()):
            label_id = int(label_id)
            predictions.append(
                {
                    "label_id": label_id,
                    "label": loaded.id_to_label[label_id],
                    "probability": float(probability),
                }
            )
        batch_predictions.append(predictions)
    return batch_predictions


def predict_crop(
    *,
    loaded: LoadedClassifier,
    image: Image.Image | str | Path,
    top_k: int = 5,
    device: torch.device | str | None = None,
) -> list[dict[str, Any]]:
    return predict_crops(
        loaded=loaded,
        images=[image],
        top_k=top_k,
        device=device,
    )[0]
