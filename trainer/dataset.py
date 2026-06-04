from pathlib import Path
from typing import Any

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset


class CropDataset(Dataset):
    """Dataset for crop images exported by stage_14_store_crops."""

    def __init__(
        self,
        *,
        metadata: pd.DataFrame,
        dataset_root: str | Path,
        image_path_column: str,
        label_id_column: str,
    ) -> None:
        self.metadata = metadata.reset_index(drop=True).copy()
        self.dataset_root = Path(dataset_root)
        self.image_path_column = image_path_column
        self.label_id_column = label_id_column

        missing_columns = {
            image_path_column,
            label_id_column,
        } - set(self.metadata.columns)
        if missing_columns:
            raise ValueError(f"metadata缺少必要列: {sorted(missing_columns)}")
        if self.metadata[label_id_column].isna().any():
            raise ValueError(f"metadata列 {label_id_column!r} 存在空标签")
        self.image_paths = [
            str(path).replace("\\", "/")
            for path in self.metadata[image_path_column].tolist()
        ]
        self.full_paths = [
            self.dataset_root / image_path
            for image_path in self.image_paths
        ]
        self.labels = [
            int(label)
            for label in self.metadata[label_id_column].tolist()
        ]

    def __len__(self) -> int:
        return len(self.metadata)

    def __getitem__(self, index: int) -> dict[str, Any]:
        image_path = self.image_paths[index]
        full_path = self.full_paths[index]
        with Image.open(full_path) as image:
            image = image.convert("RGB")
            image.load()
            return {
                "image": image.copy(),
                "label": self.labels[index],
                "sample_index": index,
                "image_path": image_path,
            }


class CropCollator:
    """Batch PIL crop images with a model-specific image processor."""

    def __init__(self, processor: Any, image_size: int) -> None:
        self.processor = processor
        self.image_size = int(image_size)

    def __call__(self, batch: list[dict[str, Any]]) -> dict[str, Any]:
        images = [item["image"] for item in batch]
        processor_kwargs = {
            "images": images,
            "return_tensors": "pt",
            "size": {
                "height": self.image_size,
                "width": self.image_size,
            },
        }
        if getattr(self.processor, "crop_size", None) is not None:
            processor_kwargs["crop_size"] = {
                "height": self.image_size,
                "width": self.image_size,
            }
        encoded = self.processor(**processor_kwargs)
        return {
            "pixel_values": encoded["pixel_values"],
            "labels": torch.tensor([item["label"] for item in batch], dtype=torch.long),
            "sample_index": torch.tensor(
                [item["sample_index"] for item in batch],
                dtype=torch.long,
            ),
            "image_path": [item["image_path"] for item in batch],
        }
