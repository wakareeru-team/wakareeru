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

    def __len__(self) -> int:
        return len(self.metadata)

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.metadata.iloc[index]
        image_path = str(row[self.image_path_column]).replace("\\", "/")
        full_path = self.dataset_root / image_path
        with Image.open(full_path) as image:
            image = image.convert("RGB")
            return {
                "image": image.copy(),
                "label": int(row[self.label_id_column]),
                "sample_index": index,
                "image_path": image_path,
            }


class CropCollator:
    """Batch PIL crop images with a model-specific image processor."""

    def __init__(self, processor: Any) -> None:
        self.processor = processor

    def __call__(self, batch: list[dict[str, Any]]) -> dict[str, Any]:
        images = [item["image"] for item in batch]
        encoded = self.processor(images=images, return_tensors="pt")
        return {
            "pixel_values": encoded["pixel_values"],
            "labels": torch.tensor([item["label"] for item in batch], dtype=torch.long),
            "sample_index": torch.tensor(
                [item["sample_index"] for item in batch],
                dtype=torch.long,
            ),
            "image_path": [item["image_path"] for item in batch],
        }
