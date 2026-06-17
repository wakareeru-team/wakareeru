import argparse
import json
import time
from pathlib import Path
from typing import Any

import torch
from safetensors.torch import save_file
from transformers import AutoImageProcessor, AutoModel

from model_core.model import BackboneLinearClassifier
from pipeline import utils


ARCHITECTURE = "backbone_linear_classifier"
ARCHITECTURE_VERSION = 1
BACKBONE_DIR_NAME = "backbone"
PROCESSOR_DIR_NAME = "processor"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export a linear-head trainer checkpoint as an inference model bundle.",
    )
    parser.add_argument(
        "--config",
        default=None,
        type=Path,
        help="Path to pipeline_config.yaml. Defaults to config/pipeline_config.yaml.",
    )
    return parser.parse_args()


def load_checkpoint(checkpoint_path: Path) -> dict[str, Any]:
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    required_keys = {"model_state_dict", "config", "labels"}
    missing_keys = required_keys - set(checkpoint)
    if missing_keys:
        raise ValueError(f"Checkpoint is missing required keys: {sorted(missing_keys)}")
    return checkpoint


def build_model_config(
    *,
    checkpoint: dict[str, Any],
    artifact_version: str,
    classifier_weight: torch.Tensor,
) -> dict[str, Any]:
    trainer_cfg = checkpoint["config"]
    labels = checkpoint["labels"]
    num_classes = len(labels)
    expected_num_classes = int(classifier_weight.shape[0])
    if expected_num_classes != num_classes:
        raise ValueError(
            "Classifier output dimension does not match labels: "
            f"classifier={expected_num_classes}, labels={num_classes}"
        )

    return {
        "artifact_version": artifact_version,
        "architecture": ARCHITECTURE,
        "architecture_version": ARCHITECTURE_VERSION,
        "backbone_model_name": trainer_cfg["backbone_model_name"],
        "backbone": {
            "source_model_name": trainer_cfg["backbone_model_name"],
            "path": BACKBONE_DIR_NAME,
        },
        "feature_pooling": BackboneLinearClassifier.feature_pooling,
        "image_size": int(trainer_cfg["image_size"]),
        "num_classes": num_classes,
        "classifier": {
            "type": "linear",
            "feature_dim": int(classifier_weight.shape[1]),
        },
    }


def extract_classifier_state(state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    required_keys = {"classifier.weight", "classifier.bias"}
    missing_keys = required_keys - set(state_dict)
    if missing_keys:
        raise ValueError(f"Checkpoint is missing classifier weights: {sorted(missing_keys)}")
    return {
        "weight": state_dict["classifier.weight"].detach().cpu(),
        "bias": state_dict["classifier.bias"].detach().cpu(),
    }


def extract_backbone_state(state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    backbone_prefix = "backbone."
    backbone_state = {
        key.removeprefix(backbone_prefix): value.detach().cpu()
        for key, value in state_dict.items()
        if key.startswith(backbone_prefix)
    }
    if not backbone_state:
        raise ValueError("Checkpoint is missing backbone weights")
    return backbone_state


def write_json(path: Path, payload: Any) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def sync_processor_image_size(processor: Any, image_size: int) -> None:
    size = {
        "height": int(image_size),
        "width": int(image_size),
    }
    processor.size = size
    if getattr(processor, "crop_size", None) is not None:
        processor.crop_size = size


def export_inference_model(
    *,
    checkpoint_path: Path,
    output_dir: Path,
    artifact_version: str,
    force: bool,
) -> None:
    if output_dir.exists() and not force:
        raise FileExistsError(f"Output directory already exists: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=force)

    checkpoint = load_checkpoint(checkpoint_path)
    state_dict = checkpoint["model_state_dict"]
    classifier_state = extract_classifier_state(state_dict)
    model_config = build_model_config(
        checkpoint=checkpoint,
        artifact_version=artifact_version,
        classifier_weight=classifier_state["weight"],
    )

    backbone = AutoModel.from_pretrained(model_config["backbone_model_name"])
    incompatible_keys = backbone.load_state_dict(extract_backbone_state(state_dict), strict=False)
    if incompatible_keys.missing_keys:
        raise ValueError(
            "Missing keys while loading checkpoint backbone state: "
            f"{incompatible_keys.missing_keys}"
        )
    if incompatible_keys.unexpected_keys:
        raise ValueError(
            "Unexpected keys while loading checkpoint backbone state: "
            f"{incompatible_keys.unexpected_keys}"
        )
    backbone.save_pretrained(output_dir / BACKBONE_DIR_NAME)

    processor = AutoImageProcessor.from_pretrained(model_config["backbone_model_name"])
    sync_processor_image_size(processor, int(model_config["image_size"]))
    processor.save_pretrained(output_dir / PROCESSOR_DIR_NAME)

    save_file(classifier_state, output_dir / "classifier.safetensors")
    write_json(output_dir / "model_config.json", model_config)
    write_json(output_dir / "labels.json", checkpoint["labels"])
    write_json(
        output_dir / "manifest.json",
        {
            "artifact_version": artifact_version,
            "source_checkpoint": str(checkpoint_path),
            "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z", time.localtime()),
            "backbone_source_model_name": model_config["backbone"]["source_model_name"],
            "backbone_dir": model_config["backbone"]["path"],
            "processor_dir": PROCESSOR_DIR_NAME,
            "metrics": checkpoint.get("metrics", {}),
            "epoch": checkpoint.get("epoch"),
        },
    )


def export_inference_model_from_config(config: dict[str, Any]) -> None:
    export_cfg = config["trainer"]["export"]
    checkpoint_path = utils.join_data_root(export_cfg["checkpoint_path"], config=config)
    output_dir = utils.join_data_root(export_cfg["output_dir"], config=config)
    export_inference_model(
        checkpoint_path=checkpoint_path,
        output_dir=output_dir,
        artifact_version=str(export_cfg["artifact_version"]),
        force=bool(export_cfg["force"]),
    )


def main() -> None:
    args = parse_args()
    config = utils.load_pipeline_config(args.config)
    export_inference_model_from_config(config)


if __name__ == "__main__":
    main()
