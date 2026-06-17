import torch
from torch import nn
from transformers import AutoModel


ARCHITECTURE = "backbone_linear_classifier"
ARCHITECTURE_VERSION = 1


class BackboneLinearClassifier(nn.Module):
    """Backbone plus a linear classification head."""

    feature_pooling = "cls_patch_mean"

    def __init__(
        self,
        *,
        backbone_model_name: str | None = None,
        backbone: nn.Module | None = None,
        num_classes: int,
        freeze_backbone: bool,
        local_files_only: bool = False,
    ) -> None:
        super().__init__()
        if backbone is None:
            if backbone_model_name is None:
                raise ValueError("Either backbone_model_name or backbone must be provided")
            backbone = AutoModel.from_pretrained(
                backbone_model_name,
                local_files_only=local_files_only,
            )
        self.backbone = backbone
        self.feature_dim = int(self.backbone.config.hidden_size) * 2
        self.classifier = nn.Linear(self.feature_dim, num_classes)
        self.backbone_frozen = False
        if freeze_backbone:
            self.freeze_backbone()

    def freeze_backbone(self) -> None:
        for parameter in self.backbone.parameters():
            parameter.requires_grad = False
        self.backbone_frozen = True
        self.backbone.eval()

    def unfreeze_backbone(self) -> None:
        for parameter in self.backbone.parameters():
            parameter.requires_grad = True
        self.backbone_frozen = False

    def train_linear_head_only(self) -> None:
        self.freeze_backbone()
        for parameter in self.classifier.parameters():
            parameter.requires_grad = True

    def train(self, mode: bool = True) -> "BackboneLinearClassifier":
        super().train(mode)
        if self.backbone_frozen:
            self.backbone.eval()
        return self

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        features = self.extract_features(pixel_values)
        return self.classifier(features)

    def extract_features(self, pixel_values: torch.Tensor) -> torch.Tensor:
        if self.backbone_frozen:
            with torch.no_grad():
                outputs = self.backbone(pixel_values=pixel_values)
        else:
            outputs = self.backbone(pixel_values=pixel_values)

        last_hidden_state = outputs.last_hidden_state
        cls_features = getattr(outputs, "pooler_output", None)
        if cls_features is None:
            cls_features = last_hidden_state[:, 0]

        num_register_tokens = int(getattr(self.backbone.config, "num_register_tokens", 0))
        patch_token_start = 1 + num_register_tokens
        if last_hidden_state.shape[1] <= patch_token_start:
            raise ValueError(
                "Backbone output does not contain patch tokens after CLS and register tokens: "
                f"sequence_length={last_hidden_state.shape[1]}, "
                f"num_register_tokens={num_register_tokens}"
            )
        patch_mean_features = last_hidden_state[:, patch_token_start:].mean(dim=1)
        return torch.cat((cls_features, patch_mean_features), dim=-1)
