import torch
from peft import LoraConfig, get_peft_model
from torch import nn
from transformers import AutoModel


class BackboneLinearClassifier(nn.Module):
    """Backbone plus a linear classification head."""

    feature_pooling = "cls_patch_mean"

    def __init__(
        self,
        *,
        backbone_model_name: str,
        num_classes: int,
        freeze_backbone: bool,
    ) -> None:
        super().__init__()
        self.backbone = AutoModel.from_pretrained(backbone_model_name)
        self.feature_dim = int(self.backbone.config.hidden_size) * 2
        self.classifier = nn.Linear(self.feature_dim, num_classes)
        self.backbone_frozen = False
        self.lora_enabled = False
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

    def enable_lora(
        self,
        *,
        r: int,
        alpha: int,
        dropout: float,
        bias: str,
    ) -> None:
        if self.lora_enabled:
            return
        target_modules = self.attention_linear_module_names()
        lora_config = LoraConfig(
            r=r,
            lora_alpha=alpha,
            target_modules=target_modules,
            lora_dropout=dropout,
            bias=bias,
        )
        self.backbone = get_peft_model(self.backbone, lora_config)
        self.lora_enabled = True
        self.train_lora_and_head()

    def train_lora_and_head(self) -> None:
        if not self.lora_enabled:
            raise RuntimeError("LoRA has not been enabled")
        for name, parameter in self.backbone.named_parameters():
            parameter.requires_grad = "lora_" in name
        for parameter in self.classifier.parameters():
            parameter.requires_grad = True
        self.backbone_frozen = False

    def attention_linear_module_names(self) -> list[str]:
        target_modules = [
            name
            for name, module in self.backbone.named_modules()
            if isinstance(module, nn.Linear) and ".attention." in f".{name}."
        ]
        if not target_modules:
            raise ValueError("No attention linear modules found for LoRA injection")
        return target_modules

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
