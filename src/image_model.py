import torch
import torch.nn as nn
from torchvision.models import EfficientNet_B0_Weights, efficientnet_b0


class AdImageModel(nn.Module):
    def __init__(
        self,
        num_content_types: int = 3,
        num_moods: int = 4,
        freeze_encoder: bool = True,
        pretrained: bool = True,
    ):
        super().__init__()

        weights = EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None
        self.backbone = efficientnet_b0(weights=weights)

        self.encoder = self.backbone.features
        in_features = 1280  # EfficientNet-B0's final feature dimension after avgpool

        self.head_content = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(in_features, num_content_types),
        )
        self.head_mood = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(in_features, num_moods),
        )

        if freeze_encoder:
            self.freeze_encoder()

    def encode_image(self, x):
        features = self.encoder(x)
        features = self.backbone.avgpool(features)
        return torch.flatten(features, 1)

    def forward(self, x):
        features = self.encode_image(x)

        content_logits = self.head_content(features)
        mood_logits = self.head_mood(features)
        return content_logits, mood_logits

    def freeze_encoder(self):
        for param in self.encoder.parameters():
            param.requires_grad = False

    def unfreeze_last_blocks(self, num_blocks: int = 3):
        # Unfreeze the last `num_blocks` blocks of the encoder
        blocks = list(self.encoder.children())
        for block in blocks[-num_blocks:]:
            for param in block.parameters():
                param.requires_grad = True
