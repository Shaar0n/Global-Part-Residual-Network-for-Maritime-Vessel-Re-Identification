import torch
import torch.nn as nn
from torchvision import models


class GlobalReIDResNet(nn.Module):
    def __init__(self, num_classes: int, feat_dim: int = 512):
        super().__init__()

        resnet = models.resnet50(
            weights=models.ResNet50_Weights.IMAGENET1K_V1
        )

        # Use all layers up to layer4 (remove avgpool and fc)
        self.backbone = nn.Sequential(*list(resnet.children())[:-2])
        self.gap = nn.AdaptiveAvgPool2d(1)

        in_features = resnet.fc.in_features  # 2048

        # Embedding head
        self.embed = nn.Linear(in_features, feat_dim)
        self.bn = nn.BatchNorm1d(feat_dim)

        # Classification head
        self.classifier = nn.Linear(feat_dim, num_classes)

    def forward(self, x):
        x = self.backbone(x)          # [B, 2048, H/32, W/32]
        x = self.gap(x)               # [B, 2048, 1, 1]
        x = x.view(x.size(0), -1)     # [B, 2048]

        feat = self.embed(x)          # [B, feat_dim]
        feat = self.bn(feat)
        feat = nn.functional.normalize(feat, p=2, dim=1)  # L2 norm

        logits = self.classifier(feat)  # [B, num_classes]
        return feat, logits
