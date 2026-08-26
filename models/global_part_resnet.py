import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

from models.layers import GeM, BNNeck

class GlobalPartReIDResNet(nn.Module):
    """Global average-pooled embedding plus 3 horizontal-stripe part branches (PCB-style),
    each with its own reduction + classifier. Inference feature is the L2-normalized
    concat of global + all part embeddings."""

    def __init__(self, num_classes=1481, feat_dim=512, num_parts=3):
        super(GlobalPartReIDResNet, self).__init__()

        resnet = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
        self.backbone = nn.Sequential(*list(resnet.children())[:-2])

        self.num_parts = num_parts
        self.feat_dim = feat_dim

        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.global_reduction = nn.Sequential(
            nn.Linear(2048, feat_dim),
            nn.BatchNorm1d(feat_dim),
            nn.ReLU()
        )
        self.global_classifier = nn.Linear(feat_dim, num_classes)

        # pools to (num_parts, 1) instead of (1, 1) - splits the feature map into
        # num_parts horizontal stripes, each with its own reduction + classifier below
        self.part_pool = nn.AdaptiveAvgPool2d((num_parts, 1))
        self.part_reductions = nn.ModuleList([
            nn.Sequential(
                nn.Linear(2048, feat_dim),
                nn.BatchNorm1d(feat_dim),
                nn.ReLU()
            ) for _ in range(num_parts)
        ])
        self.part_classifiers = nn.ModuleList([
            nn.Linear(feat_dim, num_classes) for _ in range(num_parts)
        ])

    def forward(self, x):
        features = self.backbone(x)  # [B, 2048, H, W]

        g_feat = self.global_pool(features).flatten(1)  # [B, 2048]
        g_embed = self.global_reduction(g_feat)          # [B, 512]

        p_feat = self.part_pool(features)  # [B, 2048, num_parts, 1]
        part_embeds = []
        part_logits = []
        for i in range(self.num_parts):
            part_i = p_feat[:, :, i, 0]
            embed_i = self.part_reductions[i](part_i)
            part_embeds.append(embed_i)
            if self.training:
                part_logits.append(self.part_classifiers[i](embed_i))

        if self.training:
            g_logits = self.global_classifier(g_embed)
            return g_logits, part_logits
        else:
            # inference feature: L2-normalized concat of global + all parts (2048-d)
            g_norm = torch.nn.functional.normalize(g_embed, p=2, dim=1)
            p_norms = [torch.nn.functional.normalize(p, p=2, dim=1) for p in part_embeds]
            final_feat = torch.cat([g_norm] + p_norms, dim=1)
            return final_feat, None


class GlobalPartReIDResNetV2(nn.Module):
    """Strong-baseline upgrade of GlobalPartReIDResNet: GeM pooling + BNNeck.

    BNNeck (Luo et al., "Bag of Tricks", 2019) decouples the two objectives that
    otherwise fight each other on one feature: triplet loss wants an
    unconstrained metric space, softmax CE wants a feature shaped for linear
    separability. Here the *pre-BN* embedding feeds triplet loss and the
    *post-BN* embedding feeds the classifier and is what's used for retrieval.

    Unlike the original, forward() does not branch on self.training - it always
    returns logits AND the final retrieval feature. (The original's eval-mode
    forward returned only the concatenated feature, which upstream training code
    mistakenly ran through argmax as if it were class logits - this makes that
    mistake structurally impossible.)
    """

    def __init__(self, num_classes: int, feat_dim: int = 512, num_parts: int = 3):
        super().__init__()

        resnet = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
        self.backbone = nn.Sequential(*list(resnet.children())[:-2])

        self.num_parts = num_parts
        self.feat_dim = feat_dim

        self.global_pool = GeM()
        self.global_reduction = nn.Sequential(nn.Linear(2048, feat_dim), nn.ReLU(inplace=True))
        self.global_bnneck = BNNeck(feat_dim)
        self.global_classifier = nn.Linear(feat_dim, num_classes, bias=False)

        self.part_pool = GeM()
        self.part_reductions = nn.ModuleList([
            nn.Sequential(nn.Linear(2048, feat_dim), nn.ReLU(inplace=True))
            for _ in range(num_parts)
        ])
        self.part_bnnecks = nn.ModuleList([BNNeck(feat_dim) for _ in range(num_parts)])
        self.part_classifiers = nn.ModuleList([
            nn.Linear(feat_dim, num_classes, bias=False) for _ in range(num_parts)
        ])

    def _pool_stripes(self, feature_map):
        stripes = torch.chunk(feature_map, self.num_parts, dim=2)
        return [self.part_pool(s).flatten(1) for s in stripes]

    def forward(self, x):
        features = self.backbone(x)  # [B, 2048, H, W]

        g_feat = self.global_pool(features).flatten(1)
        g_embed = self.global_reduction(g_feat)      # pre-BN, used for triplet loss
        g_bn = self.global_bnneck(g_embed)            # post-BN, used for CE + retrieval
        g_logits = self.global_classifier(g_bn)

        part_bns = []
        part_logits = []
        for i, stripe_feat in enumerate(self._pool_stripes(features)):
            p_embed = self.part_reductions[i](stripe_feat)
            p_bn = self.part_bnnecks[i](p_embed)
            part_bns.append(p_bn)
            part_logits.append(self.part_classifiers[i](p_bn))

        g_norm = F.normalize(g_bn, p=2, dim=1)
        p_norms = [F.normalize(p, p=2, dim=1) for p in part_bns]
        final_feat = torch.cat([g_norm] + p_norms, dim=1)

        return g_logits, part_logits, g_embed, final_feat

