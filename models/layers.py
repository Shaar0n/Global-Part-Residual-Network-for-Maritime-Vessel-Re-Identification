import torch
import torch.nn as nn
import torch.nn.functional as F


class GeM(nn.Module):
    """Generalized-mean pooling: learnable interpolation between avg (p=1) and max (p->inf) pooling."""

    def __init__(self, p: float = 3.0, eps: float = 1e-6):
        super().__init__()
        self.p = nn.Parameter(torch.ones(1) * p)
        self.eps = eps

    def forward(self, x):
        x = x.clamp(min=self.eps).pow(self.p)
        x = F.avg_pool2d(x, (x.size(-2), x.size(-1)))
        return x.pow(1.0 / self.p)


class BNNeck(nn.Module):
    """BNNeck (Luo et al. 2019): decouples the triplet feature (pre-BN) from the
    classification feature (post-BN). The BN bias is frozen at 0 since it only
    needs to rescale, not shift, the feature for the classifier."""

    def __init__(self, feat_dim: int):
        super().__init__()
        self.bn = nn.BatchNorm1d(feat_dim)
        self.bn.bias.requires_grad_(False)

    def forward(self, x):
        return self.bn(x)
