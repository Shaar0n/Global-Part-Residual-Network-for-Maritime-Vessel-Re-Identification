import torch
import torch.nn as nn
import torch.nn.functional as F


class TripletLoss(nn.Module):
    """Batch-hard triplet loss with margin."""
    def __init__(self, margin: float = 0.3):
        super().__init__()
        self.margin = margin
        self.ranking_loss = nn.MarginRankingLoss(margin=margin)

    def forward(self, embeddings, labels):
        # embeddings: [B, D], labels: [B]
        n = embeddings.size(0)
        # pairwise distance matrix
        dist = torch.cdist(embeddings, embeddings, p=2)  # [B, B]

        # masks
        labels = labels.view(-1, 1)
        mask_pos = labels.eq(labels.t())   # positives
        mask_neg = ~mask_pos               # negatives

        dist_ap = []
        dist_an = []
        for i in range(n):
            pos_i = dist[i][mask_pos[i]]
            neg_i = dist[i][mask_neg[i]]
            if len(pos_i) > 1:
                dist_ap.append(pos_i.max().unsqueeze(0))
            else:
                dist_ap.append(torch.zeros(1, device=embeddings.device))
            dist_an.append(neg_i.min().unsqueeze(0))

        dist_ap = torch.cat(dist_ap)
        dist_an = torch.cat(dist_an)

        y = torch.ones_like(dist_an)
        loss = self.ranking_loss(dist_an, dist_ap, y)
        return loss


class ReIDLoss(nn.Module):
    """ID classification (CE with label smoothing) + Triplet."""
    def __init__(self, num_classes: int, margin: float = 0.3,
                 ce_weight: float = 1.0, triplet_weight: float = 1.0):
        super().__init__()
        self.ce_weight = ce_weight
        self.triplet_weight = triplet_weight
        self.ce_loss = nn.CrossEntropyLoss(label_smoothing=0.1)
        self.triplet_loss = TripletLoss(margin=margin)

    def forward(self, logits, embeddings, labels):
        ce = self.ce_loss(logits, labels)
        trip = self.triplet_loss(embeddings, labels)
        return self.ce_weight * ce + self.triplet_weight * trip, {
            "ce": ce.item(),
            "triplet": trip.item()
        }
