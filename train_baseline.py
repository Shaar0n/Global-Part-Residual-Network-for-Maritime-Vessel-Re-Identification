import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm


from datasets import Vessel2258Dataset
from models.baseline_resnet import GlobalReIDResNet


# Paths
ROOT = r"C:\Users\shall\VesselReID\REID_2258"
CSV  = r"C:\Users\shall\VesselReID\REID_2258\vessel2258_meta_split.csv"
CKPT_DIR = r"C:\Users\shall\VesselReID\checkpoints"



def build_loaders(batch_size=64, num_workers=2):
    train_tf = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(0.2, 0.2, 0.2, 0.1),
        transforms.ToTensor(),
    ])

    val_tf = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
    ])

    train_set = Vessel2258Dataset(ROOT, CSV, split="train", transform=train_tf)
    val_set   = Vessel2258Dataset(ROOT, CSV, split="val",   transform=val_tf)

    num_classes = len(train_set.id2label)
    print("Train images:", len(train_set), "Train IDs:", num_classes)

    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    return train_loader, val_loader, num_classes


def batch_hard_triplet_loss(embeds, labels, margin=0.3):
    """
    Standard batch-hard triplet loss:
    For each anchor, use hardest positive (max dist) and hardest negative (min dist)
    within the batch. [web:417][web:429]
    """
    # pairwise distances [B, B]
    dist = torch.cdist(embeds, embeds, p=2)

    labels = labels.unsqueeze(1)  # [B, 1]
    mask_pos = (labels == labels.t())        # positives (including self)
    mask_neg = (labels != labels.t())        # negatives

    # remove self from positives
    mask_pos.fill_diagonal_(False)

    # hardest positive: largest distance among positives
    # if a sample has no positive in batch, its row will be all 0 -> contributes 0 loss
    dist_pos = dist * mask_pos.float()
    hardest_pos, _ = dist_pos.max(dim=1)  # [B]

    # hardest negative: smallest distance among negatives
    dist_neg = dist + 1e5 * (~mask_neg)   # large for non-negatives
    hardest_neg, _ = dist_neg.min(dim=1)  # [B]

    # triplet loss: max(d_pos - d_neg + margin, 0)
    loss = torch.relu(hardest_pos - hardest_neg + margin)
    return loss.mean()


def main():
    os.makedirs(CKPT_DIR, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    train_loader, val_loader, num_classes = build_loaders(
        batch_size=64,
        num_workers=2,  # faster than 0, still stable on Windows
    )

    model = GlobalReIDResNet(num_classes=num_classes, feat_dim=512)
    model.to(device)

    ce_criterion = nn.CrossEntropyLoss()
    lambda_triplet = 1.0

    optimizer = optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
    epochs = 20

    best_val_acc = 0.0
    patience_counter = 0
    patience = 2  # stop after 2 epochs without val improvement

    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        running_ce = 0.0
        running_tri = 0.0
        correct = 0
        total = 0

        train_pbar = tqdm(train_loader, desc=f"Train E{epoch:02d}", leave=False)
        for batch_idx, (imgs, labels, _) in enumerate(train_pbar):
            imgs = imgs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            feats, logits = model(imgs)

            ce_loss = ce_criterion(logits, labels)

            # Triplet loss requires at least 2 different IDs in the batch
            if labels.unique().numel() > 1:
                tri_loss = batch_hard_triplet_loss(feats, labels, margin=0.3)
            else:
                tri_loss = torch.zeros(1, device=device)

            loss = ce_loss + lambda_triplet * tri_loss
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * imgs.size(0)
            running_ce += ce_loss.item() * imgs.size(0)
            running_tri += tri_loss.item() * imgs.size(0)

            _, preds = logits.max(1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

            train_pbar.set_postfix({
                'Loss': f"{loss.item():.4f}",
                'CE': f"{ce_loss.item():.3f}",
                'Tri': f"{tri_loss.item():.3f}",
                'Acc': f"{correct/total:.1%}"
            })

        train_loss = running_loss / total
        train_ce = running_ce / total
        train_tri = running_tri / total
        train_acc = correct / total

        model.eval()
        val_correct, val_total = 0, 0
        val_pbar = tqdm(val_loader, desc=f"Val E{epoch:02d}", leave=False)
        with torch.no_grad():
            for imgs, labels, _ in val_pbar:
                imgs = imgs.to(device)
                labels = labels.to(device)
                _, logits = model(imgs)
                _, preds = logits.max(1)
                val_correct += (preds == labels).sum().item()
                val_total += labels.size(0)
                val_pbar.set_postfix({'Val Acc': f"{val_correct/val_total:.1%}"})

        val_acc = val_correct / max(1, val_total)

        print(
            f"Epoch {epoch:02d}: "
            f"train_loss={train_loss:.4f} "
            f"(ce={train_ce:.4f}, tri={train_tri:.4f}) "
            f"train_acc={train_acc:.3f} "
            f"val_acc={val_acc:.3f}"
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
            best_path = os.path.join(CKPT_DIR, "baseline_best.pth")
            torch.save(model.state_dict(), best_path)
            print(f"  -> New best model saved! Val Acc = {val_acc:.3f}")
        else:
            patience_counter += 1
            print(f"  -> Patience {patience_counter}/{patience}")

        if patience_counter >= patience:
            print(f"Early stopping at epoch {epoch}!")
            break

        ckpt_path = os.path.join(CKPT_DIR, f"baseline_epoch_{epoch:02d}.pth")
        torch.save({
            "epoch": epoch,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "train_loss": train_loss,
            "val_acc": val_acc,
        }, ckpt_path)


if __name__ == "__main__":
    main()
