import argparse
import json
import math
import os

import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm

from datasets import Vessel2258Dataset, Vessel2258EvalDataset
from models.global_part_resnet import GlobalPartReIDResNetV2
from losses.reid_loss import ReIDLoss
from samplers import PKSampler
from eval import build_query_gallery, extract_features, evaluate as compute_cmc_map

ROOT = r"C:\Users\shall\VesselReID\REID_2258"
CSV = r"C:\Users\shall\VesselReID\REID_2258\vessel2258_meta_split.csv"
CKPT_DIR = r"C:\Users\shall\VesselReID\checkpoints"

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def build_lr_lambda(warmup_epochs: int, max_epochs: int, warmup_factor: float = 0.1):
    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            alpha = epoch / max(1, warmup_epochs)
            return warmup_factor + (1 - warmup_factor) * alpha
        progress = (epoch - warmup_epochs) / max(1, max_epochs - warmup_epochs)
        return 0.5 * (1 + math.cos(math.pi * min(progress, 1.0)))
    return lr_lambda


def build_loaders(args):
    img_size = (args.img_size, args.img_size)
    transform_train = transforms.Compose([
        transforms.Resize(img_size),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(0.15, 0.15, 0.15, 0.05),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        transforms.RandomErasing(p=0.5, scale=(0.02, 0.2)),
    ])
    transform_eval = transforms.Compose([
        transforms.Resize(img_size),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])

    train_dataset = Vessel2258Dataset(ROOT, CSV, split="train", transform=transform_train)
    num_classes = len(train_dataset.id2label)

    labels = train_dataset.df["label"].tolist()
    sampler = PKSampler(labels, num_instances=args.num_instances, batch_size=args.batch_size)
    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, sampler=sampler,
        num_workers=args.num_workers, pin_memory=True, drop_last=True,
    )

    # Validation is retrieval, not classification: val identities are disjoint from
    # train identities (open-set re-ID), so there is no shared label space to score
    # classifier accuracy against. Build a query/gallery split within "val" instead,
    # same shape as the real test-time protocol in eval.py.
    val_query_df, val_gallery_df = build_query_gallery(args.csv, split="val")
    val_q_loader = DataLoader(
        Vessel2258EvalDataset(args.root, val_query_df, transform=transform_eval),
        batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True,
    )
    val_g_loader = DataLoader(
        Vessel2258EvalDataset(args.root, val_gallery_df, transform=transform_eval),
        batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True,
    )

    return train_loader, val_q_loader, val_g_loader, num_classes, len(train_dataset)


def _checkpoint_is_loadable(path):
    """A hard power-loss can truncate a checkpoint mid-write and leave a file that
    exists, has a plausible size, but isn't a valid torch archive (seen in practice:
    torch.load raising 'filename storages not found' on a file cut off by a crash)."""
    try:
        torch.load(path, map_location="cpu", weights_only=False)
        return True
    except Exception as e:
        print(f"[auto-resume] {path} exists but failed to load ({type(e).__name__}: {e}) - skipping it.")
        return False


def resolve_auto_resume(ckpt_tag):
    """Pick the most-advanced *loadable* checkpoint available for this tag, trying
    in order: an in-progress mid-epoch snapshot (if ahead of the last fully-completed
    epoch), the last completed epoch, then the best-so-far checkpoint - falling back
    down this chain if a candidate turns out to be corrupt. Lets a crashed run be
    relaunched with the exact same command line (plus --auto-resume) with no manual
    epoch bookkeeping and no manual intervention if the crash corrupted a checkpoint."""
    last_epoch_path = os.path.join(CKPT_DIR, f"global_part_{ckpt_tag}_last_epoch.txt")
    mid_meta_path = os.path.join(CKPT_DIR, f"global_part_{ckpt_tag}_midepoch.json")
    best_epoch_path = os.path.join(CKPT_DIR, f"global_part_{ckpt_tag}_best_epoch.txt")
    last_path = os.path.join(CKPT_DIR, f"global_part_{ckpt_tag}_last.pth")
    mid_path = os.path.join(CKPT_DIR, f"global_part_{ckpt_tag}_midepoch.pth")
    best_path = os.path.join(CKPT_DIR, f"global_part_{ckpt_tag}_best.pth")

    last_epoch = 0
    if os.path.exists(last_epoch_path):
        with open(last_epoch_path) as f:
            last_epoch = int(f.read().strip())

    candidates = []

    if os.path.exists(mid_meta_path) and os.path.exists(mid_path):
        with open(mid_meta_path) as f:
            meta = json.load(f)
        if meta["epoch"] >= last_epoch:
            candidates.append((
                mid_path, meta["epoch"],
                f"in-progress checkpoint for epoch {meta['epoch'] + 1} "
                f"(batch {meta['batch']}/{meta['total_batches']})",
            ))

    if last_epoch > 0:
        candidates.append((last_path, last_epoch, f"last completed epoch {last_epoch}"))

    if os.path.exists(best_epoch_path) and os.path.exists(best_path):
        with open(best_epoch_path) as f:
            best_epoch = int(f.read().strip())
        candidates.append((best_path, best_epoch, f"best-so-far checkpoint (epoch {best_epoch})"))

    for path, epoch, description in candidates:
        if _checkpoint_is_loadable(path):
            print(f"[auto-resume] Resuming from {description}.")
            return path, epoch

    print("[auto-resume] No loadable checkpoint found for this tag - starting fresh.")
    return None, 0


def validate(model, val_q_loader, val_g_loader, device):
    extract_feat = lambda outputs: outputs[3]
    qf, q_pids, q_cams = extract_features(model, val_q_loader, device, extract_feat)
    gf, g_pids, g_cams = extract_features(model, val_g_loader, device, extract_feat)
    distmat = torch.cdist(qf, gf, p=2).numpy()
    cmc, mAP = compute_cmc_map(distmat, q_pids, q_cams, g_pids, g_cams)
    return mAP, cmc[0]


def train():
    parser = argparse.ArgumentParser(description="Strong-baseline training for GlobalPartReIDResNetV2")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--warmup-epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-instances", type=int, default=4, help="K images per identity per batch")
    parser.add_argument("--lr", type=float, default=3.5e-4)
    parser.add_argument("--weight-decay", type=float, default=5e-4)
    parser.add_argument("--feat-dim", type=int, default=512)
    parser.add_argument("--num-parts", type=int, default=3)
    parser.add_argument("--triplet-margin", type=float, default=0.3)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--csv", type=str, default=CSV)
    parser.add_argument("--root", type=str, default=ROOT)
    parser.add_argument("--img-size", type=int, default=256)
    parser.add_argument("--ckpt-tag", type=str, default="v2",
                        help="checkpoint filename tag, e.g. global_part_{tag}_best.pth")
    parser.add_argument("--resume", type=str, default=None,
                        help="checkpoint path to load model weights from before training")
    parser.add_argument("--start-epoch", type=int, default=0,
                        help="epochs already completed (0-based) - offsets the LR schedule and the loop")
    parser.add_argument("--auto-resume", action="store_true",
                        help="ignore --resume/--start-epoch and pick the most-advanced checkpoint "
                             "(mid-epoch or last-completed) found under --ckpt-tag automatically")
    parser.add_argument("--ckpt-every-batches", type=int, default=250,
                        help="save a mid-epoch checkpoint every N batches (0 disables); caps work lost "
                             "to a crash/power-loss to well under one epoch")
    args = parser.parse_args()

    if args.auto_resume:
        args.resume, args.start_epoch = resolve_auto_resume(args.ckpt_tag)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    train_loader, val_q_loader, val_g_loader, num_classes, n_train = build_loaders(args)
    print(f"Training on {n_train} images, {num_classes} classes ({len(train_loader)} batches/epoch).")

    model = GlobalPartReIDResNetV2(num_classes=num_classes, feat_dim=args.feat_dim, num_parts=args.num_parts)
    model.to(device)

    if args.resume:
        model.load_state_dict(torch.load(args.resume, map_location=device))
        print(f"Resumed weights from {args.resume}")

    criterion = ReIDLoss(num_classes=num_classes, margin=args.triplet_margin)
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    if args.start_epoch > 0:
        for group in optimizer.param_groups:
            group.setdefault("initial_lr", args.lr)
    scheduler = optim.lr_scheduler.LambdaLR(
        optimizer, build_lr_lambda(args.warmup_epochs, args.epochs),
        last_epoch=args.start_epoch - 1 if args.start_epoch > 0 else -1,
    )
    scaler = torch.amp.GradScaler("cuda", enabled=(device.type == "cuda"))

    os.makedirs(CKPT_DIR, exist_ok=True)
    best_val_map = 0.0
    patience_counter = 0

    if args.resume:
        model.eval()
        with torch.no_grad():
            best_val_map, resumed_rank1 = validate(model, val_q_loader, val_g_loader, device)
        print(f"Resumed model val mAP: {best_val_map:.4f}, val Rank-1: {resumed_rank1:.4f}")

    for epoch in range(args.start_epoch, args.epochs):
        model.train()
        total_loss, total_ce, total_tri = 0.0, 0.0, 0.0
        correct, total = 0, 0

        pbar = tqdm(train_loader, desc=f"Train E{epoch+1:02d}", leave=False)
        for batch_idx, (imgs, labels, _) in enumerate(pbar):
            imgs, labels = imgs.to(device), labels.to(device)
            optimizer.zero_grad()

            with torch.amp.autocast("cuda", enabled=(device.type == "cuda")):
                g_logits, part_logits, g_embed, _ = model(imgs)
                loss, parts = criterion(g_logits, g_embed, labels)
                part_ce = sum(criterion.ce_loss(p_logit, labels) for p_logit in part_logits)
                loss = loss + part_ce

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            total_loss += loss.item()
            total_ce += parts["ce"] + part_ce.item()
            total_tri += parts["triplet"]
            _, preds = torch.max(g_logits, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

            pbar.set_postfix({
                "Loss": f"{loss.item():.4f}",
                "Acc": f"{correct/total:.1%}",
                "LR": f"{scheduler.get_last_lr()[0]:.2e}",
            })

            if args.ckpt_every_batches > 0 and (batch_idx + 1) % args.ckpt_every_batches == 0:
                mid_path = os.path.join(CKPT_DIR, f"global_part_{args.ckpt_tag}_midepoch.pth")
                torch.save(model.state_dict(), mid_path)
                with open(os.path.join(CKPT_DIR, f"global_part_{args.ckpt_tag}_midepoch.json"), "w") as f:
                    json.dump({"epoch": epoch, "batch": batch_idx + 1, "total_batches": len(train_loader)}, f)

        scheduler.step()
        train_loss = total_loss / len(train_loader)
        train_acc = correct / total

        model.eval()
        with torch.no_grad():
            val_map, val_rank1 = validate(model, val_q_loader, val_g_loader, device)

        print(
            f"Epoch {epoch+1:02d}: train_loss={train_loss:.4f} "
            f"(ce={total_ce/len(train_loader):.4f}, tri={total_tri/len(train_loader):.4f}) "
            f"train_acc={train_acc:.3f} val_mAP={val_map:.4f} val_rank1={val_rank1:.4f}"
        )

        last_path = os.path.join(CKPT_DIR, f"global_part_{args.ckpt_tag}_last.pth")
        torch.save(model.state_dict(), last_path)
        with open(os.path.join(CKPT_DIR, f"global_part_{args.ckpt_tag}_last_epoch.txt"), "w") as f:
            f.write(str(epoch + 1))

        # epoch fully committed to last.pth - the mid-epoch snapshot (if any) is now stale
        for stale in (
            os.path.join(CKPT_DIR, f"global_part_{args.ckpt_tag}_midepoch.pth"),
            os.path.join(CKPT_DIR, f"global_part_{args.ckpt_tag}_midepoch.json"),
        ):
            if os.path.exists(stale):
                os.remove(stale)

        if val_map > best_val_map:
            best_val_map = val_map
            patience_counter = 0
            best_path = os.path.join(CKPT_DIR, f"global_part_{args.ckpt_tag}_best.pth")
            torch.save(model.state_dict(), best_path)
            with open(os.path.join(CKPT_DIR, f"global_part_{args.ckpt_tag}_best_epoch.txt"), "w") as f:
                f.write(str(epoch + 1))
            print(f"  -> New best model saved! Val mAP = {val_map:.4f}")
        else:
            patience_counter += 1
            print(f"  -> Patience {patience_counter}/{args.patience}")

        if patience_counter >= args.patience:
            print(f"Early stopping at epoch {epoch+1}!")
            break

    torch.save(model.state_dict(), os.path.join(CKPT_DIR, f"global_part_{args.ckpt_tag}_final.pth"))
    print(f"\nTraining finished! Best validation mAP: {best_val_map:.4f}")


if __name__ == "__main__":
    train()
