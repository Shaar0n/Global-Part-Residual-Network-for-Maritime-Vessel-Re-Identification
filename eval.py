import argparse
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
import pandas as pd
import numpy as np

from datasets import Vessel2258EvalDataset
from models.baseline_resnet import GlobalReIDResNet
from models.global_part_resnet import GlobalPartReIDResNet, GlobalPartReIDResNetV2

ROOT = r"C:\Users\shall\VesselReID\REID_2258"
CSV = r"C:\Users\shall\VesselReID\REID_2258\vessel2258_meta_split.csv"
NUM_TRAIN_CLASSES = 1481

def imagenet_normalize(img_size=256):
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])


IMAGENET_NORMALIZE = imagenet_normalize(256)

MODEL_REGISTRY = {
    "baseline": {
        "build": lambda: GlobalReIDResNet(num_classes=NUM_TRAIN_CLASSES, feat_dim=512),
        # train_baseline.py trains with Resize+ToTensor only (no ImageNet normalize).
        "transform": transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
        ]),
        "default_ckpt": r"C:\Users\shall\VesselReID\checkpoints\global_baseline_epoch10.pth",
        # forward() returns (feat, logits)
        "extract_feat": lambda outputs: outputs[0],
    },
    "global_part": {
        "build": lambda: GlobalPartReIDResNet(num_classes=NUM_TRAIN_CLASSES, feat_dim=512, num_parts=3),
        # train_global_part.py trains with ImageNet normalization - eval must match.
        "transform": IMAGENET_NORMALIZE,
        "default_ckpt": r"C:\Users\shall\VesselReID\checkpoints\global_part_epoch10.pth",
        # eval-mode forward() returns (final_feat, None)
        "extract_feat": lambda outputs: outputs[0],
    },
    "global_part_v2": {
        "build": lambda: GlobalPartReIDResNetV2(num_classes=NUM_TRAIN_CLASSES, feat_dim=512, num_parts=3),
        "transform": IMAGENET_NORMALIZE,
        "default_ckpt": r"C:\Users\shall\VesselReID\checkpoints\global_part_v2_best.pth",
        # forward() always returns (g_logits, part_logits, g_embed, final_feat)
        "extract_feat": lambda outputs: outputs[3],
    },
    "global_part_v2_384": {
        "build": lambda: GlobalPartReIDResNetV2(num_classes=NUM_TRAIN_CLASSES, feat_dim=512, num_parts=3),
        "transform": imagenet_normalize(384),
        # global_part_384_best.pth is the full 30-epoch run (train_v2.py --ckpt-tag 384);
        # global_part_v2_384_best.pth was an earlier 5-epoch partial run, kept on disk but obsolete.
        "default_ckpt": r"C:\Users\shall\VesselReID\checkpoints\global_part_384_best.pth",
        "extract_feat": lambda outputs: outputs[3],
    },
}


def build_query_gallery(csv_path, split="test"):
    df = pd.read_csv(csv_path)
    df_test = df[df["split"] == split].reset_index(drop=True)

    query_df = df_test.groupby("vessel_id").head(1)
    gallery_df = df_test.drop(query_df.index).reset_index(drop=True)
    query_df = query_df.reset_index(drop=True)

    print(f"{split} IDs: {df_test['vessel_id'].nunique()}")
    print(f"Query images: {len(query_df)}, Gallery images: {len(gallery_df)}")

    return query_df, gallery_df


def extract_features(model, loader, device, extract_feat, flip_tta=False):
    model.eval()
    feats, pids, cams = [], [], []
    with torch.no_grad():
        for imgs, vid, cam in loader:
            imgs = imgs.to(device)
            feat = extract_feat(model(imgs))
            if flip_tta:
                flipped = torch.flip(imgs, dims=[3])
                feat_flip = extract_feat(model(flipped))
                feat = torch.nn.functional.normalize(feat + feat_flip, p=2, dim=1)
            feats.append(feat.cpu())
            pids.append(vid)
            cams.append(cam)
    feats = torch.cat(feats, dim=0)
    pids = torch.cat(pids, dim=0)
    cams = torch.cat(cams, dim=0)
    return feats, pids, cams


def k_reciprocal_rerank(qf, gf, k1=20, k2=6, lambda_value=0.3):
    """k-reciprocal re-ranking (Zhong et al. 2017), simplified for a single feature set."""
    all_num = qf.size(0) + gf.size(0)
    feat = torch.cat([qf, gf], dim=0)
    dist = torch.cdist(feat, feat, p=2).numpy()
    dist = np.power(dist, 2).astype(np.float32)
    dist = dist / (np.max(dist, axis=0) + 1e-12)
    original_dist = dist.copy()
    V = np.zeros_like(dist).astype(np.float32)
    initial_rank = np.argsort(dist).astype(np.int32)

    for i in range(all_num):
        forward_k_neigh_index = initial_rank[i, :k1 + 1]
        backward_k_neigh_index = initial_rank[forward_k_neigh_index, :k1 + 1]
        fi = np.where(backward_k_neigh_index == i)[0]
        k_reciprocal_index = forward_k_neigh_index[fi]
        k_reciprocal_expansion_index = k_reciprocal_index
        for j in range(len(k_reciprocal_index)):
            candidate = k_reciprocal_index[j]
            candidate_forward_k_neigh_index = initial_rank[candidate, :int(np.around(k1 / 2)) + 1]
            candidate_backward_k_neigh_index = initial_rank[candidate_forward_k_neigh_index, :int(np.around(k1 / 2)) + 1]
            fi_candidate = np.where(candidate_backward_k_neigh_index == candidate)[0]
            candidate_k_reciprocal_index = candidate_forward_k_neigh_index[fi_candidate]
            if len(np.intersect1d(candidate_k_reciprocal_index, k_reciprocal_index)) > 2 / 3 * len(candidate_k_reciprocal_index):
                k_reciprocal_expansion_index = np.append(k_reciprocal_expansion_index, candidate_k_reciprocal_index)

        k_reciprocal_expansion_index = np.unique(k_reciprocal_expansion_index)
        weight = np.exp(-dist[i, k_reciprocal_expansion_index])
        V[i, k_reciprocal_expansion_index] = weight / np.sum(weight)

    if k2 != 1:
        V_qe = np.zeros_like(V, dtype=np.float32)
        for i in range(all_num):
            V_qe[i, :] = np.mean(V[initial_rank[i, :k2], :], axis=0)
        V = V_qe

    invIndex = [np.where(V[:, i] != 0)[0] for i in range(all_num)]

    jaccard_dist = np.zeros_like(original_dist, dtype=np.float32)
    for i in range(all_num):
        temp_min = np.zeros(shape=(1, all_num), dtype=np.float32)
        indNonZero = np.where(V[i, :] != 0)[0]
        for j in indNonZero:
            temp_min[0, invIndex[j]] += np.minimum(V[i, j], V[invIndex[j], j])
        jaccard_dist[i] = 1 - temp_min / (2 - temp_min)

    final_dist = jaccard_dist * (1 - lambda_value) + original_dist * lambda_value
    final_dist = final_dist[:qf.size(0), qf.size(0):]
    return final_dist


def evaluate(distmat, q_pids, q_cams, g_pids, g_cams, max_rank=50):
    """
    CMC + mAP over the whole gallery, matching by vessel_id only.

    Note: this dataset's `camera_id`/`seq_id` fields are constant (every filename
    follows the literal pattern "<id>_c1s1_<frame>_01.jpg") - there is no real
    multi-camera signal to exclude "same camera" gallery hits on, so the classic
    Market-1501 same-ID-different-camera protocol is not applicable here. Junk is
    only pid == -1 (unused placeholder, kept for compatibility with distractor sets).
    """
    num_q, num_g = distmat.shape
    indices = distmat.argsort(axis=1)

    all_cmc = []
    all_ap = []
    num_valid_q = 0

    for q_idx in range(num_q):
        q_pid = q_pids[q_idx].item()
        order = indices[q_idx]

        good_mask = (g_pids.numpy() == q_pid)
        good_index = np.where(good_mask)[0]

        junk_mask = (g_pids.numpy() == -1)
        junk_index = np.where(junk_mask)[0]

        if len(good_index) == 0:
            continue

        mask = np.ones(len(order), dtype=bool)
        mask[junk_index] = False
        order = order[mask]

        matches = np.in1d(order, good_index)
        ngood = len(good_index)

        cmc = matches.cumsum()
        cmc[cmc > 1] = 1
        all_cmc.append(cmc)

        hit_inds = np.where(matches)[0]
        precisions = []
        for pos in hit_inds:
            prec = matches[:pos + 1].sum() / (pos + 1)
            precisions.append(prec)
        ap = sum(precisions) / ngood
        all_ap.append(ap)

        num_valid_q += 1

    if num_valid_q == 0:
        raise RuntimeError("No valid query with good matches in gallery.")

    padded_cmc = []
    for cmc in all_cmc:
        if len(cmc) >= max_rank:
            padded_cmc.append(cmc[:max_rank])
        else:
            tmp = np.zeros(max_rank)
            tmp[:len(cmc)] = cmc
            padded_cmc.append(tmp)
    all_cmc = np.stack(padded_cmc)
    cmc_curve = all_cmc.mean(axis=0)
    mAP = float(sum(all_ap) / num_valid_q)

    return cmc_curve, mAP


def main():
    parser = argparse.ArgumentParser(description="VesselReID evaluation (unified)")
    parser.add_argument("--model", choices=list(MODEL_REGISTRY.keys()), required=True)
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--csv", type=str, default=CSV)
    parser.add_argument("--root", type=str, default=ROOT)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--rerank", action="store_true", help="Apply k-reciprocal re-ranking")
    parser.add_argument("--flip-tta", action="store_true", help="Average features with horizontal flip")
    args = parser.parse_args()

    cfg = MODEL_REGISTRY[args.model]
    ckpt_path = args.checkpoint or cfg["default_ckpt"]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)
    print(f"Model: {args.model}  Checkpoint: {ckpt_path}")

    query_df, gallery_df = build_query_gallery(args.csv)

    tf = cfg["transform"]
    q_dataset = Vessel2258EvalDataset(args.root, query_df, transform=tf)
    g_dataset = Vessel2258EvalDataset(args.root, gallery_df, transform=tf)

    q_loader = DataLoader(q_dataset, batch_size=args.batch_size, shuffle=False,
                          num_workers=args.num_workers, pin_memory=True)
    g_loader = DataLoader(g_dataset, batch_size=args.batch_size, shuffle=False,
                          num_workers=args.num_workers, pin_memory=True)

    model = cfg["build"]()
    state_dict = torch.load(ckpt_path, map_location=device)
    if isinstance(state_dict, dict) and "model_state" in state_dict:
        state_dict = state_dict["model_state"]
    model.load_state_dict(state_dict, strict=False)
    model.to(device)

    print("Extracting query features...")
    qf, q_pids, q_cams = extract_features(model, q_loader, device, cfg["extract_feat"], flip_tta=args.flip_tta)
    print("Extracting gallery features...")
    gf, g_pids, g_cams = extract_features(model, g_loader, device, cfg["extract_feat"], flip_tta=args.flip_tta)

    if args.rerank:
        print("Applying k-reciprocal re-ranking...")
        distmat = k_reciprocal_rerank(qf, gf)
    else:
        distmat = torch.cdist(qf, gf, p=2).numpy()

    print("Computing CMC and mAP...")
    cmc, mAP = evaluate(distmat, q_pids, q_cams, g_pids, g_cams)

    print("\n" + "=" * 50)
    print(f"{args.model} results (match by vessel_id"
          f"{', rerank' if args.rerank else ''}{', flip-tta' if args.flip_tta else ''}):")
    print(f"mAP: {mAP:.4f}")
    for r in [1, 5, 10, 20, 50]:
        if r <= len(cmc):
            print(f"Rank-{r}: {cmc[r-1]:.4f}")
    print("=" * 50)


if __name__ == "__main__":
    main()
