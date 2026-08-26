import argparse
import os
import random

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from PIL import Image
from torch.utils.data import DataLoader

from datasets import Vessel2258EvalDataset
from eval import MODEL_REGISTRY, ROOT, CSV, build_query_gallery, extract_features, k_reciprocal_rerank

OUT_DIR = r"C:\Users\shall\VesselReID\figures"


def main():
    parser = argparse.ArgumentParser(description="Visualize top-K retrieval results")
    parser.add_argument("--model", choices=list(MODEL_REGISTRY.keys()), default="global_part_v2")
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--num-queries", type=int, default=6)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--rerank", action="store_true")
    parser.add_argument("--flip-tta", action="store_true")
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()

    cfg = MODEL_REGISTRY[args.model]
    ckpt_path = args.checkpoint or cfg["default_ckpt"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Model: {args.model}  Checkpoint: {ckpt_path}")

    query_df, gallery_df = build_query_gallery(CSV)
    tf = cfg["transform"]
    q_dataset = Vessel2258EvalDataset(ROOT, query_df, transform=tf)
    g_dataset = Vessel2258EvalDataset(ROOT, gallery_df, transform=tf)
    q_loader = DataLoader(q_dataset, batch_size=64, shuffle=False, num_workers=2, pin_memory=True)
    g_loader = DataLoader(g_dataset, batch_size=64, shuffle=False, num_workers=2, pin_memory=True)

    model = cfg["build"]()
    state_dict = torch.load(ckpt_path, map_location=device)
    if isinstance(state_dict, dict) and "model_state" in state_dict:
        state_dict = state_dict["model_state"]
    model.load_state_dict(state_dict, strict=False)
    model.to(device)

    print("Extracting features...")
    qf, q_pids, _ = extract_features(model, q_loader, device, cfg["extract_feat"], flip_tta=args.flip_tta)
    gf, g_pids, _ = extract_features(model, g_loader, device, cfg["extract_feat"], flip_tta=args.flip_tta)

    if args.rerank:
        print("Applying k-reciprocal re-ranking...")
        distmat = k_reciprocal_rerank(qf, gf)
    else:
        distmat = torch.cdist(qf, gf, p=2).numpy()

    random.seed(args.seed)
    query_indices = random.sample(range(len(query_df)), min(args.num_queries, len(query_df)))

    os.makedirs(OUT_DIR, exist_ok=True)
    fig, axes = plt.subplots(
        len(query_indices), args.top_k + 1,
        figsize=((args.top_k + 1) * 2.0, len(query_indices) * 2.2),
    )

    for row, q_idx in enumerate(query_indices):
        q_row = query_df.iloc[q_idx]
        q_img = Image.open(os.path.join(ROOT, q_row["rel_path"])).convert("RGB")
        ax = axes[row, 0]
        ax.imshow(q_img)
        ax.set_title(f"Query\nID {q_row['vessel_id']}", fontsize=8)
        ax.set_xticks([])
        ax.set_yticks([])

        order = distmat[q_idx].argsort()[: args.top_k]
        for col, g_idx in enumerate(order, start=1):
            g_row = gallery_df.iloc[g_idx]
            g_img = Image.open(os.path.join(ROOT, g_row["rel_path"])).convert("RGB")
            ax = axes[row, col]
            ax.imshow(g_img)
            correct = g_row["vessel_id"] == q_row["vessel_id"]
            color = "limegreen" if correct else "red"
            for spine in ax.spines.values():
                spine.set_visible(True)
                spine.set_color(color)
                spine.set_linewidth(3)
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_title(f"#{col} {'OK' if correct else 'X'}", fontsize=8, color=color)

    plt.tight_layout()
    suffix = ("_rerank" if args.rerank else "") + ("_flip" if args.flip_tta else "")
    out_path = os.path.join(OUT_DIR, f"retrieval_{args.model}{suffix}.png")
    plt.savefig(out_path, dpi=150)
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
