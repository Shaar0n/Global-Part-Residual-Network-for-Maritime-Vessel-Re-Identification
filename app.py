import os

import streamlit as st
import torch
from PIL import Image
from torch.utils.data import DataLoader

from datasets import Vessel2258EvalDataset
from eval import MODEL_REGISTRY, ROOT, CSV, build_query_gallery, extract_features

st.set_page_config(page_title="VesselReID Demo", layout="wide")


@st.cache_resource(show_spinner="Loading model and extracting gallery features...")
def load_model_and_gallery(model_name: str):
    cfg = MODEL_REGISTRY[model_name]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = cfg["build"]()
    state_dict = torch.load(cfg["default_ckpt"], map_location=device)
    if isinstance(state_dict, dict) and "model_state" in state_dict:
        state_dict = state_dict["model_state"]
    model.load_state_dict(state_dict, strict=False)
    model.to(device).eval()

    query_df, gallery_df = build_query_gallery(CSV, split="test")
    g_loader = DataLoader(
        Vessel2258EvalDataset(ROOT, gallery_df, transform=cfg["transform"]),
        batch_size=64, shuffle=False, num_workers=2, pin_memory=True,
    )
    gf, g_pids, _ = extract_features(model, g_loader, device, cfg["extract_feat"])
    return model, device, query_df, gallery_df, gf, g_pids


def main():
    st.title("VesselReID — Retrieval Demo")
    st.caption(
        "Pick a query vessel from the held-out test split and retrieve its closest "
        "matches from a gallery of ~24k unseen images."
    )

    available_models = [m for m in ["global_part_v2_384", "global_part_v2", "global_part", "baseline"] if m in MODEL_REGISTRY]
    model_name = st.sidebar.selectbox("Model", available_models)
    top_k = st.sidebar.slider("Top-K", 1, 10, 5)

    cfg = MODEL_REGISTRY[model_name]
    if not os.path.exists(cfg["default_ckpt"]):
        st.error(f"Checkpoint not found: {cfg['default_ckpt']}")
        return

    model, device, query_df, gallery_df, gf, g_pids = load_model_and_gallery(model_name)
    st.sidebar.write(f"Gallery size: {len(gallery_df)} images")
    st.sidebar.write(f"Query pool: {len(query_df)} vessels")

    query_idx = st.selectbox(
        "Query vessel",
        options=list(range(len(query_df))),
        format_func=lambda i: f"ID {query_df.iloc[i]['vessel_id']} ({query_df.iloc[i]['vessel_type']})",
    )

    q_row = query_df.iloc[query_idx]
    q_img = Image.open(os.path.join(ROOT, q_row["rel_path"])).convert("RGB")

    col_query, col_results = st.columns([1, 4])
    with col_query:
        st.image(q_img, caption=f"Query — ID {q_row['vessel_id']}", width='stretch')

    tf = cfg["transform"]
    x = tf(q_img).unsqueeze(0).to(device)
    with torch.no_grad():
        feat = cfg["extract_feat"](model(x)).cpu()

    dist = torch.cdist(feat, gf, p=2).squeeze(0).numpy()
    order = dist.argsort()[:top_k]

    with col_results:
        cols = st.columns(top_k)
        for rank, (g_idx, c) in enumerate(zip(order, cols), start=1):
            g_row = gallery_df.iloc[g_idx]
            g_img = Image.open(os.path.join(ROOT, g_row["rel_path"])).convert("RGB")
            correct = g_row["vessel_id"] == q_row["vessel_id"]
            label = f"#{rank} {'correct' if correct else 'wrong'}  (d={dist[g_idx]:.2f})"
            with c:
                st.image(g_img, caption=label, width='stretch')
                if correct:
                    st.success("Match")
                else:
                    st.error("No match")


if __name__ == "__main__":
    main()
