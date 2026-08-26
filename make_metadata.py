from pathlib import Path
import pandas as pd
import numpy as np

ROOT = Path(r"C:\Users\shall\VesselReID\REID_2258")

rows = []
for type_dir in ROOT.iterdir():
    if not type_dir.is_dir():
        continue
    vessel_type = type_dir.name
    for img_path in type_dir.glob("*.jpg"):
        name = img_path.stem  # e.g. 0104_c1s1_000061_01
        parts = name.split("_")
        if len(parts) < 3:
            continue
        vid_str = parts[0]
        cam_seq = parts[1]
        frame_str = parts[2]
        extra = parts[3] if len(parts) > 3 else None

        vid = int(vid_str)
        cam = int(cam_seq.split("s")[0][1:])
        seq = int(cam_seq.split("s")[1])

        rows.append({
            "rel_path": str(img_path.relative_to(ROOT)),
            "vessel_id": vid,
            "camera_id": cam,
            "seq_id": seq,
            "frame": int(frame_str),
            "extra": extra,
            "vessel_type": vessel_type,
        })

df = pd.DataFrame(rows)
meta_path = ROOT / "vessel2258_meta.csv"
df.to_csv(meta_path, index=False)
print("Images:", len(df), "unique IDs:", df["vessel_id"].nunique())

# create identity-wise splits
ids = df["vessel_id"].unique()
rng = np.random.default_rng(seed=42)
rng.shuffle(ids)

n = len(ids)
n_train = int(0.6 * n)
n_val   = int(0.2 * n)

train_ids = set(ids[:n_train])
val_ids   = set(ids[n_train:n_train+n_val])
test_ids  = set(ids[n_train+n_val:])

def split_for_id(v):
    if v in train_ids: return "train"
    if v in val_ids:   return "val"
    return "test"

df["split"] = df["vessel_id"].map(split_for_id)
split_path = ROOT / "vessel2258_meta_split.csv"
df.to_csv(split_path, index=False)

print(df["split"].value_counts())
print("Train IDs:", len(train_ids), "Val IDs:", len(val_ids), "Test IDs:", len(test_ids))
print("Saved:", meta_path, "and", split_path)
