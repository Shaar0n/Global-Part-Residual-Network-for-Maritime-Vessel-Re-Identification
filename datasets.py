from pathlib import Path
from torch.utils.data import Dataset
from PIL import Image
import pandas as pd


class Vessel2258Dataset(Dataset):
    def __init__(self, root, csv_path, split="train", transform=None):
        self.root = Path(root)
        self.df = pd.read_csv(csv_path)

        # keep only the chosen split
        self.df = self.df[self.df["split"] == split].reset_index(drop=True)
        self.transform = transform

        # map vessel_id -> 0..num_classes-1 for CE loss
        ids = sorted(self.df["vessel_id"].unique())
        self.id2label = {vid: i for i, vid in enumerate(ids)}
        self.df["label"] = self.df["vessel_id"].map(self.id2label)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = self.root / row["rel_path"]
        img = Image.open(img_path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        label = int(row["label"])
        cam_id = int(row["camera_id"])
        return img, label, cam_id


class Vessel2258EvalDataset(Dataset):

    def __init__(self, root, df, transform=None):
        self.root = Path(root)
        self.df = df.reset_index(drop=True)
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = self.root / row["rel_path"]
        img = Image.open(img_path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        vid = int(row["vessel_id"])
        cam = int(row["camera_id"])
        return img, vid, cam

