import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import pandas as pd
from tqdm import tqdm


from models.global_part_resnet import GlobalPartReIDResNet


ROOT = r"C:\Users\shall\VesselReID\REID_2258"
CSV  = r"C:\Users\shall\VesselReID\REID_2258\vessel2258_meta_split.csv"
BATCH_SIZE = 32
LR = 3e-4
MAX_EPOCHS = 20
NUM_PARTS = 3
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class TrainDataset(Dataset):
    def __init__(self, root_dir, df, transform=None):
        self.root_dir = root_dir
        self.df = df
        self.transform = transform
        
        self.vessel_ids = sorted(self.df['vessel_id'].unique())
        self.id_to_label = {vid: i for i, vid in enumerate(self.vessel_ids)}

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        rel_path = row['rel_path']          # e.g. "Cargo ship\0001_c1s1_000001_01.jpg"
        vid = row['vessel_id']
        label = self.id_to_label[vid]

        img_path = os.path.join(self.root_dir, rel_path)
        img_path = img_path.replace("\\", os.sep).replace("/", os.sep)

        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image, label, vid


def train():
    print("Loading Data...")
    df = pd.read_csv(CSV)
    df_train = df[df["split"] == "train"].reset_index(drop=True)
    df_val = df[df["split"] == "val"].reset_index(drop=True)
    
    transform_train = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    transform_val = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    train_dataset = TrainDataset(ROOT, df_train, transform=transform_train)
    val_dataset = TrainDataset(ROOT, df_val, transform=transform_val)
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False,
                            num_workers=2, pin_memory=True)
    
    num_classes = len(train_dataset.vessel_ids)
    print(f"Training on {len(df_train)} images, {num_classes} classes.")
    print(f"Validation on {len(df_val)} images.")
    
    model = GlobalPartReIDResNet(num_classes=num_classes, feat_dim=512, num_parts=NUM_PARTS)
    model.to(DEVICE)
    
    optimizer = optim.Adam(model.parameters(), lr=LR)
    criterion_ce = nn.CrossEntropyLoss()
    
    print(f"Start training Global + {NUM_PARTS} Parts on {DEVICE}...")
    os.makedirs("checkpoints", exist_ok=True)

    best_val_acc = 0.0
    patience_counter = 0
    patience = 2  # stop if validation accuracy doesn't improve for 2 consecutive epochs
    patience_counter_max = patience

    for epoch in range(MAX_EPOCHS):
        model.train()
        total_loss = 0
        train_correct = 0
        train_total = 0
        
        train_pbar = tqdm(train_loader, desc=f"Train E{epoch+1:02d}", leave=False)
        for imgs, labels, _ in train_pbar:
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            
            g_logits, p_logits_list = model(imgs)
            
            loss_g = criterion_ce(g_logits, labels)
            loss_p = sum(criterion_ce(p_logit, labels) for p_logit in p_logits_list)
            loss = loss_g + loss_p
            
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            _, preds = torch.max(g_logits, 1)
            train_correct += (preds == labels).sum().item()
            train_total += labels.size(0)
            
            train_pbar.set_postfix({
                "Loss": f"{loss.item():.4f}", 
                "Acc": f"{train_correct/train_total:.1%}"
            })
        
        train_loss = total_loss / len(train_loader)
        train_acc = train_correct / train_total
        
        model.eval()
        val_correct = 0
        val_total = 0
        
        val_pbar = tqdm(val_loader, desc=f"Val E{epoch+1:02d}", leave=False)
        with torch.no_grad():
            for imgs, labels, _ in val_pbar:
                imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
                g_logits, _ = model(imgs)
                _, preds = torch.max(g_logits, 1)
                val_correct += (preds == labels).sum().item()
                val_total += labels.size(0)
                val_pbar.set_postfix({"Val Acc": f"{val_correct/val_total:.1%}"})
        
        val_acc = val_correct / val_total
        
        print(f"Epoch {epoch+1}: Train Loss={train_loss:.4f}, Train Acc={train_acc:.3f}, Val Acc={val_acc:.3f}")
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
            best_path = os.path.join("checkpoints", "global_part_best.pth")
            torch.save(model.state_dict(), best_path)
            print(f"  -> New best model saved! Val Acc = {val_acc:.3f}")
        else:
            patience_counter += 1
            print(f"  -> Patience {patience_counter}/{patience}")
        
        if patience_counter >= patience:
            print(f"Early stopping at epoch {epoch+1}!")
            break
        
        torch.save(model.state_dict(), f"checkpoints/global_part_epoch{epoch+1:02d}.pth")

    print(f"\nTraining finished!")
    print(f"Best validation accuracy: {best_val_acc:.3f}")
    print(f"Final model saved as 'checkpoints/global_part_best.pth'")


if __name__ == "__main__":
    train()
