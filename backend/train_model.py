from models.model import GalaxyClassifier
import torch
from data.transform_data import (
    train_transform,
    val_transform,
    GalaxyDataset
)

from torch.utils.data import DataLoader
from data.make_labels import df
from sklearn.model_selection import train_test_split
import torch.nn as nn
from tqdm import tqdm
import torch.optim as optim
import os

# ── paths ───────────────────────────────────────────────────────────────
IMAGE_DIR = "/content/images_training_rev1"

# ── device ──────────────────────────────────────────────────────────────
device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("Using device:", device)

# ── model ───────────────────────────────────────────────────────────────
model = GalaxyClassifier().to(device)

# ── split data ──────────────────────────────────────────────────────────
train_df, val_df = train_test_split(
    df,
    test_size=0.2,
    random_state=0,
    stratify=df["label"]
)

train_df = train_df.reset_index(drop=True)
val_df = val_df.reset_index(drop=True)

# ── datasets ────────────────────────────────────────────────────────────
train_dataset = GalaxyDataset(
    IMAGE_DIR,
    train_df,
    transform=train_transform
)

val_dataset = GalaxyDataset(
    IMAGE_DIR,
    val_df,
    transform=val_transform
)

# ── loaders ─────────────────────────────────────────────────────────────
train_loader = DataLoader(
    train_dataset,
    batch_size=16,
    shuffle=True,
    num_workers=2,
    pin_memory=True
)

val_loader = DataLoader(
    val_dataset,
    batch_size=16,
    shuffle=False,
    num_workers=2,
    pin_memory=True
)

# ── training loop ──────────────────────────────────────────────────────
def training_loop(model, train_loader, val_loader, epochs=100):

    criterion = nn.CrossEntropyLoss()

    optimizer = optim.AdamW(
        model.parameters(),
        lr=1e-4
    )

    records = {
        "train_loss": [],
        "val_loss": []
    }

    for epoch in range(epochs):

        # ── TRAIN ───────────────────────────────────────────────────────
        model.train()

        train_loss = 0

        for images, labels in tqdm(
            train_loader,
            desc=f"Epoch {epoch+1} Training"
        ):

            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            logits, _ = model(images)

            loss = criterion(logits, labels)

            train_loss += loss.item()

            loss.backward()

            optimizer.step()

        # ── VALIDATION ─────────────────────────────────────────────────
        model.eval()

        val_loss = 0

        with torch.no_grad():

            for images, labels in tqdm(
                val_loader,
                desc=f"Epoch {epoch+1} Validation"
            ):

                images = images.to(device)
                labels = labels.to(device)

                logits, _ = model(images)

                loss = criterion(logits, labels)

                val_loss += loss.item()

        avg_train_loss = train_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)

        records["train_loss"].append(avg_train_loss)
        records["val_loss"].append(avg_val_loss)

        print(
            f"Epoch {epoch+1} | "
            f"train_loss: {avg_train_loss:.4f} | "
            f"val_loss: {avg_val_loss:.4f}"
        )

    return records


# ── main ────────────────────────────────────────────────────────────────
if __name__ == "__main__":

    records = training_loop(
        model,
        train_loader,
        val_loader,
        epochs=100
    )

    os.makedirs("backend/weights", exist_ok=True)

    torch.save(
        model.state_dict(),
        "backend/weights/galaxy_classifier.pth"
    )

    print("Model saved.")