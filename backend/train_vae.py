import os
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from data.transform_data import train_transform
from PIL import Image
import pandas as pd
from models.vae import GalaxyVAE, vae_loss

# ── config ─────────────────────────────────────────────────────────────
IMAGE_DIR = "/content/images_training_rev1"
LABELS_CSV = "/content/training_solutions_rev1.csv"

WEIGHTS_OUT = "backend/weights/galaxy_vae.pth"

LATENT_DIM = 128
BATCH_SIZE = 32
EPOCHS = 30
BETA = 1.0
LR = 1e-4
IMG_SIZE = 224

# Colab GPU setup
NUM_WORKERS = 2

DEVICE = (
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

print("Using device:", DEVICE)

# ── dataset ────────────────────────────────────────────────────────────
class GalaxyImageDataset(Dataset):

    def __init__(
        self,
        image_dir: str,
        galaxy_ids: list,
        transform
    ):
        self.image_dir = image_dir
        self.ids = galaxy_ids
        self.transform = transform

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):

        gid = self.ids[idx]

        img_path = os.path.join(
            self.image_dir,
            f"{gid}.jpg"
        )

        img = Image.open(img_path).convert("RGB")

        return self.transform(img)


# ── main ───────────────────────────────────────────────────────────────
def main():

    transform = train_transform

    df = pd.read_csv(LABELS_CSV)

    galaxy_ids = df["GalaxyID"].tolist()

    dataset = GalaxyImageDataset(
        IMAGE_DIR,
        galaxy_ids,
        transform
    )

    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=True
    )

    print(f"Dataset size: {len(dataset):,} images")

    # ── model ──────────────────────────────────────────────────────────
    vae = GalaxyVAE(
        latent_dim=LATENT_DIM
    ).to(DEVICE)

    optimizer = optim.AdamW(
        vae.parameters(),
        lr=LR,
        weight_decay=1e-4
    )

    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=EPOCHS
    )

    # ── training loop ──────────────────────────────────────────────────
    best_loss = float("inf")

    for epoch in range(1, EPOCHS + 1):

        vae.train()

        total_loss = 0.0

        for batch in loader:

            images = batch.to(DEVICE)

            optimizer.zero_grad()

            recon, mu, log_var = vae(images)

            loss = vae_loss(
                recon,
                images,
                mu,
                log_var,
                beta=BETA
            )

            loss.backward()

            torch.nn.utils.clip_grad_norm_(
                vae.parameters(),
                max_norm=1.0
            )

            optimizer.step()

            total_loss += loss.item()

        scheduler.step()

        avg_loss = total_loss / len(loader)

        print(
            f"Epoch {epoch:>3}/{EPOCHS} | "
            f"loss: {avg_loss:.2f}"
        )

        # ── save best model ───────────────────────────────────────────
        if avg_loss < best_loss:

            best_loss = avg_loss

            os.makedirs(
                os.path.dirname(WEIGHTS_OUT),
                exist_ok=True
            )

            torch.save(
                vae.state_dict(),
                WEIGHTS_OUT
            )

    print(f"\nDone. Weights saved to: {WEIGHTS_OUT}")


if __name__ == "__main__":
    main()