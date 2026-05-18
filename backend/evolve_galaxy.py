import os
import torch
import pandas as pd
from PIL import Image
from torchvision import transforms

from models.vae import GalaxyVAE

# ── config ────────────────────────────────────────────────────────────────────
IMAGE_DIR      = "images_training_rev1"
REDSHIFT_CSV   = "galaxy_redshifts.csv"       # from assign_redshifts.py
VAE_WEIGHTS    = "backend/weights/galaxy_vae.pth"
LATENT_DIM     = 128
IMG_SIZE       = 224
N_FRAMES       = 20

Z_OLD_MIN      = 0.15     # high redshift = farther away = older light
Z_NEW_MAX      = 0.05     # low redshift  = nearby       = modern galaxies
MAX_SAMPLES    = 300      # galaxies to sample when computing time direction

DEVICE = (
    "cuda" if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available()
    else "cpu"
)
# ─────────────────────────────────────────────────────────────────────────────

val_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
])


def load_vae(weights_path: str) -> GalaxyVAE:
    vae = GalaxyVAE(latent_dim=LATENT_DIM).to(DEVICE)
    vae.load_state_dict(torch.load(weights_path, map_location=DEVICE))
    vae.eval()
    print(f"VAE loaded from {weights_path}")
    return vae


def encode_image(vae: GalaxyVAE, img_path: str):
    """Encode a single image to its latent mean vector (mu). Returns None if file missing."""
    try:
        img    = Image.open(img_path).convert("RGB")
        tensor = val_transform(img).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            _, mu, _ = vae(tensor)
        return mu.squeeze(0).cpu()
    except FileNotFoundError:
        return None


def compute_time_offset(vae: GalaxyVAE, redshift_csv: str) -> torch.Tensor:
    """
    Compute the 'time direction' vector in latent space.

    1. Load galaxy_redshifts.csv (GalaxyID + redshift).
    2. Split into old (high-z) and young (low-z) galaxies.
    3. Encode a sample of each group through the trained VAE.
    4. offset = mean(old latents) - mean(young latents)
       Adding this to any galaxy latent pushes it 'back in time'.
    """
    import random
    random.seed(42)

    df = pd.read_csv(redshift_csv)
    print(f"Redshift file loaded: {len(df):,} galaxies")

    old_ids = df[df["redshift"] > Z_OLD_MIN]["GalaxyID"].tolist()
    new_ids = df[df["redshift"] < Z_NEW_MAX]["GalaxyID"].tolist()
    print(f"  High-z (old) galaxies : {len(old_ids):,}")
    print(f"  Low-z  (young) galaxies: {len(new_ids):,}")

    old_ids = random.sample(old_ids, min(MAX_SAMPLES, len(old_ids)))
    new_ids = random.sample(new_ids, min(MAX_SAMPLES, len(new_ids)))

    def encode_batch(ids):
        mus = []
        for gid in ids:
            path = os.path.join(IMAGE_DIR, f"{gid}.jpg")
            mu   = encode_image(vae, path)
            if mu is not None:
                mus.append(mu)
        return mus

    print("Encoding old galaxies...")
    z_old = encode_batch(old_ids)
    print(f"  encoded {len(z_old)} images")

    print("Encoding young galaxies...")
    z_new = encode_batch(new_ids)
    print(f"  encoded {len(z_new)} images")

    if not z_old or not z_new:
        raise RuntimeError(
            "Could not encode any galaxies. "
            "Check IMAGE_DIR and REDSHIFT_CSV paths."
        )

    z_old_mean = torch.stack(z_old).mean(dim=0)
    z_new_mean = torch.stack(z_new).mean(dim=0)
    offset     = z_old_mean - z_new_mean

    print(f"Time offset vector computed. Norm: {offset.norm():.4f}")
    return offset


def slerp(z0: torch.Tensor, z1: torch.Tensor, t: float) -> torch.Tensor:
    """Spherical linear interpolation between two latent vectors."""
    z0_n  = z0 / z0.norm().clamp(min=1e-8)
    z1_n  = z1 / z1.norm().clamp(min=1e-8)
    dot   = (z0_n * z1_n).sum().clamp(-1.0, 1.0)
    omega = torch.acos(dot)
    if omega.abs() < 1e-6:
        return (1 - t) * z0 + t * z1
    return (torch.sin((1 - t) * omega) * z0 + torch.sin(t * omega) * z1) / torch.sin(omega)


def evolve_galaxy(
    vae: GalaxyVAE,
    image_tensor: torch.Tensor,
    time_offset: torch.Tensor,
    n_frames: int = N_FRAMES,
) -> list:
    """
    Generate n_frames images showing a galaxy evolving from early universe to now.

    Args:
        vae:           trained GalaxyVAE
        image_tensor:  preprocessed image tensor, shape (1, 3, H, W)
        time_offset:   vector from compute_time_offset()
        n_frames:      number of animation frames

    Returns:
        list of numpy arrays, shape (H, W, 3), float32 values in [0, 1]
        index 0 = earliest epoch, index -1 = today
    """
    image_tensor = image_tensor.to(DEVICE)
    time_offset  = time_offset.to(DEVICE)

    with torch.no_grad():
        _, mu_now, _ = vae(image_tensor)
        mu_now  = mu_now.squeeze(0)

        z_early = mu_now + time_offset
        z_today = mu_now

        frames = []
        for t in torch.linspace(0.0, 1.0, n_frames):
            z     = slerp(z_early, z_today, t.item())
            z_in  = z.unsqueeze(0).to(DEVICE)
            recon = vae.decoder(vae.fc_decode(z_in).view(1, 256, 14, 14))
            frame = recon.squeeze(0).permute(1, 2, 0).cpu().numpy()
            frames.append(frame)

    return frames


# ── smoke test ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys

    test_image = sys.argv[1] if len(sys.argv) > 1 else None

    if not os.path.exists(VAE_WEIGHTS):
        print(f"ERROR: VAE weights not found at '{VAE_WEIGHTS}'.")
        print("       Run train_vae.py first.")
        sys.exit(1)

    vae         = load_vae(VAE_WEIGHTS)
    time_offset = compute_time_offset(vae, REDSHIFT_CSV)

    if test_image:
        img    = Image.open(test_image).convert("RGB")
        tensor = val_transform(img).unsqueeze(0)
        frames = evolve_galaxy(vae, tensor, time_offset)
        print(f"\nGenerated {len(frames)} frames. Shape: {frames[0].shape}")
        print(f"Pixel range: {frames[0].min():.3f} to {frames[0].max():.3f}")
        print("evolve_galaxy() is working correctly.")
    else:
        print("\nNo test image provided — offset computed successfully.")
        print("Import evolve_galaxy() from this module in main.py to use it.")