"""
Randomly assigns real redshifts from gz2sample.csv.gz to your GalaxyIDs.

Since the Kaggle GalaxyIDs (e.g. 100008) are internal IDs that don't match
SDSS objids in the gz2 datasets, we sample from the real redshift distribution
in gz2sample to keep the values physically realistic.

Usage:
    python assign_redshifts.py \
        --galaxy-ids training_solutions_rev1.csv \   # or a plain list of IDs
        --redshift-source gz2sample.csv.gz \
        --output galaxy_redshifts.csv \
        --seed 42

Inputs:
    --galaxy-ids   CSV with a 'GalaxyID' column  (or a .txt with one ID per line)
    --redshift-source  gz2sample.csv.gz (has OBJID + REDSHIFT columns)
    --output       output CSV path
    --seed         random seed for reproducibility
"""

import argparse
import gzip
import pandas as pd
import numpy as np


def load_galaxy_ids(path: str) -> list:
    if path.endswith(".csv") or path.endswith(".csv.gz"):
        df = pd.read_csv(path)
        # try common column names
        for col in ["GalaxyID", "galaxy_id", "objid", "OBJID", "id", "ID"]:
            if col in df.columns:
                return df[col].tolist()
        raise ValueError(f"No recognised ID column in {path}. Columns: {df.columns.tolist()}")
    else:
        # plain text, one ID per line
        with open(path) as f:
            return [line.strip() for line in f if line.strip()]


def load_redshifts(gz_path: str) -> np.ndarray:
    print(f"Loading redshifts from {gz_path} ...")
    df = pd.read_csv(gz_path, usecols=["REDSHIFT"])
    redshifts = pd.to_numeric(df["REDSHIFT"], errors="coerce")
    redshifts = redshifts.dropna()
    redshifts = redshifts[redshifts > 0]          # drop missing / sentinel values
    redshifts = redshifts[redshifts < 1.0]        # keep physically sensible range
    arr = redshifts.values
    print(f"  Valid redshifts: {len(arr):,}  "
          f"(min={arr.min():.4f}, median={np.median(arr):.4f}, max={arr.max():.4f})")
    return arr


def main():
    parser = argparse.ArgumentParser(description="Assign redshifts to GalaxyIDs by random sampling.")
    parser.add_argument("--galaxy-ids",      default="training_solutions_rev1.csv",
                        help="CSV (with GalaxyID column) or text file with one ID per line")
    parser.add_argument("--redshift-source", default="gz2sample.csv.gz",
                        help="gz2sample.csv.gz (must have a REDSHIFT column)")
    parser.add_argument("--output",          default="galaxy_redshifts.csv")
    parser.add_argument("--seed",            type=int, default=42)
    args = parser.parse_args()

    # ── 1. Load your GalaxyIDs ──────────────────────────────────────────────
    galaxy_ids = load_galaxy_ids(args.galaxy_ids)
    print(f"GalaxyIDs loaded: {len(galaxy_ids):,}")

    # ── 2. Load real redshifts ──────────────────────────────────────────────
    redshifts = load_redshifts(args.redshift_source)

    # ── 3. Random sample (with replacement so any size works) ───────────────
    rng = np.random.default_rng(args.seed)
    sampled = rng.choice(redshifts, size=len(galaxy_ids), replace=True)

    # ── 4. Build output dataframe ───────────────────────────────────────────
    out = pd.DataFrame({
        "GalaxyID":      galaxy_ids,
        "redshift":      sampled
    })

    out.to_csv(args.output, index=False)
    print(f"\nSaved {len(out):,} rows → {args.output}")
    print(out.head())


if __name__ == "__main__":
    main()
