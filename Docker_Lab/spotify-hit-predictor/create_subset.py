"""
create_subset.py — run this ONCE locally before pushing to git.

Input:  data/spotify_tracks.csv   (full Kaggle dataset, git-ignored)
Output: data/spotify_subset.csv   (5k stratified rows, committed to git)

Stratification: balances across popularity buckets so the model
sees enough hits AND non-hits in training.
"""

import pandas as pd
from pathlib import Path

INPUT  = Path("data/spotify_tracks.csv")
OUTPUT = Path("data/spotify_subset.csv")
N      = 5000
SEED   = 42

FEATURES = [
    "energy", "danceability", "tempo", "valence",
    "loudness", "acousticness", "speechiness", "popularity"
]

def main():
    if not INPUT.exists():
        raise FileNotFoundError(
            f"Could not find {INPUT}\n"
            "Download from: https://www.kaggle.com/datasets/maharshipandya/-spotify-tracks-dataset\n"
            "and place it at data/spotify_tracks.csv"
        )

    df = pd.read_csv(INPUT)
    print(f"Loaded {len(df):,} rows from {INPUT}")

    # Rename if needed
    if "track_popularity" in df.columns:
        df.rename(columns={"track_popularity": "popularity"}, inplace=True)

    df = df[FEATURES].dropna()

    # Stratify by popularity bucket (0-29 / 30-59 / 60-69 / 70-100)
    bins   = [0, 30, 60, 70, 101]
    labels = ["low", "mid", "high", "hit"]
    df["_bucket"] = pd.cut(df["popularity"], bins=bins, labels=labels, right=False)

    subset = (
        df.groupby("_bucket", observed=True)
          .apply(lambda g: g.sample(min(len(g), N // len(labels)), random_state=SEED))
          .reset_index(drop=True)
    )
    # Top up to exactly N if stratified sample came short
    if len(subset) < N:
        extra = df.drop(subset.index, errors="ignore").sample(
            N - len(subset), random_state=SEED
        )
        subset = pd.concat([subset, extra]).reset_index(drop=True)

    subset = subset.drop(columns=["_bucket"]).sample(frac=1, random_state=SEED)

    OUTPUT.parent.mkdir(exist_ok=True)
    subset.to_csv(OUTPUT, index=False)

    print(f"Saved {len(subset):,} rows → {OUTPUT}")
    print(f"Hit rate (popularity ≥ 70): {(subset['popularity'] >= 70).mean():.1%}")
    print(f"File size: {OUTPUT.stat().st_size / 1024:.0f} KB")

if __name__ == "__main__":
    main()