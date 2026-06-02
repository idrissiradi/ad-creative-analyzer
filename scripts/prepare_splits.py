"""
prepare_splits.py — Generate train/val/test CSVs from labeled manifest.

Run this AFTER:
  1. download_open_images.py  → data/raw/manifest.csv (content_type labeled)
  2. Label Studio export      → data/labeled/mood_labels.csv (mood labeled)

Usage:
  uv run python scripts/prepare_splits.py

Output:
  data/splits/train.csv  (70%)
  data/splits/val.csv    (15%)
  data/splits/test.csv   (15%)
"""

import os
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

SPLITS_DIR = Path("data/splits")
RAW_DIR = Path("data/raw")
LABELED_DIR = Path("data/labeled")


def load_and_merge() -> pd.DataFrame:
    """
    Merge content_type labels (from OI download) with mood labels (from Label Studio).
    """
    manifest_path = RAW_DIR / "manifest.csv"
    if not manifest_path.exists():
        raise FileNotFoundError(
            "data/raw/manifest.csv not found.\n"
            "Run: uv run python scripts/download_open_images.py"
        )

    df = pd.read_csv(manifest_path)
    print(f"Manifest loaded: {len(df)} images")

    # Check if mood labels exist (from Label Studio export)
    mood_path = LABELED_DIR / "mood_labels.csv"
    if mood_path.exists():
        mood_df = pd.read_csv(mood_path)
        # mood_labels.csv must have columns: image_path, mood_label
        df = df.merge(
            mood_df[["image_path", "mood_label"]],
            on="image_path",
            how="left",
            suffixes=("_raw", ""),
        )
        # If mood_label exists in both, use the new one
        if "mood_label_raw" in df.columns:
            df["mood_label"] = df["mood_label"].fillna(df["mood_label_raw"])
            df.drop(columns=["mood_label_raw"], inplace=True)
        print(f"Mood labels merged from: {mood_path}")
    else:
        print("⚠  mood_labels.csv not found — mood_label will be -1 (unlabeled)")
        print("   Label moods in Label Studio and re-run this script.")

    # Drop rows with missing labels
    before = len(df)
    df = df[df["mood_label"] != -1].copy()
    after = len(df)
    if before != after:
        print(f"   Dropped {before - after} unlabeled rows (mood_label = -1)")

    return df


def stratified_split(df: pd.DataFrame) -> tuple:
    """
    Stratified split on content_type to keep class balance in all 3 sets.
    70% train / 15% val / 15% test
    """
    # Create a combined label for stratification
    df["strat_key"] = (
        df["content_type_label"].astype(str) + "_" + df["mood_label"].astype(str)
    )

    train_df, temp_df = train_test_split(
        df, test_size=0.30, stratify=df["strat_key"], random_state=42
    )
    val_df, test_df = train_test_split(
        temp_df, test_size=0.50, stratify=temp_df["strat_key"], random_state=42
    )

    train_df = train_df.drop(columns=["strat_key"])
    val_df = val_df.drop(columns=["strat_key"])
    test_df = test_df.drop(columns=["strat_key"])

    return train_df, val_df, test_df


def save_splits(train_df, val_df, test_df):
    SPLITS_DIR.mkdir(parents=True, exist_ok=True)

    train_df.to_csv(SPLITS_DIR / "train.csv", index=False)
    val_df.to_csv(SPLITS_DIR / "val.csv", index=False)
    test_df.to_csv(SPLITS_DIR / "test.csv", index=False)

    print(f"\n✅ Splits saved to {SPLITS_DIR}/")
    print(f"   train : {len(train_df)} samples")
    print(f"   val   : {len(val_df)} samples")
    print(f"   test  : {len(test_df)} samples")

    print("\n   Content type distribution in train:")
    ct_names = ["Product Showcase", "Lifestyle", "Testimonial", "Promotional"]
    for i, name in enumerate(ct_names):
        count = (train_df["content_type_label"] == i).sum()
        print(f"     {name}: {count}")


def main():
    print("=" * 60)
    print("Prepare Train/Val/Test Splits")
    print("=" * 60)

    df = load_and_merge()

    if len(df) < 100:
        print(f"\n⚠  Only {len(df)} labeled images. Need at least 100 to split.")
        print("   Complete labeling in Label Studio first.")
        return

    train_df, val_df, test_df = stratified_split(df)
    save_splits(train_df, val_df, test_df)


if __name__ == "__main__":
    main()
