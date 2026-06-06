import os
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

SPLITS_DIR = Path("data/splits")
LABELED_DIR = Path("data/labeled")

MOOD_MAP = {
    "Energetic": 0,
    "Calm": 1,
    "Professional": 2,
    "Playful": 3,
}

CONTENT_TYPE_MAP = {
    "Product Showcase": 0,
    "Lifestyle": 1,
    "Promotional": 2,
}

CONTENT_TYPE_NAMES = ["Product Showcase", "Lifestyle", "Promotional"]
MOOD_NAMES = ["Energetic", "Calm", "Professional", "Playful"]


def load_data() -> pd.DataFrame:
    """
    Load labels.csv and change content_type and mood from string to int labels.
    - content_type: Product Showcase=0, Lifestyle=1, Testimonial=2, Promotional=3
    - mood: Calm=0, Happy=1, Energetic=2, Sad=3, Angry=4
    """
    labeled_path = LABELED_DIR / "labels.csv"
    if not labeled_path.exists():
        raise FileNotFoundError("data/labeled/labels.csv not found.")

    df = pd.read_csv(labeled_path)
    print(f"Labels loaded: {len(df)} images")
    print(f"Columns: {list(df.columns)}")

    if "image_name" in df.columns:
        df = df.rename(columns={"image_name": "image_path"})

    df["content_type_label"] = df["content_type"].map(CONTENT_TYPE_MAP)
    df["mood_label"] = df["mood"].map(MOOD_MAP)

    df = df[["image_path", "content_type_label", "mood_label"]].copy()
    df = df.reset_index(drop=True)
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
    """Save the splits to CSV files in data/splits/ directory."""
    SPLITS_DIR.mkdir(parents=True, exist_ok=True)

    train_df.to_csv(SPLITS_DIR / "train.csv", index=False)
    val_df.to_csv(SPLITS_DIR / "val.csv", index=False)
    test_df.to_csv(SPLITS_DIR / "test.csv", index=False)

    print(f"\n✅ Splits saved to {SPLITS_DIR}/")
    print(f"   train : {len(train_df)} samples")
    print(f"   val   : {len(val_df)} samples")
    print(f"   test  : {len(test_df)} samples")

    print("\n   Content type distribution in train:")

    for i, name in enumerate(CONTENT_TYPE_NAMES):
        count = (train_df["content_type_label"] == i).sum()
        print(f"{name}: {count}")


def main():
    print("=" * 60)
    print("Prepare Train/Val/Test Splits")
    print("=" * 60)

    df = load_data()

    if len(df) < 100:
        print(f"\n⚠  Only {len(df)} labeled images. Need at least 100 to split.")
        return

    train_df, val_df, test_df = stratified_split(df)
    save_splits(train_df, val_df, test_df)


if __name__ == "__main__":
    main()
