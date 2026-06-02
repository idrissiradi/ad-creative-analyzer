"""
convert_ls_export.py — Convert Label Studio export CSV to mood_labels.csv.

Label Studio exports columns like: id, image_path, mood_label, content_type_label, etc.
This script normalizes that into the exact format prepare_splits.py expects.

Usage:
  uv run python scripts/convert_ls_export.py --input data/labeled/ls_export.csv
"""

import argparse
from pathlib import Path

import pandas as pd


def convert(input_path: str):
    df = pd.read_csv(input_path)
    print(f"Loaded {len(df)} rows from Label Studio export")
    print(f"Columns: {list(df.columns)}")

    # Label Studio exports annotations in various formats depending on version.
    # Common column names to try:
    mood_col = None
    for candidate in ["mood_label", "label", "choice", "annotation"]:
        if candidate in df.columns:
            mood_col = candidate
            break

    if mood_col is None:
        print("\nERROR: Could not find mood label column.")
        print("Available columns:", list(df.columns))
        print("Rename the mood column to 'mood_label' manually and re-run.")
        return

    # Normalize image path — Label Studio may include full path
    df["image_path"] = df["image_path"].apply(lambda p: Path(p).name)

    # Keep only what we need
    out_df = df[["image_path", mood_col]].copy()
    out_df = out_df.rename(columns={mood_col: "mood_label"})

    # Convert string labels to int if needed
    mood_map = {"Energetic": 0, "Calm": 1, "Professional": 2, "Playful": 3}
    if out_df["mood_label"].dtype == object:
        # Strip emoji prefix if present (⚡ Energetic → Energetic)
        out_df["mood_label"] = out_df["mood_label"].str.strip()
        out_df["mood_label"] = out_df["mood_label"].map(
            {**mood_map, **{str(v): v for v in mood_map.values()}}
        )

    out_df = out_df.dropna(subset=["mood_label"])
    out_df["mood_label"] = out_df["mood_label"].astype(int)

    output_path = Path("data/labeled/mood_labels.csv")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(output_path, index=False)

    print(f"\n✅ Saved {len(out_df)} mood labels → {output_path}")
    print("\nMood distribution:")
    labels = {0: "Energetic", 1: "Calm", 2: "Professional", 3: "Playful"}
    for i, name in labels.items():
        count = (out_df["mood_label"] == i).sum()
        print(f"  {name}: {count}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input", required=True, help="Path to Label Studio export CSV"
    )
    args = parser.parse_args()
    convert(args.input)
