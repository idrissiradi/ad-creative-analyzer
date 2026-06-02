import argparse
import os
import shutil
from pathlib import Path

import fiftyone as fo
import fiftyone.zoo as foz
import pandas as pd
from tqdm import tqdm

DOWNLOAD_PLAN = [
    # Content Type 0 — Product Showcase (375)
    ("Coffee cup", 0, 80),
    ("Cosmetics", 0, 80),
    ("Clothing", 0, 80),
    ("Footwear", 0, 75),
    ("Mobile phone", 0, 60),
    # Content Type 1 — Lifestyle (375)
    ("Hiking equipment", 1, 70),
    ("Sports equipment", 1, 70),
    ("Yoga", 1, 70),
    ("Cooking", 1, 75),
    ("Swimming", 1, 90),
    # Content Type 2 — Testimonial (375)
    ("Man", 2, 130),
    ("Woman", 2, 130),
    ("Smile", 2, 115),
    # Content Type 3 — Promotional (375)
    ("Sale sign", 3, 80),
    ("Billboard", 3, 80),
    ("Poster", 3, 80),
    ("Banner", 3, 80),
    ("Gift wrapping", 3, 55),
]

CONTENT_TYPE_NAMES = [
    "Product Showcase",
    "Lifestyle",
    "Testimonial",
    "Promotional",
]


def download_with_fiftyone(plan: list, output_dir: str, split: str = "train"):
    """Download images using fiftyone Open Images v7 integration."""

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    records = []
    seen_img_ids = set()

    for label_name, content_type_id, count in tqdm(plan, desc="Downloading categories"):
        print(
            f"\n→ Downloading {count}x '{label_name}' (content_type={content_type_id})"
        )

        try:
            dataset = foz.load_zoo_dataset(
                "open-images-v7",
                split=split,
                label_types=["classifications"],
                classes=[label_name],
                max_samples=count,
                shuffle=True,
                seed=42,
            )
        except Exception as e:
            print(f"  ⚠ Skipping '{label_name}': {e}")
            continue

        for sample in dataset:
            src = sample.filepath
            # Copy to our data/raw folder with a clean name
            img_id = Path(src).stem

            if img_id in seen_img_ids:
                continue

            ext = Path(src).suffix
            dst_name = f"{content_type_id}_{img_id}{ext}"
            dst_path = output_path / dst_name

            try:
                shutil.copy2(src, dst_path)
            except Exception as copy_err:
                print(f"  ⚠ Failed to copy {src}: {copy_err}")
                continue

            seen_img_ids.add(img_id)

            records.append(
                {
                    "image_path": dst_name,
                    "content_type_label": content_type_id,
                    "content_type_name": CONTENT_TYPE_NAMES[content_type_id],
                    "mood_label": -1,
                    "oi_class": label_name,
                }
            )

        # Cleanup fiftyone dataset from memory
        fo.delete_dataset(dataset.name)

    df = pd.DataFrame(records).drop_duplicates(subset=["image_path"])
    manifest_path = Path(output_dir) / "manifest.csv"
    df.to_csv(manifest_path, index=False)

    print(f"\n✅ Done. {len(df)} images saved to {output_dir}")
    print(f"   Manifest: {manifest_path}")
    print("\n   Class distribution:")
    print(df["content_type_name"].value_counts().to_string())
    return df


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--total", type=int, default=1500, help="Target number of images (default 1500)"
    )
    parser.add_argument(
        "--output", type=str, default="data/raw", help="Output directory"
    )
    parser.add_argument(
        "--split", type=str, default="train", choices=["train", "validation", "test"]
    )
    args = parser.parse_args()

    print("=" * 60)
    print("Open Images v7 Downloader — Ad Creative Analyzer PFE")
    print("=" * 60)
    print(f"Target: {args.total} images → {args.output}")
    print()

    df = download_with_fiftyone(DOWNLOAD_PLAN, args.output, args.split)
    return df


if __name__ == "__main__":
    main()
