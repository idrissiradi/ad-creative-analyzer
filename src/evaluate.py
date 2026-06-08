import json
import os

import matplotlib.pyplot as plt
import pandas as pd
import torch
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    classification_report,
    confusion_matrix,
)
from torch.utils.data import DataLoader
from torchmetrics.classification import MulticlassF1Score

from src.dataset import AdImageDataset
from src.image_model import AdImageModel

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

CONTENT_TYPES = ["Product Showcase", "Lifestyle", "Promotional"]
MOODS = ["Energetic", "Calm", "Professional", "Playful"]

NUM_CONTENT = len(CONTENT_TYPES)  # 3
NUM_MOODS = len(MOODS)  # 4


def evaluate(
    checkpoint: str = "checkpoints/best_phase2.pt",
    img_dir: str = "data/raw",
    test_csv: str = "data/splits/test.csv",
    save_path: str = "eval/final_results.json",
) -> dict:
    """Evaluate the model on the test set and save results."""
    print(f"Loading model on {DEVICE}")

    model = AdImageModel(
        num_content_types=NUM_CONTENT, num_moods=NUM_MOODS, freeze_encoder=False
    )
    model.load_state_dict(
        torch.load(checkpoint, map_location=DEVICE, weights_only=True)
    )
    model.to(DEVICE)
    model.eval()

    print("Loading test data")
    test_ds = AdImageDataset(csv_path=test_csv, img_dir=img_dir, split="test")
    test_dl = DataLoader(test_ds, batch_size=32, shuffle=False, num_workers=2)

    f1_ct = MulticlassF1Score(num_classes=NUM_CONTENT, average="weighted").to(DEVICE)
    f1_mood = MulticlassF1Score(num_classes=NUM_MOODS, average="weighted").to(DEVICE)

    all_content_true = []
    all_content_pred = []
    all_mood_true = []
    all_mood_pred = []

    with torch.no_grad():
        for batch in test_dl:
            imgs = batch["image"].to(DEVICE)
            content_labels = batch["content_type"].to(DEVICE)
            mood_labels = batch["mood"].to(DEVICE)

            content_logits, mood_logits = model(imgs)

            f1_ct.update(content_logits, content_labels)
            f1_mood.update(mood_logits, mood_labels)

            all_content_true.extend(content_labels.cpu().numpy())
            all_content_pred.extend(content_logits.argmax(dim=1).cpu().numpy())
            all_mood_true.extend(mood_labels.cpu().numpy())
            all_mood_pred.extend(mood_logits.argmax(dim=1).cpu().numpy())

    content_score = f1_ct.compute().item()
    mood_score = f1_mood.compute().item()

    print("\n" + "=" * 50)
    print("TEST RESULTS")
    print("=" * 50)
    print(f"Content Type F1 : {content_score:.4f}")
    print(f"Mood F1         : {mood_score:.4f}")
    print(f"Average F1      : {(content_score + mood_score) / 2:.4f}")

    print("\nConfusion Matrices")
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    os.makedirs("eval", exist_ok=True)
    save_dir = os.path.dirname(save_path)

    print("\nContent Type Confusion Matrix:")
    content_cm = confusion_matrix(
        all_content_true, all_content_pred, labels=list(range(NUM_CONTENT))
    )
    disp_content = ConfusionMatrixDisplay(
        confusion_matrix=content_cm, display_labels=CONTENT_TYPES
    )
    fig_content, ax_ct = plt.subplots(figsize=(8, 6))

    disp_content.plot(cmap="Blues", ax=ax_ct, colorbar=False)
    plt.title(
        "Content Type Confusion Matrix (Test Set)", fontsize=14, fontweight="bold"
    )
    plt.tight_layout()
    plt.savefig(f"{save_dir}/content_type_confusion_matrix.png", dpi=300)
    plt.close()

    print("\nMood Confusion Matrix:")
    mood_cm = confusion_matrix(
        all_mood_true, all_mood_pred, labels=list(range(NUM_MOODS))
    )
    disp_mood = ConfusionMatrixDisplay(confusion_matrix=mood_cm, display_labels=MOODS)
    fig_mood, ax_mood = plt.subplots(figsize=(8, 6))
    disp_mood.plot(cmap="Blues", ax=ax_mood, colorbar=False)
    plt.title("Mood Confusion Matrix (Test Set)", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(f"{save_dir}/mood_confusion_matrix.png", dpi=300)
    plt.close()

    print(
        "Saved 'eval/content_type_confusion_matrix.png' and 'eval/mood_confusion_matrix.png'"
    )

    # classification report
    print("\nContent Type Classification Report:")
    print(
        classification_report(
            all_content_true,
            all_content_pred,
            target_names=CONTENT_TYPES,
            zero_division=0,
        )
    )

    print("\nMood Classification Report:")
    print(
        classification_report(
            all_mood_true, all_mood_pred, target_names=MOODS, zero_division=0
        )
    )

    results = {
        "test_f1_content_type": round(content_score, 4),
        "test_f1_mood": round(mood_score, 4),
        "target_content_type": 0.78,
        "target_mood": 0.75,
        "content_types": CONTENT_TYPES,
        "moods": MOODS,
        "checkpoint": checkpoint,
    }

    with open(save_path, "w") as f:
        json.dump(results, f, indent=2)
    print("\nResults saved")

    return results
