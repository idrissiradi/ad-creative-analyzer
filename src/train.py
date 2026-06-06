import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchmetrics.classification import MulticlassF1Score

from src.dataset import AdImageDataset
from src.image_model import AdImageModel

NUM_CONTENT = 3
NUM_MOODS = 4


def get_device(requested: str = "auto") -> str:
    if requested == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return requested


def calculate_class_weights(
    csv_path, column_name, num_classes, device: str
) -> torch.Tensor:
    """Compute inverse-frequency weights from training CSV."""
    df = pd.read_csv(csv_path)

    # Fill missing classes with 0 to avoid index errors
    counts = df[column_name].value_counts().reindex(range(num_classes), fill_value=0)

    counts = torch.tensor(counts.values, dtype=torch.float32)
    counts = counts.clamp(min=1)

    weights = 1.0 / counts
    weights = weights / weights.sum() * num_classes  # normalize
    return weights.to(device)


def train_epoch(model, loader, optimizer, criterion_content, criterion_mood, device):
    """Train for one epoch and return average loss."""
    model.train()
    total_loss = 0.0
    correct_content = 0
    correct_mood = 0
    total_samples = 0

    for batch in loader:
        image = batch["image"].to(device)
        mood_labels = batch["mood"].to(device)
        content_labels = batch["content_type"].to(device)

        optimizer.zero_grad()

        content_logits, mood_logits = model(image)

        loss_content = criterion_content(content_logits, content_labels)
        loss_mood = criterion_mood(mood_logits, mood_labels)
        loss = loss_content + loss_mood

        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        _, content_preds = torch.max(content_logits, 1)
        _, mood_preds = torch.max(mood_logits, 1)

        # Update correct predictions
        correct_content += (content_preds == content_labels).sum().item()
        correct_mood += (mood_preds == mood_labels).sum().item()
        total_samples += image.size(0)

    # Compute average metrics
    avg_loss = total_loss / len(loader)
    content_acc = 100.0 * correct_content / total_samples
    mood_acc = 100.0 * correct_mood / total_samples
    return avg_loss, content_acc, mood_acc


@torch.no_grad()
def eval_epoch(model, loader, device):
    model.eval()

    f1_ct = MulticlassF1Score(num_classes=NUM_CONTENT, average="weighted").to(device)
    f1_mood = MulticlassF1Score(num_classes=NUM_MOODS, average="weighted").to(device)

    for batch in loader:
        imgs = batch["image"].to(device)
        ct_labels = batch["content_type"].to(device)
        m_labels = batch["mood"].to(device)

        content_logits, mood_logits = model(imgs)

        f1_ct.update(content_logits, ct_labels)
        f1_mood.update(mood_logits, m_labels)

    return f1_ct.compute().item(), f1_mood.compute().item()
