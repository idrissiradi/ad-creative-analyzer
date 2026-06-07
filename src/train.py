import argparse
import os

import mlflow
import mlflow.pytorch
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
    """Evaluate on validation set and return F1 scores."""
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


def main(args):
    device = get_device(getattr(args, "device", "auto"))
    save_dir = getattr(args, "save_dir", "checkpoints")
    os.makedirs(save_dir, exist_ok=True)

    # Model setup
    freeze = args.phase == 1
    model = AdImageModel(
        num_content_types=NUM_CONTENT, num_moods=NUM_MOODS, freeze_encoder=freeze
    ).to(device)

    if args.phase == 2 and getattr(args, "checkpoint", None):
        print("  Loading Phase 1 checkpoint")
        model.load_state_dict(
            torch.load(args.checkpoint, map_location=device, weights_only=True)
        )
        model.unfreeze_last_blocks(num_blocks=3)
        print("  Unfroze last 3 encoder blocks")

    # Data setup
    img_dir = getattr(args, "img_dir", "data/raw")
    train_csv = getattr(args, "train_csv", "data/splits/train.csv")
    val_csv = getattr(args, "val_csv", "data/splits/val.csv")
    batch_size = getattr(args, "batch_size", 16)

    train_ds = AdImageDataset(csv_path=train_csv, img_dir=img_dir, split="train")
    val_ds = AdImageDataset(csv_path=val_csv, img_dir=img_dir, split="val")

    train_dl = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=(device == "cuda"),
    )
    val_dl = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=(device == "cuda"),
    )

    # Loss and optimizer
    content_weights = calculate_class_weights(
        train_csv, "content_type_label", NUM_CONTENT, device
    )
    mood_weights = calculate_class_weights(train_csv, "mood_label", NUM_MOODS, device)

    content_criterion = nn.CrossEntropyLoss(weight=content_weights)
    mood_criterion = nn.CrossEntropyLoss(weight=mood_weights)

    lr = getattr(args, "lr", 1e-4)

    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=lr,
        weight_decay=1e-4,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    # Training loop
    best_f1 = 0.0
    checkpoint_path = os.path.join(save_dir, f"best_phase{args.phase}.pt")

    with mlflow.start_run(run_name=f"phase{args.phase}_e{args.epochs}"):
        mlflow.log_params(
            {
                "phase": args.phase,
                "epochs": args.epochs,
                "lr": lr,
                "batch_size": batch_size,
                "device": device,
                "num_content": NUM_CONTENT,
                "num_moods": NUM_MOODS,
                "freeze_encoder": freeze,
            }
        )

        for epoch in range(1, args.epochs + 1):
            loss, content_acc, mood_acc = train_epoch(
                model, train_dl, optimizer, content_criterion, mood_criterion, device
            )
            f1_content, f1_mood = eval_epoch(model, val_dl, device)
            avg_f1 = (f1_content + f1_mood) / 2
            scheduler.step()

            mlflow.log_metrics(
                {
                    "train_loss": loss,
                    "train_content_accuracy": content_acc,
                    "train_mood_accuracy": mood_acc,
                    "val_f1_content": f1_content,
                    "val_f1_mood": f1_mood,
                    "val_avg_f1": avg_f1,
                },
                step=epoch,
            )

            print(
                f"Epoch {epoch:02d}/{args.epochs} | "
                f"loss={loss:.4f} | "
                f"F1_content={f1_content:.4f} | "
                f"F1_mood={f1_mood:.4f} | "
                f"Avg_F1={avg_f1:.4f}"
            )

            if avg_f1 > best_f1:
                best_f1 = avg_f1
                torch.save(model.state_dict(), checkpoint_path)
                mlflow.pytorch.log_model(model, f"model_phase{args.phase}")  # type: ignore
                print(f"    Checkpoint saved (avg_F1={avg_f1:.4f})")

    print(f"\nPhase {args.phase} done. Best avg F1 = {best_f1:.4f}")
    return checkpoint_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", type=int, default=1, choices=[1, 2])
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--img_dir", type=str, default="data/raw")
    parser.add_argument("--train_csv", type=str, default="data/splits/train.csv")
    parser.add_argument("--val_csv", type=str, default="data/splits/val.csv")
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--save_dir", type=str, default="checkpoints")
    parser.add_argument("--device", type=str, default="auto")
    main(parser.parse_args())
