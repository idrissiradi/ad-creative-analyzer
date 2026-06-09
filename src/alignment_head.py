import os

import mlflow
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, random_split

IMAGE_DIM = 1280
TEXT_DIM = 384
SHARED_DIM = 512
TEMPERATURE = 0.07


class ProjectionMLP(nn.Module):
    """Simple 2-layer MLP with BatchNorm, GELU, and Dropout."""

    def __init__(
        self, input_dim: int, hidden_dim: int, output_dim: int, dropout: float = 0.3
    ):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.net(x), dim=-1)


class AlignmentHead(nn.Module):
    """MLP-based head to align image and text vectors in a shared space."""

    def __init__(self, image_dim=IMAGE_DIM, text_dim=TEXT_DIM, shared_dim=SHARED_DIM):
        super().__init__()
        self.image_proj = ProjectionMLP(image_dim, image_dim // 2, shared_dim)
        self.text_proj = ProjectionMLP(text_dim, shared_dim, shared_dim)
        self.temperature = nn.Parameter(torch.tensor(TEMPERATURE), requires_grad=True)

    def forward(self, image_vectors: torch.Tensor, text_vectors: torch.Tensor) -> dict:
        temp = self.temperature.clamp(0.01, 0.5)
        image_embed = self.image_proj(image_vectors)  # [B, 512]
        text_embed = self.text_proj(text_vectors)  # [B, 512]

        scores = (image_embed * text_embed).sum(dim=-1)  # [B]
        logits_img = (image_embed @ text_embed.T) / temp  # [B, B]
        logits_txt = logits_img.T

        return {
            "image_embed": image_embed,
            "text_embed": text_embed,
            "scores": scores,
            "logits_img": logits_img,
            "logits_txt": logits_txt,
        }

    def score(self, image_vector: torch.Tensor, text_vector: torch.Tensor) -> float:
        """Inference: score a single pair. Returns float in [0, 100]."""
        self.eval()
        with torch.no_grad():
            img = image_vector.unsqueeze(0) if image_vector.dim() == 1 else image_vector
            txt = text_vector.unsqueeze(0) if text_vector.dim() == 1 else text_vector
            cos = (self.image_proj(img) * self.text_proj(txt)).sum(dim=-1).item()
        return round((cos + 1) / 2 * 100, 1)


class InfoNCELoss(nn.Module):
    """Computes InfoNCE loss for a batch of image-text pairs."""

    def forward(
        self, logits_img: torch.Tensor, logits_txt: torch.Tensor
    ) -> torch.Tensor:
        B = logits_img.shape[0]
        labels = torch.arange(B, device=logits_img.device)
        loss_i2t = F.cross_entropy(logits_img, labels)
        loss_t2i = F.cross_entropy(logits_txt, labels)
        return (loss_i2t + loss_t2i) / 2


class AlignmentPairDataset(Dataset):
    """Dataset for image-text pairs with similarity labels."""

    def __init__(
        self,
        image_vectors: torch.Tensor,
        text_vectors: torch.Tensor,
    ):
        self.image_vectors = image_vectors
        self.text_vectors = text_vectors

    def __len__(self):
        return len(self.image_vectors)

    def __getitem__(self, idx):
        return {
            "image_vector": self.image_vectors[idx],
            "text_vector": self.text_vectors[idx],
        }


def train_alignment_head(
    image_vectors: torch.Tensor,
    text_vectors: torch.Tensor,
    epochs: int = 40,
    batch_size: int = 32,
    lr: float = 1e-3,
    device: str = "cpu",
    save_path: str = "checkpoints/alignment_head.pt",
) -> AlignmentHead:
    """Train and return best AlignmentHead."""

    dataset = AlignmentPairDataset(image_vectors, text_vectors)
    n_val = max(int(0.2 * len(dataset)), 1)
    n_train = len(dataset) - n_val

    train_ds, val_ds = random_split(
        dataset, [n_train, n_val], generator=torch.Generator().manual_seed(42)
    )

    train_dl = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, drop_last=True, num_workers=2
    )
    val_dl = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=2)

    model = AlignmentHead().to(device)
    criterion = InfoNCELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    best_score = float("inf")
    best_state = None

    with mlflow.start_run(run_name="alignment_head_training"):
        mlflow.log_params(
            {
                "epochs": epochs,
                "batch_size": batch_size,
                "lr": lr,
                "shared_dim": SHARED_DIM,
                "temperature_init": TEMPERATURE,
                "n_train": n_train,
                "n_val": n_val,
            }
        )

        for epoch in range(1, epochs + 1):
            # Train
            model.train()
            epoch_loss = 0.0
            for batch in train_dl:
                img = batch["image_vector"].to(device)
                txt = batch["text_vector"].to(device)

                optimizer.zero_grad()
                out = model(img, txt)

                loss = criterion(out["logits_img"], out["logits_txt"])
                loss.backward()

                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

                epoch_loss += loss.item()
            scheduler.step()

            # Validate
            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for batch in val_dl:
                    img = batch["image_vector"].to(device)
                    txt = batch["text_vector"].to(device)

                    out = model(img, txt)
                    val_loss += criterion(out["logits_img"], out["logits_txt"]).item()

            avg_loss = epoch_loss / len(train_dl)
            avg_val_loss = val_loss / len(val_dl)

            mlflow.log_metrics(
                {"train_loss": avg_loss, "val_loss": avg_val_loss}, step=epoch
            )
            print(
                f"Epoch {epoch:02d} | train_loss={avg_loss:.4f} | val_loss={avg_val_loss:.4f}"
            )

            if avg_val_loss < best_score:
                best_score = avg_val_loss
                best_state = {k: v.clone() for k, v in model.state_dict().items()}
                print(f" Best checkpoint (val_loss={avg_val_loss:.4f})")

    if best_state is None:
        raise RuntimeError("Training finished without producing a best checkpoint.")

    model.load_state_dict(best_state)

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save(model.state_dict(), save_path)

    print(f"\nDone. Best val_loss = {best_score:.4f}")

    print("checkpoint saved! to: checkpoints/alignment_head.pt")
    return model
