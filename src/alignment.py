from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn.functional as F

from src.alignment_head import AlignmentHead

DEFAULT_CHECKPOINT = "checkpoints/alignment_head.pt"

THRESHOLD_HIGH = 70
THRESHOLD_LOW = 40


@dataclass
class AlignmentResult:
    score: float  # 0–100
    label: str  # Aligned / Neutral / Mismatched
    explanation: str  # filled by explanation.py
    method: str  # "projection_head" or "cosine_fallback" (Great for debugging!)


def load_head(checkpoint: str = DEFAULT_CHECKPOINT) -> AlignmentHead | None:
    """Load AlignmentHead. Returns None if checkpoint not found."""
    checkpoint_path = Path(checkpoint)
    if not checkpoint_path.exists():
        return None

    head = AlignmentHead()
    head.load_state_dict(
        torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    )
    head.eval()
    return head


def compute_alignment(
    image_vector: torch.Tensor,
    text_vector: torch.Tensor,
    checkpoint: str = DEFAULT_CHECKPOINT,
) -> AlignmentResult:
    """Score alignment between image and text vectors."""
    head = load_head(checkpoint)

    if head is not None:
        score = head.score(image_vector, text_vector)  # float [0, 100]
        method = "projection_head"
    else:
        # Fallback to cosine similarity if head not available.
        min_dim = min(image_vector.shape[-1], text_vector.shape[-1])
        img_norm = F.normalize(image_vector.float(), dim=-1)
        txt_norm = F.normalize(text_vector.float(), dim=-1)

        cos = torch.dot(img_norm[:min_dim], txt_norm[:min_dim]).item()
        score = round((cos + 1) / 2 * 100, 1)
        method = "cosine_fallback"

    if score >= THRESHOLD_HIGH:
        label = "Aligned"
    elif score >= THRESHOLD_LOW:
        label = "Neutral"
    else:
        label = "Mismatched"

    explanation = ""  # filled by explanation.py in inference pipeline

    return AlignmentResult(
        score=score,
        label=label,
        explanation=explanation,
        method=method,
    )
