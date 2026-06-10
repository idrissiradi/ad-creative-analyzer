"""
inference.py — Full inference pipeline.
Updated to use AlignmentHead projection head.
"""

import time
from pathlib import Path

import torch
from PIL import Image
from torchvision import transforms

from src.alignment import compute_alignment
from src.caption_generator import CaptionGenerator
from src.colors import extract_dominant_colors
from src.image_model import AdImageModel
from src.text_encoder import TextEncoder

CONTENT_TYPES = ["Product Showcase", "Lifestyle", "Promotional"]
MOODS = ["Energetic", "Calm", "Professional", "Playful"]

DEVICE = "cpu"

_transform = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)


class InferencePipeline:
    def __init__(
        self,
        image_checkpoint: str = "checkpoints/best_phase2.pt",
        alignment_checkpoint: str = "checkpoints/alignment_head.pt",
    ):

        print("Loading EfficientNet-B0...")
        image_checkpoint_path = Path(image_checkpoint)
        if not image_checkpoint_path.exists():
            raise FileNotFoundError(f"Image checkpoint not found: {image_checkpoint}")

        self.image_model = AdImageModel(freeze_encoder=False, pretrained=False)
        self.image_model.load_state_dict(
            torch.load(image_checkpoint_path, map_location=DEVICE, weights_only=True)
        )
        self.image_model.eval()

        print("Loading MiniLM text encoder...")
        self.text_encoder = TextEncoder(device=DEVICE)

        alignment_exists = Path(alignment_checkpoint).exists()
        print(f"AlignmentHead: {'loaded' if alignment_exists else 'not trained yet'}")

        print("Loading Flan-T5...")
        self.caption_gen = CaptionGenerator(device=DEVICE)
        self.alignment_checkpoint = alignment_checkpoint
        print("Ready.")

    def analyze(self, image: Image.Image, caption: str, platform: str) -> dict:
        t0 = time.perf_counter()

        img_t = _transform(image).unsqueeze(0)  # type: ignore
        with torch.no_grad():
            image_vector = self.image_model.encode_image(img_t)
            logits_content = self.image_model.head_content(image_vector)
            logits_mood = self.image_model.head_mood(image_vector)

        image_vector = image_vector.squeeze(0)
        content_type_id = logits_content.argmax(-1).item()
        mood_id = logits_mood.argmax(-1).item()
        content_type = CONTENT_TYPES[content_type_id]
        mood = MOODS[mood_id]
        ct_conf = torch.softmax(logits_content, -1).max().item()
        mood_conf = torch.softmax(logits_mood, -1).max().item()

        dominant_colors = extract_dominant_colors(image, k=3)
        text_vector = self.text_encoder.encode(caption)

        alignment = compute_alignment(
            image_vector,
            text_vector,
            checkpoint=self.alignment_checkpoint,
        )

        caption_out = self.caption_gen.generate(content_type, mood, platform, caption)

        return {
            "content_type": content_type,
            "content_type_confidence": round(ct_conf, 3),
            "mood": mood,
            "mood_confidence": round(mood_conf, 3),
            "dominant_colors": dominant_colors,
            "alignment_score": alignment.score,
            "alignment_label": alignment.label,
            "alignment_explanation": self._alignment_explanation(alignment.label),
            "suggested_caption": caption_out["suggested_caption"],
            "keyword_suggestions": caption_out["keyword_suggestions"],
            "alignment_method": alignment.method,
            "processing_time_ms": round((time.perf_counter() - t0) * 1000),
        }

    def _alignment_explanation(self, label: str) -> str:
        explanations = {
            "Aligned": "L'image et le texte portent un message coherent pour la plateforme choisie.",
            "Neutral": "L'image et le texte partagent une intention generale, mais le message peut etre renforce.",
            "Mismatched": "Le texte semble peu relie au signal visuel; une reformulation plus specifique aiderait.",
        }
        return explanations.get(
            label, "Score calcule a partir de la similarite image-texte."
        )
