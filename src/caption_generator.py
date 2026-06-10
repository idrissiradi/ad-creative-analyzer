"""
caption_generator.py — Flan-T5-base caption generator.
Optimized for direct, task-oriented instructions that Flan-T5 understands.
"""

import torch
from transformers import AutoTokenizer, T5ForConditionalGeneration

from src.platform_guides import PLATFORM_GUIDES

MODEL_NAME = "google/flan-t5-base"


class CaptionGenerator:
    def __init__(self, model_name: str = MODEL_NAME, device: str = "cpu"):
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = T5ForConditionalGeneration.from_pretrained(model_name)
        self.model.to(device)
        self.model.eval()

    def generate(
        self,
        content_type: str,
        mood: str,
        platform: str,
        original_caption: str,
        max_new_tokens: int = 80,
    ) -> dict:
        """
        Generate an improved caption for the given ad context.
        """
        platform_guide = PLATFORM_GUIDES.get(platform, PLATFORM_GUIDES["default"])
        prompt = self._build_prompt(
            content_type, mood, platform, platform_guide, original_caption
        )

        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)

        with torch.no_grad():
            output_ids = self.model.generate(**inputs)

        suggested = self.tokenizer.decode(
            output_ids[0], skip_special_tokens=True
        ).strip()
        keywords = self._extract_keywords(content_type, mood, platform)

        return {
            "suggested_caption": suggested,
            "keyword_suggestions": keywords,
        }

    def _build_prompt(
        self, content_type: str, mood: str, platform: str, guide: str, original: str
    ) -> str:

        return (
            f"Given the original caption: {original}, "
            f"write a new version for {platform} with a {mood} mood"
        )

    def _extract_keywords(
        self, content_type: str, mood: str, platform: str
    ) -> list[str]:
        """Rule-based keyword suggestions."""
        base = {
            "Product Showcase": ["#ProductLaunch", "#NewArrival"],
            "Lifestyle": ["#LifestyleGoals", "#Inspiration"],
            "Testimonial": ["#CustomerStory", "#TrueReview"],
            "Promotional": ["#LimitedOffer", "#DontMissOut"],
        }.get(content_type, [])

        mood_kw = {
            "Energetic": ["#GetStarted", "#TakeAction"],
            "Calm": ["#Peace", "#Mindful"],
            "Professional": ["#Leadership", "#Growth"],
            "Playful": ["#FunTime", "#JoinUs"],
        }.get(mood, [])

        return (base + mood_kw)[:4]
