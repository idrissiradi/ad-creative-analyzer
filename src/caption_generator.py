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

        Returns:
            dict: {"suggested_caption": str, "keyword_suggestions": list}
        """
        platform_guide = PLATFORM_GUIDES.get(platform, PLATFORM_GUIDES["default"])
        prompt = self._build_prompt(
            content_type, mood, platform, platform_guide, original_caption
        )

        inputs = self.tokenizer(
            prompt, return_tensors="pt", max_length=512, truncation=True
        ).to(self.device)

        with torch.no_grad():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                num_beams=4,
                early_stopping=True,
                no_repeat_ngram_size=3,
            )

        suggested = self.tokenizer.decode(output_ids[0], skip_special_tokens=True)
        keywords = self._extract_keywords(content_type, mood, platform)

        return {
            "suggested_caption": suggested,
            "keyword_suggestions": keywords,
        }

    def _build_prompt(self, content_type, mood, platform, guide, original):
        return (
            f"You are a digital marketing copywriter.\n"
            f"Ad type: {content_type}\n"
            f"Visual mood: {mood}\n"
            f"Target platform: {platform}\n"
            f"Platform style: {guide}\n"
            f'Original caption: "{original}"\n\n'
            f"Rewrite the caption to better match the visual mood and platform style. "
            f"Keep it concise and engaging."
        )

    def _extract_keywords(self, content_type, mood, platform) -> list[str]:
        """Rule-based keyword suggestions."""
        base = {
            "Product Showcase": ["#ProductLaunch", "#NewArrival"],
            "Lifestyle": ["#LifestyleGoals", "#Inspiration"],
            "Promotional": ["#LimitedOffer", "#DontMissOut"],
        }.get(content_type, [])

        mood_kw = {
            "Energetic": ["#GetStarted", "#TakeAction"],
            "Calm": ["#Peace", "#Mindful"],
            "Professional": ["#Leadership", "#Growth"],
            "Playful": ["#FunTime", "#JoinUs"],
        }.get(mood, [])

        return (base + mood_kw)[:4]
