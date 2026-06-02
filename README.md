# Ad Creative Analyzer + Caption Generator

A multimodal AI system that analyzes the alignment between advertising images and their captions, then generates improved copy optimized for the target platform.

---

## 🧠 What It Does

| Output            | Description                                              |
| ----------------- | -------------------------------------------------------- |
| Content Type      | Product Showcase / Lifestyle / Testimonial / Promotional |
| Mood              | Energetic / Calm / Professional / Playful                |
| Dominant Colors   | 3 hex codes via K-Means                                  |
| Alignment Score   | 0–100 cosine similarity                                  |
| Alignment Label   | Aligned / Neutral / Mismatched                           |
| Suggested Caption | Improved platform-specific copy                          |
| Keywords          | Hashtags / power words                                   |

---

## 🏗️ Architecture

```
[image] + [caption] + [platform]
        ↓
EfficientNet-B0 → content_type, mood, colors, image_vector [1280]
        ↓
MiniLM-L6-v2   → text_vector [384]
        ↓
Cosine similarity → alignment_score, label, explanation
        ↓
Flan-T5-base   → suggested_caption, keywords
        ↓
FastAPI → React frontend
```

---

## 📊 Targets

| Metric              | Target |
| ------------------- | ------ |
| Content Type F1     | > 0.78 |
| Mood F1             | > 0.75 |
| Alignment Precision | > 0.80 |
| Caption BLEU        | > 0.25 |
| Latency (CPU)       | < 3s   |

---

## 📦 Tech Stack

PyTorch · EfficientNet-B0 · HuggingFace Transformers · Sentence-Transformers  
scikit-learn · Albumentations · MLflow · FastAPI · React · Docker