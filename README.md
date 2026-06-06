# Ad Creative Analyzer + Caption Generator

A multimodal AI system that analyzes the alignment between advertising images and their captions, then generates improved copy optimized for the target platform.

Dataset: **1,610 images** from Google Open Images v7
Labels: Content type + Mood — both auto-labeled by Qwen-VL  
Split: 70% train / 15% val / 15% test — stratified by content type
---

## 🧠 What It Does

| Output            | Description                                |
| ----------------- | ------------------------------------------ |
| Content Type      | Product Showcase / Lifestyle / Promotional |
| Mood              | Energetic / Calm / Professional / Playful  |
| Dominant Colors   | 3 hex codes via K-Means                    |
| Alignment Score   | 0–100 cosine similarity                    |
| Alignment Label   | Aligned / Neutral / Mismatched             |
| Suggested Caption | Improved platform-specific copy            |
| Keywords          | Hashtags / power words                     |

---

## 🏗️ Architecture

```
[image] + [caption] + [platform]
        ↓
EfficientNet-B0 ──► content_type, mood, colors
                └─► image_vector [1280]
                              ↓
                    Projection Head (MLP)
                              ↓ shared 512-dim space
MiniLM-L6-v2   ──► text_vector [384]
                    Projection Head (MLP)
                              ↓
                    Cosine similarity + VADER
                              ↓
              alignment_score, label, explanation
                              ↓
Flan-T5-base   ──► suggested_caption, keywords
                              ↓
              FastAPI ──► React + Vite + Tailwind
```

---

## 📊 Targets

| Metric               | Target |
| -------------------- | ------ |
| Content Type F1      | > 0.78 |
| Mood F1              | > 0.75 |
| Alignment Spearman ρ | > 0.55 |
| Latency (CPU)        | < 3s   |

---

## 📦 Tech Stack

| Layer             | Technology                                  |
| ----------------- | ------------------------------------------- |
| Image model       | PyTorch 2.2 · EfficientNet-B0 (torchvision) |
| Text encoder      | HuggingFace · all-MiniLM-L6-v2              |
| Caption generator | HuggingFace · Flan-T5-base (zero-shot)      |
| Color extraction  | scikit-learn · K-Means                      |
| API               | FastAPI · Uvicorn                           |
| Frontend          | React 18 · Vite · Tailwind CSS              |
| Containerization  | Docker · docker-compose                     |
| Package manager   | uv                                          |
| Training          | Google Colab T4 GPU                         |

---