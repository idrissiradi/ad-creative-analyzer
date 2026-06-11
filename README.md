# Ad Creative Analyzer + Caption Generator

A multimodal AI system that analyzes the alignment between advertising images and their captions, then generates improved copy optimized for the target platform.

Dataset: **1,610 images** from Google Open Images v7 + 200 image from Facebook Ads Library<br/>
Labels: [Content type + Mood] both auto-labeled by Qwen-VL  <br/>
Split: 70% train / 15% val / 15% test — stratified by content type
---


## 🚀 Setup

```bash
# 1. Clone & enter
git clone https://github.com/idrissiradi/ad-creative-analyzer.git
cd ad-creative-analyzer

# 2. Backend (Python 3.13, uses uv)
uv sync
# Train checkpoints are not in the repo (see .gitignore). To reproduce:
#   uv run python scripts/download_open_images.py --total 1500
#   uv run jupyter execute notebooks/auto_labeling_qwen.ipynb
#   uv run python scripts/prepare_splits.py
#   uv run jupyter execute notebooks/colab_training.ipynb     # recommended on Colab T4
#   uv run python -m src.evaluate                            # writes eval/final_results.json
uv run uvicorn api.main:app --reload --port 8000

# 3. Frontend
cd frontend
bun install
bun run dev    # http://localhost:5173
```

The frontend reads `VITE_API_URL` (defaults to `/api/analyze`). For local dev
with the FastAPI server on :8000, set `VITE_API_URL=http://localhost:8000/analyze`
in `frontend/.env.local`.

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
                    Cosine similarity 
                              ↓
              alignment_score, label
                              ↓
Flan-T5-base   ──► suggested_caption, keywords
                              ↓
              FastAPI ──► React + Vite + Tailwind
```


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
| Package manager   | uv                                          |
| Training          | Google Colab T4 GPU                         |

---