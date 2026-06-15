# VisioMark -- Ad Creative Analyzer + Caption Generator

A multimodal AI system that analyzes the alignment between advertising images and their captions, then generates improved copy optimized for the target platform.

Dataset: **1,610 images** from Google Open Images v7 + 200 image from Facebook Ads Library<br/>
Labels: [Content type + Mood] both auto-labeled by Qwen-VL  <br/>
Split: 70% train / 15% val / 15% test — stratified by content type.

---
 
## Overview
 
| Output            | Description                                                |
| ----------------- | ---------------------------------------------------------- |
| Content Type      | Product Showcase / Lifestyle / Promotional (3 classes)     |
| Mood              | Energetic / Calm / Professional / Playful (4 classes)      |
| Dominant Colors   | 3 hex codes extracted via K-Means                          |
| Alignment Score   | 0–100, from a projection head trained with InfoNCE         |
| Alignment Label   | Aligned (70–100) / Neutral (40–69) / Mismatched (0–39)     |
| Suggested Caption | Improved, platform-specific copy (Flan-T5-base, zero-shot) |
| Keywords          | 4 relevant hashtags                                        |
 
**Dataset:** 1,632 images downloaded from Google Open Images v7 + Facebook Ads Library. After filtering corrupted files (15 removed), **1,617 images** retained.
**Labels:** Content type + mood, auto-annotated by Qwen2.5-VL-3B-Instruct (85% agreement with human judgment on a 10% validation sample).
**Split:** 70% train (1,131) / 15% val (243) / 15% test (243) — stratified by content type.
 
---


## Setup & Installation
 
### Prerequisites
- Python 3.13
- [uv](https://docs.astral.sh/uv/) (Python package manager)
- [Bun](https://bun.sh/) (frontend package manager)
- Git
  
### Clone the repository

```bash
git clone https://github.com/idrissiradi/ad-creative-analyzer.git
cd ad-creative-analyzer
```
 
### Backend setup
```bash
uv sync
```
 
Trained checkpoints are **not included** in the repository (see `.gitignore`). To reproduce them from scratch:
 
```bash
# 1. Download raw images (Open Images v7 + Facebook Ads Library)
# uv run python scripts/download_open_images.py --total 1632
 
# 2. Auto-label the dataset with Qwen2.5-VL-3B-Instruct (run on Colab T4)
# uv run jupyter execute notebooks/auto_labeling_qwen.ipynb
 
# 3. Build stratified train/val/test splits
# uv run python scripts/prepare_splits.py
 
# 4. Train all models — Phase 1, Phase 2, AlignmentHead (run on Colab T4)
# uv run jupyter execute notebooks/colab_training.ipynb
 
```
 
### Run the backend API
```bash
uv run uvicorn api.main:app --reload --port 8000
```
The API will be available at `http://localhost:8000`, with interactive docs at `http://localhost:8000/docs`.
 
### Frontend setup
```bash
cd frontend
bun install
bun run dev
```
The app will be available at `http://localhost:5173`.
 
The frontend reads the `VITE_API_URL` environment variable (defaults to `/api/analyze`). For local development against the FastAPI server on port 8000, create `frontend/.env.local`:
```
VITE_API_URL=http://localhost:8000/analyze
```

---
 
## 5. Usage
 
1. Open the frontend in your browser.
2. Upload an advertising image (JPEG/PNG/WebP, max 10 MB) via drag & drop.
3. Enter a candidate caption (max 500 characters).
4. Select the target platform (Instagram, Facebook, LinkedIn, or TikTok).
5. Click **Analyze**. After ~18–22 seconds, the dashboard displays:
   - Content type and mood, each with a confidence score
   - The 3 dominant colors (hex codes)
   - An alignment score (0–100) with label and explanation
   - The original caption alongside a Flan-T5-generated suggestion
   - 4 generated hashtags
  
---

## 3. Project Structure
 
```
ad-creative-analyzer/
├── api/
│   ├── __init__.py
│   ├── inference.py             # InferencePipeline: loads all models, runs end-to-end inference
│   └── main.py                  # FastAPI app: exposes /analyze endpoint, CORS, app lifecycle
├── src/
│   ├── image_model.py           # AdImageModel: EfficientNet-B0 + 3 classification heads
│   ├── text_encoder.py          # TextEncoder: MiniLM-L6-v2 sentence embeddings
│   ├── alignment_head.py        # ProjectionMLP, AlignmentHead, InfoNCELoss, AlignmentPairDataset, train_alignment_head
│   ├── alignment.py             # AlignmentResult, load_head, compute_alignment
│   ├── caption_generator.py     # CaptionGenerator: Flan-T5-base zero-shot caption + hashtags
│   ├── colors.py                # extract_dominant_colors: K-Means color extraction
│   ├── dataset.py               # AdImageDataset, get_transforms: PyTorch Dataset + image transforms
│   ├── platform_guides.py       # PLATFORM_GUIDES: tone/format rules per platform
│   ├── train.py                 # train_model, train_epoch, eval_epoch: Phase 1 / Phase 2 training loop
│   └── evaluate.py              # evaluate: test-set metrics, confusion matrices, report
├── scripts/
│   ├── download_open_images.py  # Downloads raw images from Open Images v7
│   └── prepare_splits.py        # Builds train/val/test CSVs (stratified split)
├── notebooks/
│   ├── auto_labeling_qwen.ipynb # Auto-annotation of the dataset with Qwen2.5-VL-3B-Instruct
│   ├── colab_training.ipynb     # End-to-end training (Phase 1, Phase 2, AlignmentHead, eval)
│   └── testing.ipynb            # Test function
├── data/
│   ├── raw/                     # Downloaded images (not in repo, see .gitignore)
│   ├── splits/                  # train.csv / val.csv / test.csv
│   ├── labeled/                 # labels.csv — auto-generated labels (Qwen2.5-VL)
│   ├── alignment_pairs/         # alignment_pairs.csv / alignment_pairs_valid.csv
│   └── eval/                    # final_results.json, confusion matrix images
├── checkpoints/                 # Trained model weights (not in repo, see .gitignore)
├── frontend/                    # React + Vite + Tailwind SPA
│   └── src/
│       └── components/          # UploadZone, ResultsPanel, AlignmentGauge
└── pyproject.toml               # Python dependencies (managed with uv)
```

---

## Architecture
 
```
[image] + [caption] + [target platform]
        ↓
EfficientNet-B0 ──► content_type, mood, dominant colors
                └─► image_vector [1280]
                              ↓
                    Projection Head (MLP, 1280 → 512)
                              ↓ shared 512-dim space
MiniLM-L6-v2   ──► text_vector [384]
                    Projection Head (MLP, 384 → 512)
                              ↓
                    Cosine similarity → InfoNCE-trained alignment
                              ↓
              alignment_score (0–100), alignment_label, explanation
                              ↓
Flan-T5-base   ──► suggested_caption, hashtags (zero-shot, platform-adapted)
                              ↓
              FastAPI (backend) ──► React + Vite + Tailwind (frontend)
```
 
Three-tier architecture:
- **Presentation layer (frontend):** React 18 SPA, Vite, Tailwind CSS.
- **Business logic layer (backend):** FastAPI + Uvicorn, orchestrating the ML pipeline and exposing a REST API (Swagger docs at `/docs`).
- **Data & models layer:** local PyTorch checkpoints, cached HuggingFace models (MiniLM, Flan-T5), MLflow experiment tracking (SQLite + Google Drive).

---

## Tech Stack
 
| Layer               | Technology                                           |
| ------------------- | ---------------------------------------------------- |
| Image model         | PyTorch  · EfficientNet-B0 (torchvision)             |
| Text encoder        | HuggingFace · sentence-transformers/all-MiniLM-L6-v2 |
| Caption generator   | HuggingFace · Flan-T5-base (zero-shot)               |
| Color extraction    | scikit-learn · K-Means                               |
| Experiment tracking | MLflow (SQLite + Google Drive)                       |
| API                 | FastAPI · Uvicorn                                    |
| Frontend            | React · Vite · Tailwind CSS                          |
| Package manager     | uv (Python) · Bun (frontend)                         |
| Training hardware   | Google Colab — Tesla T4 GPU (15.6 GB VRAM)           |
 
---