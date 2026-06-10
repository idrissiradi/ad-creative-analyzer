import io
from contextlib import asynccontextmanager

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image

from api.inference import InferencePipeline

pipeline: InferencePipeline | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global pipeline
    pipeline = InferencePipeline()
    yield
    pipeline = None


app = FastAPI(
    title="Ad Creative Analyzer API",
    description="Multimodal AI system for digital marketing ad analysis",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Tighten in production
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
async def health():
    return {
        "status": "ok",
        "models": ["EfficientNet-B0", "MiniLM-L6-v2", "Flan-T5-base"],
    }


@app.post("/analyze")
async def analyze(
    image: UploadFile = File(...),
    caption: str = Form(...),
    platform: str = Form("default"),
):
    if pipeline is None:
        raise HTTPException(503, "Models not loaded")

    try:
        img_bytes = await image.read()
        pil_image = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    except Exception:
        raise HTTPException(400, "Invalid image file")

    result = pipeline.analyze(pil_image, caption, platform)
    return result
