from fastapi import APIRouter, UploadFile, File
import numpy as np
import cv2
from collections import Counter

from ..core.pipeline import RicePipeline

router = APIRouter()

pipeline = RicePipeline()


@router.post("/predict")
async def predict(file: UploadFile = File(...)):

    # =========================
    # Validate file type
    # =========================
    if not file.content_type.startswith("image/"):
        raise ValueError(f"Invalid file type: {file.content_type}")

    contents = await file.read()

    if not contents:
        raise ValueError("Empty file uploaded")

    nparr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    # =========================
    # Final safety check
    # =========================
    if img is None:
        raise ValueError("Image decoding failed completely")

    print("✅ Image shape:", img.shape)

    results = pipeline.predict(img)

    labels = [p["class"] for p in results]
    counts = Counter(labels)

    return {
        "num_grains": len(results),
        "summary": dict(counts),
        "predictions": results
    }

@router.get("/")
def root():
    return {
        "status": "API is running",
        "service": "Rice Grain Classification",
        "version": "1.0",
        "endpoints": {
            "predict": "/predict",
            "docs": "/docs"
        }
    }