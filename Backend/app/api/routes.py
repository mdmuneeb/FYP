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

    print("📂 File:", file.filename)
    print("🧾 Type:", file.content_type)
    print("📦 Size (bytes):", len(contents))

    # =========================
    # Try OpenCV first (JPG/PNG)
    # =========================
    nparr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    # =========================
    # Fallback → HEIC support
    # =========================
    if img is None:
        print("⚠️ OpenCV failed → trying PIL (HEIC support)")

        try:
            from PIL import Image
            import pillow_heif
            import io

            pillow_heif.register_heif_opener()

            image = Image.open(io.BytesIO(contents)).convert("RGB")
            img = np.array(image)

        except Exception as e:
            raise ValueError(f"Unsupported image format: {str(e)}")

    # =========================
    # Final safety check
    # =========================
    if img is None:
        raise ValueError("Image decoding failed completely")

    print("✅ Image shape:", img.shape)

    # =========================
    # Run pipeline
    # =========================
    results = pipeline.predict(img)

    # =========================
    # Summary
    # =========================
    labels = [p["class"] for p in results]
    counts = Counter(labels)

    return {
        "num_grains": len(results),
        "summary": dict(counts),
        "predictions": results
    }