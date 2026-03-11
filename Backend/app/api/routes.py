from fastapi import APIRouter, UploadFile, File
import numpy as np
import cv2
from collections import Counter

from ..core.pipeline import RicePipeline

router = APIRouter()

pipeline = RicePipeline()


@router.post("/predict")

async def predict(file: UploadFile = File(...)):

    contents = await file.read()

    nparr = np.frombuffer(contents, np.uint8)

    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    results = pipeline.predict(img)

    labels = [p["class"] for p in results]

    counts = Counter(labels)

    summary = {
        
    }

    return {
        "num_grains": len(results),
        "summary": dict(counts),
        "predictions": results
    }