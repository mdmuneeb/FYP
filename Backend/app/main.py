from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from .api.routes import router

app = FastAPI(
    title="Rice Grain Variety Classification API",
    version="1.0"
)

app.mount("/grain_crops", StaticFiles(directory="grain_crops"), name="grain_crops")

app.include_router(router)