from fastapi import FastAPI
from .api.routes import router

app = FastAPI(
    title="Rice Grain Variety Classification API",
    version="1.0"
)

app.include_router(router)