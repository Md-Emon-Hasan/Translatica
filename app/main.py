"""
FastAPI Application Entry Point
"""

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from app.api.routes import router
from app.core.config import settings
from app.core.model import model_manager
from app.utils.logger import get_logger

logger = get_logger("main")

templates = Jinja2Templates(directory=str(settings.TEMPLATES_DIR))


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan - load model on startup, cleanup on shutdown."""
    logger.info("Starting application...")
    logger.info("Loading ML model...")
    model_manager.load()
    logger.info("Model loaded successfully!")
    yield
    logger.info("Shutting down application...")
    model_manager.cleanup()
    logger.info("Cleanup complete.")


app = FastAPI(
    title=settings.APP_NAME,
    description="AI-powered English to Spanish translation API using LoRA fine-tuned transformer",
    version="1.0.0",
    lifespan=lifespan,
)

# Mount static files FIRST (before router, so 'static' route is available)
app.mount("/static", StaticFiles(directory=str(settings.STATIC_DIR)), name="static")

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include API router
app.include_router(router)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "app.main:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=settings.DEBUG,
    )
