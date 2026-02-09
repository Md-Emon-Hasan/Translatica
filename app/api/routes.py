"""
API Routes for Translation Service
"""

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel, Field

from app.core.config import settings
from app.services.translation import TranslationService
from app.utils.logger import get_logger

router = APIRouter()
logger = get_logger("routes")
templates = Jinja2Templates(directory=str(settings.TEMPLATES_DIR))


class TranslationRequest(BaseModel):
    """Request model for translation."""

    text: str = Field(
        ..., min_length=1, max_length=5000, description="Text to translate"
    )


class TranslationResponse(BaseModel):
    """Response model for translation."""

    translation: str = Field(..., description="Translated text")


class ErrorResponse(BaseModel):
    """Error response model."""

    error: str = Field(..., description="Error message")


class HealthResponse(BaseModel):
    """Health check response model."""

    status: str = Field(..., description="Service status")
    model_loaded: bool = Field(..., description="Whether the model is loaded")


@router.get("/", response_class=HTMLResponse, include_in_schema=False)
async def index(request: Request):
    """Serve the main translation page."""
    return templates.TemplateResponse(request, "index.html")


@router.post(
    "/translate",
    response_model=TranslationResponse,
    responses={400: {"model": ErrorResponse}, 500: {"model": ErrorResponse}},
)
async def translate(request: TranslationRequest):
    """
    Translate English text to Spanish.

    - **text**: English text to translate (1-5000 characters)

    Returns the Spanish translation.
    """
    try:
        text = request.text.strip()
        if not text:
            raise HTTPException(status_code=400, detail="Empty input")

        logger.info(f"Translating text of length {len(text)}")
        translation = TranslationService.translate(text)
        logger.info("Translation completed successfully")

        return TranslationResponse(translation=translation)

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Translation error: {e}")
        raise HTTPException(status_code=500, detail="Translation failed") from e


@router.get("/health", response_model=HealthResponse)
async def health_check():
    """
    Health check endpoint.

    Returns the service status and whether the model is loaded.
    """
    from app.core.model import model_manager

    return HealthResponse(
        status="healthy",
        model_loaded=model_manager.is_loaded,
    )
