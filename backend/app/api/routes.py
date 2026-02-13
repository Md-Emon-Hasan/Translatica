"""
API Routes for Translation Service
"""

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.database import get_db
from app.models.translation import Translation
from app.services.translation import TranslationService
from app.utils.logger import get_logger

router = APIRouter()
logger = get_logger("routes")
# Templates removed


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


# Index route removed as frontend is served separately


@router.post(
    "/translate",
    response_model=TranslationResponse,
    responses={400: {"model": ErrorResponse}, 500: {"model": ErrorResponse}},
)
async def translate(
    request: TranslationRequest,
    db: AsyncSession = Depends(get_db),
):
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
        translation_text = TranslationService.translate(text)
        logger.info("Translation completed successfully")

        # Save to database
        try:
            db_translation = Translation(
                source_text=text,
                translated_text=translation_text,
            )
            db.add(db_translation)
            await db.commit()
            await db.refresh(db_translation)
            logger.info(f"Translation saved to DB with ID: {db_translation.id}")
        except Exception as e:
            logger.error(f"Failed to save translation to DB: {e}")
            # We don't fail the request if saving to DB fails, just log it
            pass

        return TranslationResponse(translation=translation_text)

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
