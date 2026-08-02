"""
API Routes for Translation Service
"""

import json
import time
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, HTTPException, Query, Request
from pydantic import BaseModel, Field, field_validator
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession
from starlette.concurrency import run_in_threadpool

from app.core.cache import cache_stats
from app.core.config import settings
from app.core.database import get_db
from app.core.model import build_generation_kwargs
from app.core.rate_limit import rate_limit
from app.models.translation import Translation
from app.services.translation import TranslationService
from app.utils.logger import get_logger
from app.utils.validation import validate_input_text

router = APIRouter()
logger = get_logger("routes")
# Templates removed


class TranslationRequest(BaseModel):
    """Request model for translation."""

    text: str = Field(
        ..., min_length=1, max_length=5000, description="Text to translate"
    )

    @field_validator("text")
    @classmethod
    def check_text(cls, value: str) -> str:
        """Validate the input text."""
        return validate_input_text(value)


class TranslationResponse(BaseModel):
    """Response model for translation."""

    translation: str = Field(..., description="Translated text")
    chunk_count: Optional[int] = Field(
        None, description="How many chunks the input was split into"
    )
    cached: Optional[bool] = Field(
        None, description="Whether the result came from the cache"
    )
    duration_ms: Optional[float] = Field(
        None, description="Server-side processing time in milliseconds"
    )
    warning: Optional[str] = Field(
        None, description="Non-fatal warning (e.g. input looks like Spanish already)"
    )


class BatchTranslationRequest(BaseModel):
    """Request model for batch translation."""

    texts: List[str] = Field(..., description="Texts to translate")

    @field_validator("texts")
    @classmethod
    def check_texts(cls, value: List[str]) -> List[str]:
        """Validate batch size and total length."""
        if not value:
            raise ValueError("The batch must contain at least one text.")

        if len(value) > settings.MAX_BATCH_SIZE:
            raise ValueError(
                f"A batch may contain at most {settings.MAX_BATCH_SIZE} items "
                f"(received {len(value)})."
            )

        total_chars = sum(len(text or "") for text in value)
        if total_chars > settings.MAX_BATCH_CHARS:
            raise ValueError(
                f"A batch may contain at most {settings.MAX_BATCH_CHARS} characters "
                f"in total (received {total_chars})."
            )

        return value


class BatchItemResponse(BaseModel):
    """Per-item result inside a batch response."""

    index: int = Field(..., description="Position in the submitted batch")
    success: bool = Field(..., description="Whether this item was translated")
    translation: Optional[str] = Field(None, description="Translated text")
    error: Optional[str] = Field(None, description="Why this item failed")
    chunk_count: Optional[int] = Field(None, description="Chunks used for this item")
    cached: Optional[bool] = Field(None, description="Served from cache")
    warning: Optional[str] = Field(None, description="Non-fatal warning")


class BatchTranslationResponse(BaseModel):
    """Response model for batch translation."""

    count: int = Field(..., description="Number of items in the batch")
    succeeded: int = Field(..., description="Number of successfully translated items")
    failed: int = Field(..., description="Number of failed items")
    duration_ms: float = Field(..., description="Server-side processing time (ms)")
    results: List[BatchItemResponse] = Field(
        ..., description="Results in the submitted order"
    )


class HistoryItem(BaseModel):
    """A single logged translation."""

    id: int = Field(..., description="Row id")
    source_text: str = Field(..., description="Source text (possibly truncated)")
    translated_text: str = Field(
        ..., description="Translated text (possibly truncated)"
    )
    source_truncated: bool = Field(..., description="Source text was truncated")
    translated_truncated: bool = Field(..., description="Translated text was truncated")
    timestamp: datetime = Field(..., description="When the translation was logged")


class HistoryResponse(BaseModel):
    """Paginated translation history."""

    total: int = Field(..., description="Total rows matching the filters")
    limit: int = Field(..., description="Page size")
    offset: int = Field(..., description="Rows skipped")
    items: List[HistoryItem] = Field(..., description="Newest first by default")


class DailyCount(BaseModel):
    """One day of the daily translation series."""

    date: str = Field(..., description="Day in YYYY-MM-DD form")
    count: int = Field(..., description="Translations logged that day")


class StatsResponse(BaseModel):
    """Aggregate statistics over the translation log."""

    total_translations: int = Field(..., description="All logged translations")
    translations_today: int = Field(..., description="Logged today (UTC)")
    translations_last_7_days: int = Field(..., description="Logged in the last 7 days")
    translations_last_30_days: int = Field(
        ..., description="Logged in the last 30 days"
    )
    avg_source_chars: float = Field(..., description="Average source length")
    avg_translated_chars: float = Field(..., description="Average translation length")
    daily_counts: List[DailyCount] = Field(
        ..., description="Daily counts for the last 30 days"
    )
    cache: Dict[str, Any] = Field(..., description="Current cache statistics")


class ModelInfoResponse(BaseModel):
    """Static model and inference metadata."""

    base_model: Optional[str] = Field(None, description="Base model checkpoint")
    peft_type: Optional[str] = Field(None, description="Adapter type (e.g. LORA)")
    task_type: Optional[str] = Field(None, description="PEFT task type")
    lora_rank: Optional[int] = Field(None, description="LoRA rank (r)")
    lora_alpha: Optional[int] = Field(None, description="LoRA alpha")
    lora_dropout: Optional[float] = Field(None, description="LoRA dropout")
    target_modules: List[str] = Field(
        default_factory=list, description="Modules LoRA was applied to"
    )
    trainable_parameters: Optional[int] = Field(
        None, description="Adapter parameter count, when the model is loaded"
    )
    generation: Dict[str, Any] = Field(
        ..., description="Generation parameters currently in effect"
    )
    limits: Dict[str, Any] = Field(..., description="Token and character limits")
    model_loaded: bool = Field(..., description="Whether the model is loaded")


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
    dependencies=[Depends(rate_limit("RATE_LIMIT_TRANSLATE", "translate"))],
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
    from app.core.model import model_manager

    if not model_manager.is_loaded:
        raise HTTPException(
            status_code=503,
            detail="Model is loading (takes 30-60s). Please refresh.",
        )

    try:
        text = request.text.strip()
        if not text:
            raise HTTPException(status_code=400, detail="Empty input")

        logger.info(f"Translating text of length {len(text)}")
        result = await run_in_threadpool(TranslationService.translate_document, text)
        translation_text = result.translation
        logger.info(
            f"Translation completed successfully "
            f"(chunks={result.chunk_count}, cached={result.cached})"
        )

        # Save to database
        try:
            db_translation = Translation(
                source_text=text,
                translated_text=translation_text,
            )
            db.add(db_translation)
            await db.commit()
            await db.refresh(db_translation)
            logger.info(f"Saved to DB: {db_translation.id}")
        except Exception as e:
            logger.error(f"Failed to save translation to DB: {e}")
            # We don't fail the request if saving to DB fails, just log it
            pass

        return TranslationResponse(
            translation=translation_text,
            chunk_count=result.chunk_count,
            cached=result.cached,
            duration_ms=result.duration_ms,
            warning=result.warning,
        )

    except HTTPException:
        raise
    except ValueError as e:
        logger.warning(f"Invalid translation input: {e}")
        raise HTTPException(status_code=422, detail=str(e)) from e
    except Exception as e:
        logger.error(f"Translation error: {e}")
        raise HTTPException(status_code=500, detail="Translation error") from e


@router.post(
    "/translate/batch",
    response_model=BatchTranslationResponse,
    responses={500: {"model": ErrorResponse}},
    dependencies=[Depends(rate_limit("RATE_LIMIT_BATCH", "translate_batch"))],
)
async def translate_batch(
    request: Request,
    payload: BatchTranslationRequest,
    db: AsyncSession = Depends(get_db),
):
    """
    Translate several texts in one request.

    - **texts**: list of English texts (max MAX_BATCH_SIZE items)

    Returns the translations in the submitted order.
    """
    from app.core.model import model_manager

    if not model_manager.is_loaded:
        raise HTTPException(
            status_code=503,
            detail="Model is loading (takes 30-60s). Please refresh.",
        )

    started = time.perf_counter()

    try:
        results = await run_in_threadpool(
            TranslationService.translate_batch, payload.texts
        )
    except Exception as e:
        logger.error(f"Batch translation error: {e}")
        raise HTTPException(status_code=500, detail="Translation error") from e

    duration_ms = round((time.perf_counter() - started) * 1000, 2)

    # Save to database
    try:
        for item in results:
            if item.success and item.translation:
                db.add(
                    Translation(
                        source_text=payload.texts[item.index].strip(),
                        translated_text=item.translation,
                    )
                )
        await db.commit()
    except Exception as e:
        logger.error(f"Failed to save batch translations to DB: {e}")

    succeeded = sum(1 for item in results if item.success)
    logger.info(
        f"Batch of {len(results)} translated in {duration_ms}ms "
        f"({succeeded} succeeded)"
    )

    return BatchTranslationResponse(
        count=len(results),
        succeeded=succeeded,
        failed=len(results) - succeeded,
        duration_ms=duration_ms,
        results=[BatchItemResponse(**vars(item)) for item in results],
    )


def _truncate(text: str) -> tuple:
    """Shorten text for list responses."""
    limit = settings.HISTORY_TEXT_PREVIEW_CHARS
    if text is not None and len(text) > limit:
        return text[:limit], True
    return text, False


@router.get(
    "/history",
    response_model=HistoryResponse,
    dependencies=[Depends(rate_limit("RATE_LIMIT_HISTORY", "history"))],
)
async def get_history(
    request: Request,
    limit: int = Query(
        default=settings.HISTORY_DEFAULT_LIMIT,
        ge=1,
        le=settings.HISTORY_MAX_LIMIT,
        description="Page size",
    ),
    offset: int = Query(default=0, ge=0, description="Rows to skip"),
    search: Optional[str] = Query(
        default=None, description="Substring search on the source text"
    ),
    start_date: Optional[datetime] = Query(
        default=None, description="Only rows at or after this timestamp"
    ),
    end_date: Optional[datetime] = Query(
        default=None, description="Only rows at or before this timestamp"
    ),
    newest_first: bool = Query(default=True, description="Sort newest first"),
    db: AsyncSession = Depends(get_db),
):
    """
    Paginated translation history.

    - **limit** / **offset**: page size and position
    - **search**: substring match on the source text
    - **start_date** / **end_date**: timestamp range

    Returns a page of translations, newest first by default.
    """
    filters = []
    if search:
        filters.append(Translation.source_text.ilike(f"%{search}%"))
    if start_date is not None:
        filters.append(Translation.timestamp >= start_date)
    if end_date is not None:
        filters.append(Translation.timestamp <= end_date)

    count_stmt = select(func.count()).select_from(Translation)
    rows_stmt = select(Translation)
    for condition in filters:
        count_stmt = count_stmt.where(condition)
        rows_stmt = rows_stmt.where(condition)

    order = (
        Translation.timestamp.desc() if newest_first else Translation.timestamp.asc()
    )
    rows_stmt = rows_stmt.order_by(order, Translation.id.desc()).limit(limit)
    rows_stmt = rows_stmt.offset(offset)

    total = (await db.execute(count_stmt)).scalar() or 0
    rows = (await db.execute(rows_stmt)).scalars().all()

    items = []
    for row in rows:
        source_text, source_truncated = _truncate(row.source_text)
        translated_text, translated_truncated = _truncate(row.translated_text)
        items.append(
            HistoryItem(
                id=row.id,
                source_text=source_text,
                translated_text=translated_text,
                source_truncated=source_truncated,
                translated_truncated=translated_truncated,
                timestamp=row.timestamp,
            )
        )

    return HistoryResponse(total=total, limit=limit, offset=offset, items=items)


@router.get(
    "/stats",
    response_model=StatsResponse,
    dependencies=[Depends(rate_limit("RATE_LIMIT_STATS", "stats"))],
)
async def get_stats(
    request: Request,
    db: AsyncSession = Depends(get_db),
):
    """
    Aggregate statistics over the translation log.

    Returns counts, averages, a 30-day daily series and cache statistics.
    """
    now = datetime.utcnow()
    today = now.replace(hour=0, minute=0, second=0, microsecond=0)

    async def count_since(moment: Optional[datetime]) -> int:
        stmt = select(func.count()).select_from(Translation)
        if moment is not None:
            stmt = stmt.where(Translation.timestamp >= moment)
        return (await db.execute(stmt)).scalar() or 0

    total = await count_since(None)
    today_count = await count_since(today)
    last_7 = await count_since(now - timedelta(days=7))
    last_30 = await count_since(now - timedelta(days=30))

    averages = (
        await db.execute(
            select(
                func.avg(func.length(Translation.source_text)),
                func.avg(func.length(Translation.translated_text)),
            )
        )
    ).one()
    avg_source = round(float(averages[0] or 0.0), 2)
    avg_translated = round(float(averages[1] or 0.0), 2)

    day = func.date(Translation.timestamp)
    daily_rows = (
        await db.execute(
            select(day.label("day"), func.count().label("count"))
            .where(Translation.timestamp >= now - timedelta(days=30))
            .group_by(day)
            .order_by(day)
        )
    ).all()
    daily_counts = [
        DailyCount(date=str(row.day), count=int(row.count)) for row in daily_rows
    ]

    return StatsResponse(
        total_translations=total,
        translations_today=today_count,
        translations_last_7_days=last_7,
        translations_last_30_days=last_30,
        avg_source_chars=avg_source,
        avg_translated_chars=avg_translated,
        daily_counts=daily_counts,
        cache=cache_stats(),
    )


def _read_adapter_config() -> Dict[str, Any]:
    """Read the LoRA adapter config from disk."""
    config_path = settings.MODEL_PATH / "adapter_config.json"
    try:
        with open(config_path, "r", encoding="utf-8") as handle:
            return json.load(handle)
    except Exception as e:
        logger.warning(f"Could not read adapter config at {config_path}: {e}")
        return {}


def _count_adapter_parameters() -> Optional[int]:
    """Count the LoRA parameters of the loaded model."""
    from app.core.model import model_manager

    if not model_manager.is_loaded:
        return None

    try:
        return int(
            sum(
                param.numel()
                for name, param in model_manager.model.named_parameters()
                if "lora_" in name
            )
        )
    except Exception as e:  # pragma: no cover - defensive
        logger.warning(f"Could not count adapter parameters: {e}")
        return None


@router.get("/model-info", response_model=ModelInfoResponse)
async def get_model_info():
    """
    Model and inference metadata.

    Returns the adapter settings, generation config and limits in effect.
    """
    from app.core.model import model_manager

    adapter = _read_adapter_config()

    return ModelInfoResponse(
        base_model=adapter.get("base_model_name_or_path"),
        peft_type=adapter.get("peft_type"),
        task_type=adapter.get("task_type"),
        lora_rank=adapter.get("r"),
        lora_alpha=adapter.get("lora_alpha"),
        lora_dropout=adapter.get("lora_dropout"),
        target_modules=sorted(adapter.get("target_modules") or []),
        trainable_parameters=_count_adapter_parameters(),
        generation=build_generation_kwargs(),
        limits={
            "max_input_chars": settings.MAX_INPUT_CHARS,
            "min_input_chars": settings.MIN_INPUT_CHARS,
            "max_input_tokens": settings.MAX_INPUT_TOKENS,
            "chunk_token_budget": settings.CHUNK_TOKEN_BUDGET,
            "max_output_length": settings.MAX_OUTPUT_LENGTH,
            "max_batch_size": settings.MAX_BATCH_SIZE,
            "max_batch_chars": settings.MAX_BATCH_CHARS,
        },
        model_loaded=model_manager.is_loaded,
    )


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
