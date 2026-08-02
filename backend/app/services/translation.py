"""
Translation Service
"""

import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from app.core.cache import cache_get, cache_set, make_cache_key
from app.core.model import model_manager
from app.utils.chunking import ChunkPlan, assemble, get_token_counter, plan_chunks
from app.utils.logger import get_logger
from app.utils.validation import language_warning, validate_input_text

logger = get_logger("translation_service")


@dataclass
class TranslationResult:
    """Outcome of translating a single document."""

    translation: str
    chunk_count: int
    cached: bool
    duration_ms: float
    warning: Optional[str] = None


@dataclass
class BatchItemResult:
    """Outcome of a single item inside a batch request."""

    index: int
    success: bool
    translation: Optional[str] = None
    error: Optional[str] = None
    chunk_count: Optional[int] = None
    cached: Optional[bool] = None
    warning: Optional[str] = None


class TranslationService:
    """Service layer for translation operations."""

    @staticmethod
    def validate_input(text: str) -> str:
        """
        Validate and clean input text.

        Args:
            text: Input text to validate

        Returns:
            Cleaned text

        Raises:
            ValueError: If text is empty or invalid
        """
        if text is None:
            raise ValueError("Text cannot be None")

        cleaned = text.strip()
        if not cleaned:
            raise ValueError("Text cannot be empty")

        return cleaned

    @staticmethod
    def translate(text: str) -> str:
        """
        Translate English text to Spanish.

        Args:
            text: English text to translate

        Returns:
            Spanish translation

        Raises:
            ValueError: If input is invalid
            RuntimeError: If model is not loaded
        """
        cleaned_text = TranslationService.validate_input(text)

        logger.info(f"Translating text of length {len(cleaned_text)}")

        translation = model_manager.translate(cleaned_text)

        logger.info(f"Translation completed, output length {len(translation)}")

        return translation

    @staticmethod
    def _translate_chunks(chunks: List[str]) -> List[str]:
        """
        Translate chunks, sending only the cache misses to the model.

        Args:
            chunks: Chunks to translate

        Returns:
            Translated chunks in the same order
        """
        if not chunks:
            return []

        translated: List[Optional[str]] = [None] * len(chunks)
        pending: Dict[str, List[int]] = {}

        for position, chunk in enumerate(chunks):
            key = make_cache_key(chunk, prefix="chunk")
            hit = cache_get(key)
            if hit is not None:
                translated[position] = hit
            else:
                pending.setdefault(key, []).append(position)

        if pending:
            keys = list(pending)
            batch = [chunks[pending[key][0]] for key in keys]
            logger.info(
                f"Sending {len(batch)} uncached chunk(s) of {len(chunks)} to the model"
            )
            outputs = model_manager.generate_texts(batch)

            for key, output in zip(keys, outputs):
                cache_set(key, output)
                for position in pending[key]:
                    translated[position] = output

        return [text or "" for text in translated]

    @staticmethod
    def _plan(text: str) -> ChunkPlan:
        """Build the chunk plan for one document."""
        return plan_chunks(text, count_tokens=get_token_counter())

    @staticmethod
    def translate_document(text: str) -> TranslationResult:
        """
        Translate a long, multi-paragraph document.

        Args:
            text: English text to translate

        Returns:
            TranslationResult for the document

        Raises:
            ValueError: If the input fails validation
        """
        results = TranslationService.translate_documents([text])
        return results[0]

    @staticmethod
    def translate_documents(texts: List[str]) -> List[TranslationResult]:
        """
        Translate several documents, sharing one model batch between them.

        Args:
            texts: English texts to translate

        Returns:
            Results in the same order as the inputs

        Raises:
            ValueError: If any input fails validation
        """
        started = time.perf_counter()

        cleaned = [validate_input_text(text) for text in texts]
        results: List[Optional[TranslationResult]] = [None] * len(cleaned)
        plans: List[Tuple[int, ChunkPlan]] = []
        all_chunks: List[str] = []

        for index, text in enumerate(cleaned):
            request_key = make_cache_key(text, prefix="req")
            hit = cache_get(request_key)
            if hit is not None:
                logger.info("Whole-request cache hit")
                results[index] = TranslationResult(
                    translation=hit["translation"],
                    chunk_count=hit.get("chunk_count", 1),
                    cached=True,
                    duration_ms=0.0,
                    warning=language_warning(text),
                )
                continue

            plan = TranslationService._plan(text)
            plans.append((index, plan))
            all_chunks.extend(plan.chunks)

        translated_chunks = TranslationService._translate_chunks(all_chunks)

        offset = 0
        for index, plan in plans:
            count = plan.chunk_count
            end = offset + count
            piece = translated_chunks[offset:end]
            offset = end
            translation = assemble(plan, piece)

            cache_set(
                make_cache_key(cleaned[index], prefix="req"),
                {"translation": translation, "chunk_count": count},
            )
            results[index] = TranslationResult(
                translation=translation,
                chunk_count=count,
                cached=False,
                duration_ms=0.0,
                warning=language_warning(cleaned[index]),
            )

        duration_ms = (time.perf_counter() - started) * 1000
        logger.info(
            f"Translated {len(cleaned)} document(s) "
            f"({len(all_chunks)} chunk(s)) in {duration_ms:.1f}ms"
        )

        final: List[TranslationResult] = []
        for result in results:
            result.duration_ms = round(duration_ms, 2)
            final.append(result)

        return final

    @staticmethod
    def translate_batch(texts: List[str]) -> List[BatchItemResult]:
        """
        Translate a batch of texts, tolerating individual failures.

        Args:
            texts: Texts to translate

        Returns:
            One result per input, in the submitted order
        """
        results: List[Optional[BatchItemResult]] = [None] * len(texts)
        valid_indices: List[int] = []
        valid_texts: List[str] = []

        for index, text in enumerate(texts):
            try:
                validate_input_text(text)
            except ValueError as exc:
                results[index] = BatchItemResult(
                    index=index, success=False, error=str(exc)
                )
                continue
            valid_indices.append(index)
            valid_texts.append(text)

        if valid_texts:
            try:
                translated = TranslationService.translate_documents(valid_texts)
            except Exception as exc:  # pragma: no cover - defensive
                logger.error(f"Batch translation failed: {exc}")
                for index in valid_indices:
                    results[index] = BatchItemResult(
                        index=index, success=False, error="Translation error"
                    )
                translated = []

            for index, result in zip(valid_indices, translated):
                results[index] = BatchItemResult(
                    index=index,
                    success=True,
                    translation=result.translation,
                    chunk_count=result.chunk_count,
                    cached=result.cached,
                    warning=result.warning,
                )

        return [result for result in results if result is not None]
