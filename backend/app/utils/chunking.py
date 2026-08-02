"""
Sentence-Aware Text Chunking
"""

import math
import re
from dataclasses import dataclass, field
from typing import Callable, List, Optional, Tuple

from app.core.config import settings
from app.utils.logger import get_logger

logger = get_logger("chunking")

_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?…])[\"'”»\)\]]*\s+")
_SPANISH_OPENERS_RE = re.compile(r"\s+(?=[¿¡])")
_CLAUSE_SPLIT_RE = re.compile(r"(?<=[;:,])\s+")
_PARAGRAPH_SPLIT_RE = re.compile(r"\n\s*\n+")


def estimate_tokens(text: str) -> int:
    """Estimate tokens from character count."""
    per_token = max(1, settings.CHARS_PER_TOKEN_ESTIMATE)
    return max(1, math.ceil(len(text) / per_token))


def get_token_counter() -> Callable[[str], int]:
    """
    Get a token counting function.

    Uses the loaded model's tokenizer, falling back to a character-based
    estimate when it is not reachable.
    """
    try:
        from app.core.model import model_manager

        if model_manager.is_loaded:
            tokenizer = model_manager.tokenizer

            def count(text: str) -> int:
                encoded = tokenizer(text, add_special_tokens=False)
                return len(encoded["input_ids"])

            return count
    except Exception as exc:  # pragma: no cover - defensive
        logger.warning(f"Tokenizer unavailable for chunking ({exc})")
        return estimate_tokens

    logger.warning(
        "Tokenizer not loaded; falling back to character-based token estimate"
    )
    return estimate_tokens


def split_sentences(text: str) -> List[str]:
    """Split text into sentences, keeping punctuation."""
    normalized = " ".join(text.split())
    if not normalized:
        return []

    pieces: List[str] = []
    for part in _SENTENCE_SPLIT_RE.split(normalized):
        pieces.extend(p for p in _SPANISH_OPENERS_RE.split(part) if p)

    return [p.strip() for p in pieces if p.strip()]


def split_clauses(sentence: str) -> List[str]:
    """Split a sentence on clause boundaries."""
    parts = [p.strip() for p in _CLAUSE_SPLIT_RE.split(sentence)]
    return [p for p in parts if p]


def split_paragraphs(text: str) -> List[str]:
    """Split text into paragraph blocks on blank lines."""
    blocks = [b.strip() for b in _PARAGRAPH_SPLIT_RE.split(text)]
    return [b for b in blocks if b]


def _hard_split(
    text: str, budget: int, count_tokens: Callable[[str], int]
) -> List[str]:
    """Pack words up to the budget, cutting a single oversized word if needed."""
    windows: List[str] = []
    current: List[str] = []

    for word in text.split():
        if count_tokens(word) > budget:
            if current:
                windows.append(" ".join(current))
                current = []
            width = max(1, budget * max(1, settings.CHARS_PER_TOKEN_ESTIMATE))
            for start in range(0, len(word), width):
                end = start + width
                windows.append(word[start:end])
            continue

        candidate = current + [word]
        if current and count_tokens(" ".join(candidate)) > budget:
            windows.append(" ".join(current))
            current = [word]
        else:
            current = candidate

    if current:
        windows.append(" ".join(current))

    return windows or ([text] if text else [])


def _fragments(
    sentence: str, budget: int, count_tokens: Callable[[str], int]
) -> List[str]:
    """Break an over-budget sentence into fragments that fit."""
    if count_tokens(sentence) <= budget:
        return [sentence]

    fragments: List[str] = []
    clauses = split_clauses(sentence)
    if len(clauses) <= 1:
        return _hard_split(sentence, budget, count_tokens)

    for clause in clauses:
        if count_tokens(clause) <= budget:
            fragments.append(clause)
        else:
            fragments.extend(_hard_split(clause, budget, count_tokens))

    return fragments


def chunk_text(
    text: str,
    token_budget: Optional[int] = None,
    count_tokens: Optional[Callable[[str], int]] = None,
) -> List[str]:
    """
    Split a block of text into chunks within the token budget.

    Args:
        text: Text to chunk
        token_budget: Maximum tokens per chunk (defaults to config)
        count_tokens: Token counter (defaults to the model tokenizer)

    Returns:
        Ordered list of chunks
    """
    budget = token_budget or settings.CHUNK_TOKEN_BUDGET
    counter = count_tokens or get_token_counter()

    sentences = split_sentences(text)
    if not sentences:
        return []

    chunks: List[str] = []
    current: List[str] = []

    for sentence in sentences:
        for fragment in _fragments(sentence, budget, counter):
            candidate = current + [fragment]
            if current and counter(" ".join(candidate)) > budget:
                chunks.append(" ".join(current))
                current = [fragment]
            else:
                current = candidate

    if current:
        chunks.append(" ".join(current))

    return chunks


@dataclass
class ChunkPlan:
    """A chunked document: flat chunk list plus per-paragraph spans."""

    chunks: List[str] = field(default_factory=list)
    paragraph_spans: List[Tuple[int, int]] = field(default_factory=list)

    @property
    def chunk_count(self) -> int:
        """Number of chunks in the plan."""
        return len(self.chunks)


def plan_chunks(
    text: str,
    token_budget: Optional[int] = None,
    count_tokens: Optional[Callable[[str], int]] = None,
) -> ChunkPlan:
    """
    Chunk a document, recording where each paragraph starts and ends.

    Args:
        text: Document to chunk
        token_budget: Maximum tokens per chunk (defaults to config)
        count_tokens: Token counter (defaults to the model tokenizer)

    Returns:
        ChunkPlan for the document
    """
    counter = count_tokens or get_token_counter()
    plan = ChunkPlan()

    for block in split_paragraphs(text):
        start = len(plan.chunks)
        block_chunks = chunk_text(
            block, token_budget=token_budget, count_tokens=counter
        )
        if not block_chunks:
            continue
        plan.chunks.extend(block_chunks)
        plan.paragraph_spans.append((start, len(plan.chunks)))

    return plan


def assemble(plan: ChunkPlan, translated_chunks: List[str]) -> str:
    """Rejoin chunks with spaces and paragraphs with blank lines."""
    paragraphs: List[str] = []
    for start, end in plan.paragraph_spans:
        pieces = [p.strip() for p in translated_chunks[start:end] if p and p.strip()]
        if pieces:
            paragraphs.append(" ".join(pieces))

    return "\n\n".join(paragraphs)
