"""
Sentence-Aware Chunking Tests
"""

from unittest.mock import MagicMock, patch

from app.utils.chunking import (
    ChunkPlan,
    assemble,
    chunk_text,
    estimate_tokens,
    get_token_counter,
    plan_chunks,
    split_clauses,
    split_paragraphs,
    split_sentences,
)


def word_tokens(text: str) -> int:
    """Deterministic token counter: one token per whitespace-separated word."""
    return len(text.split())


class TestSentenceSplitting:
    """Tests for sentence and clause splitting."""

    def test_splits_on_terminators(self):
        sentences = split_sentences("One. Two! Three? Four.")
        assert sentences == ["One.", "Two!", "Three?", "Four."]

    def test_keeps_abbreviated_text_together_when_no_boundary(self):
        assert split_sentences("A single sentence without an end") == [
            "A single sentence without an end"
        ]

    def test_splits_before_spanish_opening_marks(self):
        sentences = split_sentences("Hola amigo ¿cómo estás? ¡Muy bien!")
        assert sentences == ["Hola amigo", "¿cómo estás?", "¡Muy bien!"]

    def test_empty_input(self):
        assert split_sentences("   \n  ") == []

    def test_splits_clauses(self):
        clauses = split_clauses("First part; second part, third part: done")
        assert clauses == ["First part;", "second part,", "third part:", "done"]

    def test_paragraph_split(self):
        assert split_paragraphs("A\n\nB\n \nC") == ["A", "B", "C"]


class TestChunkText:
    """Tests for greedy token-budget packing."""

    def test_short_text_is_one_chunk(self):
        chunks = chunk_text("Hello world.", token_budget=50, count_tokens=word_tokens)
        assert chunks == ["Hello world."]

    def test_empty_text_yields_no_chunks(self):
        assert chunk_text("   ", token_budget=10, count_tokens=word_tokens) == []

    def test_long_text_splits_on_sentence_boundaries(self):
        text = " ".join(f"Sentence number {i} is here." for i in range(12))
        chunks = chunk_text(text, token_budget=10, count_tokens=word_tokens)

        assert len(chunks) > 1
        for chunk in chunks:
            assert chunk.endswith(".")
            assert word_tokens(chunk) <= 10

    def test_token_budget_is_respected(self):
        text = " ".join(f"Sentence {i} here." for i in range(30))
        for budget in (4, 8, 16):
            chunks = chunk_text(text, token_budget=budget, count_tokens=word_tokens)
            assert all(word_tokens(chunk) <= budget for chunk in chunks)

    def test_rejoining_reproduces_all_content(self):
        text = " ".join(f"Sentence number {i} is here." for i in range(12))
        chunks = chunk_text(text, token_budget=7, count_tokens=word_tokens)
        assert " ".join(chunks) == text

    def test_over_long_single_sentence_splits_on_clauses(self):
        sentence = "Alpha beta gamma; delta epsilon zeta, eta theta iota: kappa lambda."
        chunks = chunk_text(sentence, token_budget=3, count_tokens=word_tokens)

        assert len(chunks) > 1
        assert " ".join(chunks) == sentence
        assert all(word_tokens(chunk) <= 3 for chunk in chunks)

    def test_over_long_clauseless_sentence_hard_splits_without_losing_text(self):
        sentence = " ".join(f"word{i}" for i in range(20))
        chunks = chunk_text(sentence, token_budget=4, count_tokens=word_tokens)

        assert len(chunks) == 5
        assert " ".join(chunks) == sentence

    def test_single_word_longer_than_budget_is_windowed(self):
        chunks = chunk_text(
            "abcdefghij", token_budget=2, count_tokens=lambda text: len(text)
        )
        assert "".join(chunks) == "abcdefghij"
        assert len(chunks) > 1

    def test_words_before_an_oversized_word_are_flushed(self):
        chunks = chunk_text(
            "short " + "x" * 12,
            token_budget=6,
            count_tokens=lambda text: len(text.replace(" ", "")),
        )
        assert chunks[0] == "short"
        assert "".join(chunk.replace(" ", "") for chunk in chunks) == "short" + "x" * 12

    def test_oversized_clause_is_hard_split(self):
        sentence = "small one, " + " ".join(f"w{i}" for i in range(10)) + "."
        chunks = chunk_text(sentence, token_budget=3, count_tokens=word_tokens)

        assert " ".join(chunks) == sentence
        assert all(word_tokens(chunk) <= 3 for chunk in chunks)

    def test_uses_config_budget_by_default(self):
        chunks = chunk_text("Hello world.", count_tokens=word_tokens)
        assert chunks == ["Hello world."]


class TestPlanAndAssemble:
    """Tests for document planning and reassembly."""

    def test_paragraph_breaks_survive(self):
        text = (
            "First one. Second one. Third one.\n\n"
            "Another paragraph here. And more of it."
        )
        plan = plan_chunks(text, token_budget=4, count_tokens=word_tokens)

        assert len(plan.paragraph_spans) == 2
        assert plan.chunk_count == len(plan.chunks) > 2

        rejoined = assemble(plan, plan.chunks)
        assert rejoined.count("\n\n") == 1
        assert rejoined.startswith("First one.")
        assert "Another paragraph here." in rejoined

    def test_assemble_uses_translated_chunks_in_order(self):
        plan = ChunkPlan(chunks=["a", "b", "c"], paragraph_spans=[(0, 2), (2, 3)])
        assert assemble(plan, ["A", "B", "C"]) == "A B\n\nC"

    def test_assemble_skips_empty_translations(self):
        plan = ChunkPlan(chunks=["a", "b"], paragraph_spans=[(0, 1), (1, 2)])
        assert assemble(plan, ["A", "  "]) == "A"

    def test_plan_ignores_blank_paragraphs(self):
        plan = plan_chunks("Only this.\n\n   \n\n", count_tokens=word_tokens)
        assert plan.chunks == ["Only this."]
        assert plan.paragraph_spans == [(0, 1)]

    def test_no_text_is_dropped_across_paragraphs(self):
        paragraph_a = " ".join(f"Alpha {i} here." for i in range(8))
        paragraph_b = " ".join(f"Beta {i} here." for i in range(8))
        text = f"{paragraph_a}\n\n{paragraph_b}"

        plan = plan_chunks(text, token_budget=6, count_tokens=word_tokens)
        assert assemble(plan, plan.chunks) == text


class TestTokenCounter:
    """Tests for token counter resolution."""

    def test_estimate_tokens_is_character_based(self):
        assert estimate_tokens("") == 1
        assert estimate_tokens("a" * 40) == 10

    def test_uses_tokenizer_when_model_is_loaded(self):
        manager = MagicMock()
        manager.is_loaded = True
        manager.tokenizer = lambda text, add_special_tokens=False: {
            "input_ids": text.split()
        }

        with patch("app.core.model.model_manager", manager):
            counter = get_token_counter()

        assert counter("one two three") == 3

    def test_falls_back_to_estimate_when_not_loaded(self):
        manager = MagicMock()
        manager.is_loaded = False

        with patch("app.core.model.model_manager", manager):
            counter = get_token_counter()

        assert counter is estimate_tokens
