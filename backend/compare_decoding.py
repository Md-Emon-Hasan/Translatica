"""
Greedy vs Beam Search Comparison

Prints both decodings of a few fixed sentences side by side. Qualitative
comparison only - no metrics are computed.

Usage:
    cd backend
    python compare_decoding.py
"""

import time

from app.core.config import settings
from app.core.model import model_manager

SAMPLES = [
    "The lighthouse keeper watched the storm gather over the black water.",
    "She had never told anyone about the letter hidden beneath the floorboards.",
    "It was the kind of morning that made even old sorrows feel bearable.",
    "He walked slowly, counting the steps between the church and the harbour.",
    "The garden had grown wild in the years since her mother passed away.",
]


def translate_with(num_beams: int, texts):
    """Translate texts with a given beam count."""
    original = settings.GEN_NUM_BEAMS
    settings.GEN_NUM_BEAMS = num_beams
    try:
        started = time.perf_counter()
        outputs = model_manager.generate_texts(texts)
        duration = time.perf_counter() - started
    finally:
        settings.GEN_NUM_BEAMS = original

    return outputs, duration


def main() -> None:
    """Print greedy and beam-search translations side by side."""
    print("Loading model...")
    model_manager.load()

    greedy, greedy_time = translate_with(1, SAMPLES)
    beams, beams_time = translate_with(4, SAMPLES)

    for source, greedy_out, beam_out in zip(SAMPLES, greedy, beams):
        print("\n" + "-" * 78)
        print(f"EN            : {source}")
        print(f"num_beams=1   : {greedy_out}")
        print(f"num_beams=4   : {beam_out}")

    print("\n" + "=" * 78)
    print(f"greedy      total: {greedy_time:.2f}s for {len(SAMPLES)} sentences")
    print(f"beam search total: {beams_time:.2f}s for {len(SAMPLES)} sentences")
    print("Qualitative comparison only - no metrics are computed here.")


if __name__ == "__main__":
    main()
