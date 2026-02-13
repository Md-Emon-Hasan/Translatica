"""
Data Loading and Preprocessing for Translation Model Training
"""

from datasets import load_dataset
from transformers import AutoTokenizer


def load_translation_dataset(
    dataset_name: str = "opus_books", lang_pair: str = "en-es"
):
    """
    Load a translation dataset from Hugging Face.

    Args:
        dataset_name: Name of the dataset (default: opus_books)
        lang_pair: Language pair (default: en-es for English-Spanish)

    Returns:
        Loaded dataset
    """
    return load_dataset(dataset_name, lang_pair)


def load_tokenizer(model_checkpoint: str):
    """
    Load tokenizer from a model checkpoint.

    Args:
        model_checkpoint: HuggingFace model checkpoint name

    Returns:
        AutoTokenizer instance
    """
    return AutoTokenizer.from_pretrained(model_checkpoint)


def preprocess_translation_examples(
    examples,
    tokenizer,
    source_lang: str = "en",
    target_lang: str = "es",
    max_length: int = 128,
):
    """
    Preprocess translation examples for training.

    Args:
        examples: Batch of examples from dataset
        tokenizer: Tokenizer instance
        source_lang: Source language key
        target_lang: Target language key
        max_length: Maximum sequence length

    Returns:
        Tokenized inputs with labels
    """
    inputs = [ex[source_lang] for ex in examples["translation"]]
    targets = [ex[target_lang] for ex in examples["translation"]]

    model_inputs = tokenizer(
        inputs, max_length=max_length, truncation=True, padding="max_length"
    )

    labels = tokenizer(
        targets, max_length=max_length, truncation=True, padding="max_length"
    )

    model_inputs["labels"] = labels["input_ids"]
    return model_inputs


def prepare_dataset(
    model_checkpoint: str, test_size: float = 0.2, max_length: int = 128
):
    """
    Prepare the full dataset for training.

    Args:
        model_checkpoint: HuggingFace model checkpoint
        test_size: Fraction of data for testing
        max_length: Maximum sequence length

    Returns:
        Tuple of (tokenized_datasets, tokenizer)
    """
    # Load dataset and tokenizer
    dataset = load_translation_dataset()
    tokenizer = load_tokenizer(model_checkpoint)

    # Preprocess
    tokenized = dataset.map(
        lambda x: preprocess_translation_examples(x, tokenizer, max_length=max_length),
        batched=True,
    )

    # Split
    train_test = tokenized["train"].train_test_split(test_size=test_size)

    return train_test, tokenizer
