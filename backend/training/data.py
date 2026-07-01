"""
Data Loading and Preprocessing for Translation Model Training
"""

from datasets import load_dataset
from transformers import AutoTokenizer

# T5 is a multi-task model: it needs a task prefix telling it what to do.
TRANSLATION_PREFIX = "translate English to Spanish: "


def load_translation_dataset(
    dataset_name: str = "Helsinki-NLP/opus_books", lang_pair: str = "en-es"
):
    """
    Load a translation dataset from Hugging Face.

    Args:
        dataset_name: Name of the dataset. Newer `datasets` versions require
            the full "namespace/name" id, so the bare "opus_books" no longer
            works -> use "Helsinki-NLP/opus_books".
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
    prefix: str = TRANSLATION_PREFIX,
):
    """
    Preprocess translation examples for training.

    Args:
        examples: Batch of examples from dataset
        tokenizer: Tokenizer instance
        source_lang: Source language key
        target_lang: Target language key
        max_length: Maximum sequence length
        prefix: T5 task prefix prepended to every source sentence

    Returns:
        Tokenized inputs with labels
    """
    inputs = [prefix + ex[source_lang] for ex in examples["translation"]]
    targets = [ex[target_lang] for ex in examples["translation"]]

    model_inputs = tokenizer(
        inputs, max_length=max_length, truncation=True, padding="max_length"
    )

    # text_target is the correct, modern way to tokenize the labels for seq2seq.
    labels = tokenizer(
        text_target=targets,
        max_length=max_length,
        truncation=True,
        padding="max_length",
    )

    model_inputs["labels"] = labels["input_ids"]
    return model_inputs


def prepare_dataset(
    model_checkpoint: str,
    dataset_name: str = "Helsinki-NLP/opus_books",
    test_size: float = 0.2,
    max_length: int = 128,
):
    """
    Prepare the full dataset for training.

    Args:
        model_checkpoint: HuggingFace model checkpoint
        dataset_name: Translation dataset to load
        test_size: Fraction of data for testing
        max_length: Maximum sequence length

    Returns:
        Tuple of (tokenized_datasets, tokenizer)
    """
    # Load dataset and tokenizer
    dataset = load_translation_dataset(dataset_name)
    tokenizer = load_tokenizer(model_checkpoint)

    # Split into train/test first, then preprocess both splits identically.
    train_test = dataset["train"].train_test_split(test_size=test_size)

    tokenized = train_test.map(
        lambda x: preprocess_translation_examples(x, tokenizer, max_length=max_length),
        batched=True,
    )

    return tokenized, tokenizer
