"""
Training Configuration and Trainer Setup
"""

import evaluate
import nltk
import numpy as np
from transformers import (
    DataCollatorForSeq2Seq,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
)


def get_training_arguments(
    output_dir: str = "./results",
    num_epochs: int = 3,
    batch_size: int = 16,
    learning_rate: float = 2e-5,
    weight_decay: float = 0.01,
    logging_dir: str = "./logs",
    logging_steps: int = 100,
):
    """
    Create training arguments for Seq2Seq training.

    Args:
        output_dir: Directory to save outputs
        num_epochs: Number of training epochs
        batch_size: Batch size for training/evaluation
        learning_rate: Learning rate
        weight_decay: Weight decay for regularization
        logging_dir: Directory for logs
        logging_steps: Steps between logging

    Returns:
        Seq2SeqTrainingArguments instance
    """
    return Seq2SeqTrainingArguments(
        output_dir=output_dir,
        eval_strategy="epoch",
        learning_rate=learning_rate,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        num_train_epochs=num_epochs,
        weight_decay=weight_decay,
        logging_dir=logging_dir,
        logging_steps=logging_steps,
        save_strategy="epoch",
        predict_with_generate=True,
        generation_max_length=128,
        # IMPORTANT: T5 + fp16 often gives NaN loss; keep it off.
        fp16=False,
        load_best_model_at_end=True,
        metric_for_best_model="bleu",
        greater_is_better=True,
    )


def get_compute_metrics(tokenizer):
    """
    Create a compute_metrics function evaluating BLEU, chrF and METEOR.

    Args:
        tokenizer: Tokenizer for decoding

    Returns:
        Compute metrics function
    """
    # METEOR needs these NLTK resources.
    for resource in ("wordnet", "omw-1.4", "punkt", "punkt_tab"):
        try:
            nltk.download(resource, quiet=True)
        except Exception:  # pragma: no cover - best-effort resource download
            pass

    # Load the metrics once (more efficient than reloading every eval step).
    bleu_metric = evaluate.load("bleu")  # word n-gram overlap
    chrf_metric = evaluate.load("chrf")  # character n-gram (good for Spanish)
    meteor_metric = evaluate.load("meteor")  # synonyms + word order

    def compute_metrics(eval_preds):
        preds, labels = eval_preds

        # predict_with_generate can return a tuple; take the first element.
        if isinstance(preds, tuple):
            preds = preds[0]

        preds = np.array(preds)
        labels = np.array(labels)

        vocab_size = len(tokenizer)
        # Replace ANY invalid token id (-100 or out-of-vocab) with pad token id
        # -> this is what fixes the OverflowError during decoding.
        preds = np.where(
            (preds >= 0) & (preds < vocab_size), preds, tokenizer.pad_token_id
        )
        labels = np.where(
            (labels >= 0) & (labels < vocab_size), labels, tokenizer.pad_token_id
        )

        decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

        decoded_preds = [p.strip() for p in decoded_preds]
        decoded_labels = [label.strip() for label in decoded_labels]

        refs_nested = [[ref] for ref in decoded_labels]

        bleu_result = bleu_metric.compute(
            predictions=decoded_preds, references=refs_nested
        )
        chrf_result = chrf_metric.compute(
            predictions=decoded_preds, references=refs_nested
        )
        meteor_result = meteor_metric.compute(
            predictions=decoded_preds, references=decoded_labels
        )

        return {
            "bleu": round(bleu_result["bleu"], 4),
            "chrf": round(chrf_result["score"], 4),
            "meteor": round(meteor_result["meteor"], 4),
        }

    return compute_metrics


def create_trainer(model, tokenizer, train_dataset, eval_dataset, training_args=None):
    """
    Create a Seq2SeqTrainer instance.

    Args:
        model: PEFT model to train
        tokenizer: Tokenizer
        train_dataset: Training dataset
        eval_dataset: Evaluation dataset
        training_args: Training arguments (optional)

    Returns:
        Seq2SeqTrainer instance
    """
    if training_args is None:
        training_args = get_training_arguments()

    # Data collator
    data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model)

    # Compute metrics function
    compute_metrics = get_compute_metrics(tokenizer)

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    return trainer
