"""
Training Configuration and Trainer Setup
"""

import evaluate
import numpy as np
import torch
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
        fp16=torch.cuda.is_available(),
        load_best_model_at_end=True,
        metric_for_best_model="bleu",
    )


def get_compute_metrics(tokenizer):
    """
    Create a compute_metrics function for BLEU evaluation.

    Args:
        tokenizer: Tokenizer for decoding

    Returns:
        Compute metrics function
    """
    bleu = evaluate.load("bleu")

    def compute_metrics(eval_preds):
        preds, labels = eval_preds

        # Replace -100 with pad token id
        labels = np.where(labels != -100, labels, tokenizer.pad_token_id)

        # Decode predictions and labels
        decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

        # BLEU expects list of references
        references = [[ref] for ref in decoded_labels]

        result = bleu.compute(predictions=decoded_preds, references=references)

        return {"bleu": result["bleu"]}

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
