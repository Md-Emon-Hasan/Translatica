"""
Main Training Script for Translation Model Fine-tuning

Usage:
    python -m training.train

This script fine-tunes the Helsinki-NLP/opus-mt-en-es model using LoRA + PEFT
for English to Spanish translation.
"""

import argparse
from pathlib import Path

from training.data import prepare_dataset
from training.logger import get_training_logger
from training.model import create_peft_model, print_trainable_parameters
from training.trainer import create_trainer, get_training_arguments

# Default configuration
DEFAULT_MODEL_CHECKPOINT = "Helsinki-NLP/opus-mt-en-es"
DEFAULT_OUTPUT_DIR = "./fine-tuned-model"
DEFAULT_NUM_EPOCHS = 3
DEFAULT_BATCH_SIZE = 16
DEFAULT_LEARNING_RATE = 2e-5


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Fine-tune translation model with LoRA + PEFT"
    )
    parser.add_argument(
        "--model-checkpoint",
        type=str,
        default=DEFAULT_MODEL_CHECKPOINT,
        help="HuggingFace model checkpoint",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory to save the fine-tuned model",
    )
    parser.add_argument(
        "--num-epochs",
        type=int,
        default=DEFAULT_NUM_EPOCHS,
        help="Number of training epochs",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=DEFAULT_BATCH_SIZE,
        help="Batch size for training",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=DEFAULT_LEARNING_RATE,
        help="Learning rate",
    )
    return parser.parse_args()


def main():
    """Main training function."""
    args = parse_args()

    # Setup logger
    logger = get_training_logger("TranslationTrainer")

    logger.info("=" * 60)
    logger.info("Starting Translation Model Training")
    logger.info("=" * 60)
    logger.info(f"Model checkpoint: {args.model_checkpoint}")
    logger.info(f"Output directory: {args.output_dir}")
    logger.info(f"Epochs: {args.num_epochs}")
    logger.info(f"Batch size: {args.batch_size}")
    logger.info(f"Learning rate: {args.learning_rate}")

    # Step 1: Prepare dataset
    logger.info("Step 1: Preparing dataset...")
    train_test_dataset, tokenizer = prepare_dataset(args.model_checkpoint)
    logger.info(f"Train size: {len(train_test_dataset['train'])}")
    logger.info(f"Test size: {len(train_test_dataset['test'])}")

    # Step 2: Create PEFT model
    logger.info("Step 2: Creating PEFT model with LoRA...")
    peft_model = create_peft_model(args.model_checkpoint)
    print_trainable_parameters(peft_model)

    # Step 3: Setup training arguments
    logger.info("Step 3: Setting up training arguments...")
    training_args = get_training_arguments(
        output_dir=args.output_dir,
        num_epochs=args.num_epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
    )

    # Step 4: Create trainer
    logger.info("Step 4: Creating trainer...")
    trainer = create_trainer(
        model=peft_model,
        tokenizer=tokenizer,
        train_dataset=train_test_dataset["train"],
        eval_dataset=train_test_dataset["test"],
        training_args=training_args,
    )

    # Step 5: Train
    logger.info("Step 5: Starting training...")
    trainer.train()

    # Step 6: Save model and tokenizer
    logger.info("Step 6: Saving model and tokenizer...")
    output_path = Path(args.output_dir)

    # Save PEFT model
    model_path = output_path / "fine-tuned-model"
    peft_model.save_pretrained(model_path)
    logger.info(f"Model saved to: {model_path}")

    # Save tokenizer
    tokenizer_path = output_path / "fine-tuned-tokenizer"
    tokenizer.save_pretrained(tokenizer_path)
    logger.info(f"Tokenizer saved to: {tokenizer_path}")

    logger.info("=" * 60)
    logger.info("Training completed successfully!")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
