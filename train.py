from transformers import TrainingArguments, Trainer
from transformers import EarlyStoppingCallback
from datasets import DatasetDict
import torch
import shutil
from src.logger import get_logger
from src.data import load_data, load_tokenizer, preprocess
from src.model import load_model

logger = get_logger("TranslationTrainer")

MODEL_CKPT = "Helsinki-NLP/opus-mt-en-es"

logger.info("Loading dataset...")
dataset = load_data()
logger.info("Loading tokenizer...")
tokenizer = load_tokenizer(MODEL_CKPT)
logger.info("Preprocessing dataset...")
dataset = dataset.map(lambda x: preprocess(x, tokenizer), batched=True)
tokenized = dataset["train"].train_test_split(test_size=0.2)

logger.info("Loading model...")
model = load_model(MODEL_CKPT)

args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=3,
    weight_decay=0.01,
    logging_dir="./logs"
)

trainer = Trainer(
    model=model,
    args=args,
    train_dataset=tokenized["train"],
    eval_dataset=tokenized["test"],
    tokenizer=tokenizer,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=2)]
)

logger.info("Starting training...")
trainer.train()

logger.info("Saving model and tokenizer...")
model.save_pretrained("fine-tuned-model")
tokenizer.save_pretrained("fine-tuned-tokenizer")