from transformers import AutoTokenizer
from transformers import AutoModelForSeq2SeqLM
from transformers import Seq2SeqTrainer
from transformers import Seq2SeqTrainingArguments
from transformers import DataCollatorForSeq2Seq
from datasets import load_dataset
from peft import get_peft_model
from peft import LoraConfig
from peft import TaskType
from peft import prepare_model_for_kbit_training
import evaluate
import numpy as np
import torch
import shutil
from src.logger import get_logger
from src.data import load_data
from src.data import load_tokenizer
from src.data import preprocess
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

# Apply PEFT + LoRA Configuration
lora_config = LoraConfig(
    r=8,
    lora_alpha=32,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.1,
    bias="none",
    task_type=TaskType.SEQ_2_SEQ_LM
)

peft_model = get_peft_model(base_model, lora_config)
peft_model.print_trainable_parameters()

# Training Arguments
training_args = Seq2SeqTrainingArguments(
    output_dir="./results",
    eval_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=3,
    weight_decay=0.01,
    logging_dir="./logs",
    logging_steps=100,
    save_strategy="epoch",
    predict_with_generate=True,
    fp16=torch.cuda.is_available(),
)

# Data Collator
data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=peft_model)

# Evaluation Metric: BLEU
def compute_metrics(eval_preds):
    bleu = evaluate.load("bleu")

    preds, labels = eval_preds

    # Replace -100 in labels as tokenizer.pad_token_id or ignore index
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)

    decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    # BLEU expects list of strings for preds and list of list of strings for refs
    # So wrap references inside list
    references = [[ref] for ref in decoded_labels]

    result = bleu.compute(predictions=decoded_preds, references=references)

    return {"bleu": result["bleu"]}

# Trainer Setup
trainer = Seq2SeqTrainer(
    model=peft_model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["test"],
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)

logger.info("Starting training...")
trainer.train()

logger.info("Saving model and tokenizer...")
model.save_pretrained("fine-tuned-model")
tokenizer.save_pretrained("fine-tuned-tokenizer")