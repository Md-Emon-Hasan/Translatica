# src/data.py
from datasets import load_dataset
from transformers import AutoTokenizer

def load_data():
    return load_dataset("opus_books", "en-es")

def load_tokenizer(model_ckpt: str):
    return AutoTokenizer.from_pretrained(model_ckpt)

def preprocess(examples, tokenizer):
    inputs = [ex["en"] for ex in examples["translation"]]
    targets = [ex["es"] for ex in examples["translation"]]
    model_inputs = tokenizer(inputs, max_length=128, truncation=True, padding="max_length")
    labels = tokenizer(targets, max_length=128, truncation=True, padding="max_length")
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs