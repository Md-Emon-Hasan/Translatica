from transformers import AutoModelForSeq2SeqLM

def load_model(model_ckpt: str):
    return AutoModelForSeq2SeqLM.from_pretrained(model_ckpt)