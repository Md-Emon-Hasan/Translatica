"""
Model Loading and LoRA Configuration for Fine-tuning
"""

from peft import LoraConfig, TaskType, get_peft_model
from transformers import AutoModelForSeq2SeqLM


def load_base_model(model_checkpoint: str):
    """
    Load base translation model from HuggingFace.

    Args:
        model_checkpoint: Model checkpoint name

    Returns:
        Pre-trained model
    """
    return AutoModelForSeq2SeqLM.from_pretrained(model_checkpoint)


def get_lora_config(
    r: int = 8,
    lora_alpha: int = 32,
    lora_dropout: float = 0.1,
    target_modules: list = None,
):
    """
    Create LoRA configuration for PEFT.

    Args:
        r: LoRA rank
        lora_alpha: LoRA alpha scaling
        lora_dropout: Dropout probability
        target_modules: Modules to apply LoRA to

    Returns:
        LoraConfig instance
    """
    if target_modules is None:
        target_modules = ["q_proj", "v_proj"]

    return LoraConfig(
        r=r,
        lora_alpha=lora_alpha,
        target_modules=target_modules,
        lora_dropout=lora_dropout,
        bias="none",
        task_type=TaskType.SEQ_2_SEQ_LM,
    )


def create_peft_model(model_checkpoint: str, lora_config: LoraConfig = None):
    """
    Create a PEFT model with LoRA adapters.

    Args:
        model_checkpoint: Base model checkpoint
        lora_config: LoRA configuration (optional)

    Returns:
        PEFT model with LoRA adapters
    """
    # Load base model
    base_model = load_base_model(model_checkpoint)

    # Create LoRA config if not provided
    if lora_config is None:
        lora_config = get_lora_config()

    # Apply PEFT
    peft_model = get_peft_model(base_model, lora_config)

    return peft_model


def print_trainable_parameters(model):
    """
    Print the number of trainable parameters in the model.

    Args:
        model: PEFT model
    """
    trainable = 0
    total = 0
    for _, param in model.named_parameters():
        total += param.numel()
        if param.requires_grad:
            trainable += param.numel()

    print(
        f"Trainable params: {trainable:,} || Total params: {total:,} || "
        f"Trainable %: {100 * trainable / total:.4f}%"
    )
