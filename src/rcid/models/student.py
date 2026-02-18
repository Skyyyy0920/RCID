"""Student model loading utilities."""

from pathlib import Path

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedTokenizer

from rcid.models.adapter import ModelAdapter, get_adapter


def load_student(
    model_name: str,
    device: str = "cuda",
    dtype: torch.dtype = torch.float16,
) -> tuple[nn.Module, ModelAdapter, PreTrainedTokenizer]:
    """Load a student model for training (NOT in eval mode).

    Args:
        model_name: HuggingFace model name (e.g., 'Qwen/Qwen3-0.6B').
        device: Device or device_map string.
        dtype: Model precision (default fp16).

    Returns:
        (model, adapter, tokenizer) triple.
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=dtype,
        device_map=device,
        trust_remote_code=True,
    )
    # Student stays in train mode; caller calls model.train() if needed.

    adapter = get_adapter(model_name)
    return model, adapter, tokenizer


def load_student_from_checkpoint(
    checkpoint_path: str | Path,
    model_name: str,
    device: str = "cuda",
    dtype: torch.dtype = torch.float16,
) -> tuple[nn.Module, ModelAdapter, PreTrainedTokenizer]:
    """Load a previously distilled student from a checkpoint.

    Args:
        checkpoint_path: Path to saved state_dict (.pt file).
        model_name: Original HuggingFace model name for config/tokenizer.
        device: Device or device_map string.
        dtype: Model precision.

    Returns:
        (model, adapter, tokenizer) triple.
    """
    model, adapter, tokenizer = load_student(model_name, device, dtype)
    state_dict = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    model.load_state_dict(state_dict)
    return model, adapter, tokenizer
