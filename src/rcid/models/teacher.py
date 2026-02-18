"""Teacher model loading utilities."""

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedTokenizer

from rcid.models.adapter import ModelAdapter, get_adapter


def load_teacher(
    model_name: str,
    device: str = "cuda",
    dtype: torch.dtype = torch.float16,
) -> tuple[nn.Module, ModelAdapter, PreTrainedTokenizer]:
    """Load a teacher model in eval mode.

    Args:
        model_name: HuggingFace model name (e.g., 'Qwen/Qwen3-8B').
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
    model.eval()

    adapter = get_adapter(model_name)
    return model, adapter, tokenizer
