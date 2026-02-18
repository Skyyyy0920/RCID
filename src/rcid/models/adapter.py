"""Model adapter abstraction layer for Qwen3 and LLaMA3 architectures."""

from abc import ABC, abstractmethod

import torch
import torch.nn as nn


class ModelAdapter(ABC):
    """Unified interface hiding Qwen3/LLaMA3 architecture differences."""

    @property
    @abstractmethod
    def model_family(self) -> str:
        """Return model family identifier ('qwen3' or 'llama3')."""
        ...

    @abstractmethod
    def get_layers(self, model: nn.Module) -> nn.ModuleList:
        """Return the list of transformer layers."""
        ...

    @abstractmethod
    def get_embed_tokens(self, model: nn.Module) -> nn.Embedding:
        """Return the token embedding layer."""
        ...

    @abstractmethod
    def get_lm_head(self, model: nn.Module) -> nn.Linear:
        """Return the language model head."""
        ...

    @abstractmethod
    def get_residual_hook_point(self, model: nn.Module, layer_idx: int) -> nn.Module:
        """Return the module to hook for residual stream at layer_idx."""
        ...

    @abstractmethod
    def parse_layer_output(self, output: tuple) -> torch.Tensor:
        """Extract residual stream tensor from layer output."""
        ...

    @abstractmethod
    def get_num_layers(self, model: nn.Module) -> int:
        """Return the number of transformer layers."""
        ...

    @abstractmethod
    def get_hidden_size(self, model: nn.Module) -> int:
        """Return d_model."""
        ...


class Qwen3Adapter(ModelAdapter):
    """Adapter for Qwen3-8B and Qwen3-0.6B models."""

    @property
    def model_family(self) -> str:
        return "qwen3"

    def get_layers(self, model: nn.Module) -> nn.ModuleList:
        return model.model.layers  # type: ignore[return-value]

    def get_embed_tokens(self, model: nn.Module) -> nn.Embedding:
        return model.model.embed_tokens  # type: ignore[return-value]

    def get_lm_head(self, model: nn.Module) -> nn.Linear:
        return model.lm_head  # type: ignore[return-value]

    def get_residual_hook_point(
        self, model: nn.Module, layer_idx: int
    ) -> nn.Module:
        return model.model.layers[layer_idx]  # type: ignore[return-value]

    def parse_layer_output(self, output: tuple) -> torch.Tensor:
        return output[0]  # (batch, seq, d_model)

    def get_num_layers(self, model: nn.Module) -> int:
        return len(model.model.layers)  # type: ignore[arg-type]

    def get_hidden_size(self, model: nn.Module) -> int:
        return model.config.hidden_size  # type: ignore[return-value]


class LLaMA3Adapter(ModelAdapter):
    """Adapter for LLaMA-3-8B and LLaMA-3.2-1B models."""

    @property
    def model_family(self) -> str:
        return "llama3"

    def get_layers(self, model: nn.Module) -> nn.ModuleList:
        return model.model.layers  # type: ignore[return-value]

    def get_embed_tokens(self, model: nn.Module) -> nn.Embedding:
        return model.model.embed_tokens  # type: ignore[return-value]

    def get_lm_head(self, model: nn.Module) -> nn.Linear:
        return model.lm_head  # type: ignore[return-value]

    def get_residual_hook_point(
        self, model: nn.Module, layer_idx: int
    ) -> nn.Module:
        return model.model.layers[layer_idx]  # type: ignore[return-value]

    def parse_layer_output(self, output: tuple) -> torch.Tensor:
        return output[0]  # (batch, seq, d_model)

    def get_num_layers(self, model: nn.Module) -> int:
        return len(model.model.layers)  # type: ignore[arg-type]

    def get_hidden_size(self, model: nn.Module) -> int:
        return model.config.hidden_size  # type: ignore[return-value]


def get_adapter(model_name: str) -> ModelAdapter:
    """Return the appropriate adapter based on model name."""
    name_lower = model_name.lower()
    if "qwen" in name_lower:
        return Qwen3Adapter()
    elif "llama" in name_lower or "meta" in name_lower:
        return LLaMA3Adapter()
    else:
        raise ValueError(f"Unknown model family: {model_name}")
