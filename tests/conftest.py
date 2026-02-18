"""Shared test fixtures: tiny transformer models mimicking HuggingFace interface."""

from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn

from rcid.models.adapter import ModelAdapter


class TinyTransformerLayer(nn.Module):
    """Minimal transformer layer returning (hidden_states,) tuple."""

    def __init__(self, d_model: int) -> None:
        super().__init__()
        self.linear = nn.Linear(d_model, d_model, bias=False)

    def forward(
        self, hidden_states: torch.Tensor, **kwargs: object
    ) -> tuple[torch.Tensor, ...]:
        out = self.linear(hidden_states)  # (batch, seq, d_model)
        return (out,)


class TinyTransformerModel(nn.Module):
    """Minimal model mimicking HuggingFace CausalLM interface."""

    def __init__(
        self,
        n_layers: int = 4,
        d_model: int = 32,
        vocab_size: int = 100,
    ) -> None:
        super().__init__()
        self.config = SimpleNamespace(hidden_size=d_model)
        inner = nn.Module()
        inner.embed_tokens = nn.Embedding(vocab_size, d_model)
        inner.layers = nn.ModuleList(
            [TinyTransformerLayer(d_model) for _ in range(n_layers)]
        )
        self.model = inner
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)

    def forward(
        self, input_ids: torch.Tensor, **kwargs: object
    ) -> SimpleNamespace:
        h = self.model.embed_tokens(input_ids)  # (batch, seq, d_model)
        for layer in self.model.layers:
            h = layer(h)[0]
        logits = self.lm_head(h)  # (batch, seq, vocab)
        return SimpleNamespace(logits=logits)


class TinyAdapter(ModelAdapter):
    """Adapter for TinyTransformerModel used in tests."""

    @property
    def model_family(self) -> str:
        return "test"

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
        return output[0]

    def get_num_layers(self, model: nn.Module) -> int:
        return len(model.model.layers)  # type: ignore[arg-type]

    def get_hidden_size(self, model: nn.Module) -> int:
        return model.config.hidden_size  # type: ignore[return-value]


@pytest.fixture
def tiny_model() -> TinyTransformerModel:
    """4-layer tiny transformer, d_model=32, vocab=100."""
    torch.manual_seed(42)
    return TinyTransformerModel(n_layers=4, d_model=32, vocab_size=100)


@pytest.fixture
def tiny_adapter() -> TinyAdapter:
    return TinyAdapter()
