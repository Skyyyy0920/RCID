"""Tests for model adapter and loading utilities."""

import pytest
import torch
import torch.nn as nn

from rcid.models.adapter import (
    LLaMA3Adapter,
    ModelAdapter,
    Qwen3Adapter,
    get_adapter,
)
from conftest import TinyAdapter, TinyTransformerModel


# ---------------------------------------------------------------------------
# Adapter factory tests
# ---------------------------------------------------------------------------

class TestGetAdapter:
    def test_qwen3_by_name(self) -> None:
        adapter = get_adapter("Qwen/Qwen3-8B")
        assert isinstance(adapter, Qwen3Adapter)

    def test_qwen3_student(self) -> None:
        adapter = get_adapter("Qwen/Qwen3-0.6B")
        assert isinstance(adapter, Qwen3Adapter)

    def test_llama3_by_name(self) -> None:
        adapter = get_adapter("meta-llama/Llama-3.1-8B")
        assert isinstance(adapter, LLaMA3Adapter)

    def test_llama3_student(self) -> None:
        adapter = get_adapter("meta-llama/Llama-3.2-1B")
        assert isinstance(adapter, LLaMA3Adapter)

    def test_unknown_raises(self) -> None:
        with pytest.raises(ValueError, match="Unknown model family"):
            get_adapter("google/gemma-7b")


# ---------------------------------------------------------------------------
# Model family property
# ---------------------------------------------------------------------------

class TestModelFamily:
    def test_qwen3_family(self) -> None:
        assert Qwen3Adapter().model_family == "qwen3"

    def test_llama3_family(self) -> None:
        assert LLaMA3Adapter().model_family == "llama3"


# ---------------------------------------------------------------------------
# Adapter interface on TinyTransformerModel
# ---------------------------------------------------------------------------

class TestAdapterInterface:
    @pytest.fixture(autouse=True)
    def setup(self) -> None:
        torch.manual_seed(42)
        self.model = TinyTransformerModel(n_layers=4, d_model=32, vocab_size=100)
        self.adapters: list[ModelAdapter] = [
            Qwen3Adapter(),
            LLaMA3Adapter(),
            TinyAdapter(),
        ]

    def test_get_layers_returns_module_list(self) -> None:
        for adapter in self.adapters:
            layers = adapter.get_layers(self.model)
            assert isinstance(layers, nn.ModuleList)
            assert len(layers) == 4

    def test_get_embed_tokens(self) -> None:
        for adapter in self.adapters:
            embed = adapter.get_embed_tokens(self.model)
            assert isinstance(embed, nn.Embedding)
            assert embed.num_embeddings == 100

    def test_get_lm_head(self) -> None:
        for adapter in self.adapters:
            head = adapter.get_lm_head(self.model)
            assert isinstance(head, nn.Linear)
            assert head.out_features == 100

    def test_get_residual_hook_point(self) -> None:
        for adapter in self.adapters:
            for idx in range(4):
                hook_pt = adapter.get_residual_hook_point(self.model, idx)
                assert isinstance(hook_pt, nn.Module)

    def test_get_num_layers(self) -> None:
        for adapter in self.adapters:
            assert adapter.get_num_layers(self.model) == 4

    def test_get_hidden_size(self) -> None:
        for adapter in self.adapters:
            assert adapter.get_hidden_size(self.model) == 32


# ---------------------------------------------------------------------------
# parse_layer_output
# ---------------------------------------------------------------------------

class TestParseLayerOutput:
    def test_extracts_first_element(self) -> None:
        t = torch.randn(2, 5, 32)
        dummy_tuple = (t, None, None)
        for adapter_cls in [Qwen3Adapter, LLaMA3Adapter, TinyAdapter]:
            adapter = adapter_cls()
            result = adapter.parse_layer_output(dummy_tuple)
            assert torch.equal(result, t)


# ---------------------------------------------------------------------------
# TinyTransformerModel forward pass
# ---------------------------------------------------------------------------

class TestTinyModel:
    def test_forward_shape(self) -> None:
        model = TinyTransformerModel(n_layers=4, d_model=32, vocab_size=100)
        input_ids = torch.randint(0, 100, (2, 10))
        out = model(input_ids)
        assert out.logits.shape == (2, 10, 100)

    def test_hook_captures_residual(self) -> None:
        model = TinyTransformerModel(n_layers=4, d_model=32, vocab_size=100)
        adapter = TinyAdapter()
        cache: dict[str, torch.Tensor] = {}

        hook_pt = adapter.get_residual_hook_point(model, 2)
        handle = hook_pt.register_forward_hook(
            lambda mod, inp, out, c=cache: c.update(
                {"h": adapter.parse_layer_output(out).clone()}
            )
        )
        try:
            input_ids = torch.randint(0, 100, (2, 5))
            model(input_ids)
        finally:
            handle.remove()

        assert "h" in cache
        assert cache["h"].shape == (2, 5, 32)  # (batch, seq, d_model)
