"""教师模型加载与管理，始终处于 eval + no_grad 模式。

封装 HuggingFace GPT2LMHeadModel，提供残差流提取接口。
GPT-2 使用 pre-LN 架构，block 输出即为包含残差连接的残差流：
    block_output = block_input + Attn(LN(block_input)) + MLP(LN(...))
"""

from __future__ import annotations

import logging
from contextlib import contextmanager
from typing import Iterator

import torch
from torch import nn
from transformers import GPT2LMHeadModel, GPT2Tokenizer

logger = logging.getLogger(__name__)


class TeacherModel:
    """GPT-2 教师模型包装器。

    加载后自动设置为 eval 模式并冻结参数。
    提供基于 forward hook 的残差流提取接口。
    """

    def __init__(self, model_name: str = "gpt2", device: str | None = None) -> None:
        self.device = torch.device(
            device if device is not None
            else ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.model_name = model_name

        logger.info("Loading teacher model: %s", model_name)
        self.model: GPT2LMHeadModel = GPT2LMHeadModel.from_pretrained(model_name)
        self.model.to(self.device)
        self.model.eval()

        # 冻结所有参数 — 教师永远不更新
        for param in self.model.parameters():
            param.requires_grad = False

        self.n_layers: int = self.model.config.n_layer
        self.d_model: int = self.model.config.n_embd

        logger.info(
            "Teacher ready: %d layers, d_model=%d, device=%s",
            self.n_layers, self.d_model, self.device,
        )

    @property
    def tokenizer(self) -> GPT2Tokenizer:
        """懒加载 tokenizer（大多数流水线不需要它）。"""
        if not hasattr(self, "_tokenizer"):
            self._tokenizer = GPT2Tokenizer.from_pretrained(self.model_name)
        return self._tokenizer

    # ------------------------------------------------------------------
    # Hook 管理 — context manager 确保 hook 不泄漏
    # ------------------------------------------------------------------

    @contextmanager
    def _residual_hooks(
        self,
        layers: list[int],
        storage: dict[int, torch.Tensor],
    ) -> Iterator[dict[int, torch.Tensor]]:
        """临时注册多层 forward hook，提取 block 输出端的残差流。

        GPT-2: model.transformer.h[i] 的 output[0] 是该 block 的残差流，
        形状为 (batch, seq_len, d_model)。

        Args:
            layers: 需要提取的层索引列表。
            storage: 存放结果的字典，key=layer, value=(B, seq_len, d_model)。

        Yields:
            storage 引用，前向传播后即可读取。
        """
        handles: list[torch.utils.hooks.RemovableHook] = []

        for layer in layers:
            assert 0 <= layer < self.n_layers, (
                f"Layer {layer} out of range [0, {self.n_layers})"
            )

            def _make_hook(l: int):  # noqa: E741
                """闭包捕获层索引，避免 late-binding 陷阱。"""
                def hook(
                    module: nn.Module,
                    input: tuple[torch.Tensor, ...],
                    output: tuple[torch.Tensor, ...],
                ) -> None:
                    # GPT-2 block output: (hidden_states, present_kv, ...)
                    storage[l] = output[0]  # (B, seq_len, d_model)
                return hook

            handle = self.model.transformer.h[layer].register_forward_hook(
                _make_hook(layer)
            )
            handles.append(handle)

        try:
            yield storage
        finally:
            for handle in handles:
                handle.remove()

    # ------------------------------------------------------------------
    # 公开接口
    # ------------------------------------------------------------------

    def get_residual_stream(
        self,
        input_ids: torch.Tensor,
        layers: list[int],
        token_positions: list[int],
    ) -> dict[tuple[int, int], torch.Tensor]:
        """一次前向传播提取多层、多位置的残差流切片。

        Args:
            input_ids: 输入 token ids, shape (batch, seq_len).
            layers: 需要提取的层索引列表.
            token_positions: 需要提取的 token 位置列表.

        Returns:
            字典 {(layer, token_pos): tensor}，
            每个 tensor 的 shape 为 (batch, d_model).
        """
        input_ids = input_ids.to(self.device)  # (B, seq_len)

        assert input_ids.dim() == 2, (
            f"input_ids should be 2D (batch, seq_len), got {input_ids.dim()}D"
        )

        seq_len = input_ids.shape[1]
        for t in token_positions:
            assert 0 <= t < seq_len, (
                f"Token position {t} out of range [0, {seq_len})"
            )

        layer_residuals: dict[int, torch.Tensor] = {}

        with torch.no_grad():
            with self._residual_hooks(layers, layer_residuals):
                self.model(input_ids)

        # 从完整残差流中切出指定 token 位置
        result: dict[tuple[int, int], torch.Tensor] = {}
        for layer in layers:
            full_residual = layer_residuals[layer]  # (B, seq_len, d_model)
            for t in token_positions:
                result[(layer, t)] = full_residual[:, t, :]  # (B, d_model)

        return result

    def get_logits(
        self,
        input_ids: torch.Tensor,
    ) -> torch.Tensor:
        """获取教师模型的输出 logits（用于 KL 散度蒸馏）。

        Args:
            input_ids: shape (batch, seq_len).

        Returns:
            logits: shape (batch, seq_len, vocab_size).
        """
        input_ids = input_ids.to(self.device)  # (B, seq_len)

        with torch.no_grad():
            outputs = self.model(input_ids)

        return outputs.logits  # (B, seq_len, vocab_size)
