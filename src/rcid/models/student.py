"""学生模型定义与初始化。

4 层、384 维的小型 GPT-2 风格 Transformer，作为蒸馏目标模型。
与 TeacherModel 共享相同的 get_residual_stream 接口签名。

关键区别于教师：
- 学生的残差流提取 **保留梯度**（不用 no_grad），以便 loss.backward() 更新参数。
- hooks 中 **不** 调用 .detach()，否则梯度链断裂。
"""

from __future__ import annotations

import logging
from contextlib import contextmanager
from typing import Iterator

import torch
from torch import nn
from transformers import GPT2Config, GPT2LMHeadModel

logger = logging.getLogger(__name__)

# 默认学生架构超参（与 configs/base.yaml 一致）
_DEFAULT_STUDENT_CONFIG = {
    "n_layer": 4,
    "n_embd": 384,
    "n_head": 6,
    "vocab_size": 50257,  # GPT-2 tokenizer 词表大小
    "n_positions": 1024,
    "n_inner": 384 * 4,   # FFN 隐藏维度 = 4 * d_model
}


class StudentModel:
    """小型 GPT-2 风格学生模型。

    提供与 TeacherModel 完全一致的 get_residual_stream 接口，
    但前向传播保留计算图以支持梯度回传。
    """

    def __init__(
        self,
        model: GPT2LMHeadModel,
        device: str | None = None,
    ) -> None:
        """内部构造器，请通过 from_scratch / from_pretrained 创建实例。"""
        self.device = torch.device(
            device if device is not None
            else ("cuda" if torch.cuda.is_available() else "cpu")
        )

        self.model: GPT2LMHeadModel = model
        self.model.to(self.device)
        self.model.train()

        self.n_layers: int = self.model.config.n_layer
        self.d_model: int = self.model.config.n_embd

        logger.info(
            "Student ready: %d layers, d_model=%d, device=%s",
            self.n_layers, self.d_model, self.device,
        )

    # ------------------------------------------------------------------
    # 工厂方法
    # ------------------------------------------------------------------

    @classmethod
    def from_scratch(
        cls,
        n_layer: int = _DEFAULT_STUDENT_CONFIG["n_layer"],
        n_embd: int = _DEFAULT_STUDENT_CONFIG["n_embd"],
        n_head: int = _DEFAULT_STUDENT_CONFIG["n_head"],
        vocab_size: int = _DEFAULT_STUDENT_CONFIG["vocab_size"],
        device: str | None = None,
    ) -> StudentModel:
        """从零初始化学生模型（随机权重）。

        Args:
            n_layer: Transformer 层数（默认 4）。
            n_embd: 隐藏维度 d_model（默认 384）。
            n_head: 注意力头数（默认 6，head_dim = 384/6 = 64）。
            vocab_size: 词表大小（默认 50257，与 GPT-2 tokenizer 对齐）。
            device: 目标设备。
        """
        assert n_embd % n_head == 0, (
            f"n_embd ({n_embd}) must be divisible by n_head ({n_head})"
        )

        config = GPT2Config(
            vocab_size=vocab_size,
            n_positions=1024,
            n_embd=n_embd,
            n_layer=n_layer,
            n_head=n_head,
            n_inner=n_embd * 4,
        )

        logger.info(
            "Initializing student from scratch: %d layers, d_model=%d, n_head=%d",
            n_layer, n_embd, n_head,
        )
        model = GPT2LMHeadModel(config)
        return cls(model=model, device=device)

    @classmethod
    def from_pretrained(
        cls,
        model_name: str,
        device: str | None = None,
    ) -> StudentModel:
        """从 HuggingFace 预训练权重加载学生模型（如 "distilgpt2"）。

        Args:
            model_name: HuggingFace model ID。
            device: 目标设备。
        """
        logger.info("Loading student from pretrained: %s", model_name)
        model = GPT2LMHeadModel.from_pretrained(model_name)
        return cls(model=model, device=device)

    # ------------------------------------------------------------------
    # Hook 管理 — context manager 确保 hook 不泄漏
    # 与 TeacherModel._residual_hooks 结构一致，
    # 但 **不** 在 hook 内调用 .detach()，保留梯度链。
    # ------------------------------------------------------------------

    @contextmanager
    def _residual_hooks(
        self,
        layers: list[int],
        storage: dict[int, torch.Tensor],
    ) -> Iterator[dict[int, torch.Tensor]]:
        """临时注册多层 forward hook，提取 block 输出端的残差流。

        与 TeacherModel._residual_hooks 的关键区别：
        hook 中 **不** detach output，保留计算图以支持 backward()。

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
                    # 不 detach — 梯度必须流回学生参数
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
    # 公开接口 — 与 TeacherModel 签名完全一致
    # ------------------------------------------------------------------

    def get_residual_stream(
        self,
        input_ids: torch.Tensor,
        layers: list[int],
        token_positions: list[int],
    ) -> dict[tuple[int, int], torch.Tensor]:
        """一次前向传播提取多层、多位置的残差流切片。

        与 TeacherModel.get_residual_stream 签名完全一致。
        关键区别：**不使用 torch.no_grad()**，返回的 tensor 保留梯度。

        Args:
            input_ids: 输入 token ids, shape (batch, seq_len).
            layers: 需要提取的层索引列表.
            token_positions: 需要提取的 token 位置列表.

        Returns:
            字典 {(layer, token_pos): tensor}，
            每个 tensor 的 shape 为 (batch, d_model)，且保留梯度。
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

        # 不包裹 no_grad — 学生前向传播必须保留计算图
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
        """获取学生模型的输出 logits（用于 KL 散度蒸馏）。

        保留梯度，loss.backward() 可更新学生参数。

        Args:
            input_ids: shape (batch, seq_len).

        Returns:
            logits: shape (batch, seq_len, vocab_size).
        """
        input_ids = input_ids.to(self.device)  # (B, seq_len)
        outputs = self.model(input_ids)
        return outputs.logits  # (B, seq_len, vocab_size)
