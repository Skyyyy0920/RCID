"""因果干预一致性评估，验证学生是否保留教师的回路行为。

核心思想：
对教师和学生施加 *相同* 的 activation patching 干预
（在检查点位置将 clean 残差流替换为 corrupt 残差流），
观察两者输出变化是否一致。

两套接口：
    evaluate()                    — 符号一致性 (sign agreement) [向后兼容]
    CausalConsistencyEvaluator    — Pearson 相关 + delta 张量 + p-value
"""

from __future__ import annotations

import logging
import math
from collections import defaultdict
from contextlib import contextmanager
from typing import Iterator

import torch
from torch import nn
from torch.utils.data import DataLoader

logger = logging.getLogger(__name__)


# ======================================================================
# Activation Patching 工具函数
# ======================================================================

@contextmanager
def _patch_hook(
    model: nn.Module, layer: int, token_pos: int,
    corrupt_residual: torch.Tensor,
) -> Iterator[None]:
    """将指定层 token_pos 残差流替换为 corrupt 版本。"""
    def hook(
        module: nn.Module, input: tuple, output: tuple,
    ) -> tuple[torch.Tensor, ...]:
        hidden = output[0].clone()  # (B, seq_len, d_model)
        hidden[:, token_pos, :] = corrupt_residual[:, token_pos, :]
        return (hidden,) + output[1:]

    handle = model.transformer.h[layer].register_forward_hook(hook)
    try:
        yield
    finally:
        handle.remove()


def _get_layer_residual(
    model: nn.Module, input_ids: torch.Tensor, layer: int,
) -> torch.Tensor:
    """前向传播并提取指定层的残差流。返回 (B, seq_len, d_model)。"""
    storage: dict[str, torch.Tensor] = {}

    def hook(module: nn.Module, input: tuple, output: tuple) -> None:
        storage["r"] = output[0]

    handle = model.transformer.h[layer].register_forward_hook(hook)
    try:
        model(input_ids)
    finally:
        handle.remove()
    return storage["r"]  # (B, seq_len, d_model)


def _patched_logits(
    model: nn.Module,
    clean_ids: torch.Tensor,       # (B, seq_len)
    corrupt_ids: torch.Tensor,     # (B, seq_len)
    layer: int, token_pos: int,
) -> torch.Tensor:
    """Activation patching → patched logits (B, seq_len, V)。"""
    corrupt_residual = _get_layer_residual(model, corrupt_ids, layer)
    with _patch_hook(model, layer, token_pos, corrupt_residual):
        outputs = model(clean_ids)
    return outputs.logits  # (B, seq_len, vocab_size)


def _gather_logit_delta(
    base_logits: torch.Tensor,       # (B, vocab_size)
    patched_logits: torch.Tensor,    # (B, vocab_size)
    token_ids: torch.Tensor,         # (B,)
) -> torch.Tensor:
    """指定 token 的 logit 变化量 (B,): patched - base。"""
    base_vals = base_logits.gather(1, token_ids.unsqueeze(1)).squeeze(1)
    patched_vals = patched_logits.gather(1, token_ids.unsqueeze(1)).squeeze(1)
    return patched_vals - base_vals  # (B,)


# ======================================================================
# Pearson 相关（避免 scipy 依赖）
# ======================================================================

def _pearson_r(x: torch.Tensor, y: torch.Tensor) -> tuple[float, float]:
    """Pearson r 和双尾 p-value。n < 3 时返回 (0.0, 1.0)。"""
    n = x.numel()
    if n < 3:
        return 0.0, 1.0
    x_c = x - x.mean()
    y_c = y - y.mean()
    numer = (x_c * y_c).sum()
    denom = (x_c.pow(2).sum() * y_c.pow(2).sum()).sqrt()
    if denom < 1e-12:
        return 0.0, 1.0
    r = (numer / denom).clamp(-1.0, 1.0).item()
    r2 = r * r
    if r2 >= 1.0:
        return r, 0.0
    t_stat = abs(r) * math.sqrt((n - 2) / (1.0 - r2))
    p_value = 2.0 * _t_survival(t_stat, n - 2)
    return r, p_value


def _t_survival(t: float, df: int) -> float:
    """t 分布 P(T > t) 近似 (Abramowitz & Stegun)。"""
    if df <= 0:
        return 0.5
    z = t * (1.0 - 1.0 / (4.0 * df)) / math.sqrt(1.0 + t * t / (2.0 * df))
    return _normal_sf(z)


def _normal_sf(z: float) -> float:
    """标准正态 P(Z > z) 近似 (A&S 26.2.17)。"""
    if z < 0:
        return 1.0 - _normal_sf(-z)
    b1, b2, b3, b4, b5 = (
        0.319381530, -0.356563782, 1.781477937, -1.882496350, 1.330274429,
    )
    t = 1.0 / (1.0 + 0.2316419 * z)
    poly = t * (b1 + t * (b2 + t * (b3 + t * (b4 + t * b5))))
    pdf = math.exp(-0.5 * z * z) / math.sqrt(2.0 * math.pi)
    return max(pdf * poly, 0.0)


# ======================================================================
# CausalConsistencyEvaluator — Pearson 相关 + delta 张量
# ======================================================================

class CausalConsistencyEvaluator:
    """因果一致性评估器：收集 delta 张量，计算 Pearson 相关。

    Δ = logit_base[answer] - logit_patched[answer]
    正值 = patching 降低了正确答案的 logit（符合因果干预预期）。
    """

    def __init__(self, device: str | torch.device | None = None) -> None:
        self.device = device

    def evaluate(
        self,
        teacher: nn.Module,
        student: nn.Module,
        dataset: DataLoader[dict[str, torch.Tensor]],
        checkpoints: list[tuple[int, int]],
        layer_mapping: dict[int, int],
    ) -> dict:
        """返回 per-checkpoint Pearson r、p-value 和 delta 矩阵。"""
        device = self.device or next(teacher.parameters()).device
        device = torch.device(device)
        teacher.eval()
        student.eval()

        t_deltas: dict[int, list[torch.Tensor]] = defaultdict(list)
        s_deltas: dict[int, list[torch.Tensor]] = defaultdict(list)

        with torch.no_grad():
            for batch in dataset:
                clean = batch["clean_ids"].to(device)       # (B, S)
                corrupt = batch["corrupt_ids"].to(device)    # (B, S)
                ans = batch.get("answer_token_id")
                if ans is not None:
                    ans = ans.to(device)                     # (B,)
                last = clean.shape[1] - 1

                t_base = teacher(clean).logits[:, last, :]   # (B, V)
                s_base = student(clean).logits[:, last, :]   # (B, V)
                if ans is None:
                    ans = t_base.argmax(dim=-1)              # (B,)

                for ci, (tl, tp) in enumerate(checkpoints):
                    sl = layer_mapping[tl]
                    t_p = _patched_logits(
                        teacher, clean, corrupt, tl, tp,
                    )[:, last, :]                            # (B, V)
                    s_p = _patched_logits(
                        student, clean, corrupt, sl, tp,
                    )[:, last, :]                            # (B, V)
                    # Δ = base - patched
                    t_deltas[ci].append(
                        _gather_logit_delta(t_p, t_base, ans).cpu(),
                    )
                    s_deltas[ci].append(
                        _gather_logit_delta(s_p, s_base, ans).cpu(),
                    )

        n_cp = len(checkpoints)
        t_mat = torch.stack([torch.cat(t_deltas[i]) for i in range(n_cp)])
        s_mat = torch.stack([torch.cat(s_deltas[i]) for i in range(n_cp)])

        cp_corr: dict[tuple[int, int], float] = {}
        cp_pval: dict[tuple[int, int], float] = {}
        for i, cp in enumerate(checkpoints):
            r, p = _pearson_r(t_mat[i], s_mat[i])
            cp_corr[cp] = r
            cp_pval[cp] = p
            logger.info("CP (%d,%d): r=%.4f p=%.4e", cp[0], cp[1], r, p)

        mean_r = sum(cp_corr.values()) / max(len(cp_corr), 1)
        logger.info("Mean Pearson r: %.4f", mean_r)

        return {
            "per_checkpoint_correlation": cp_corr,
            "per_checkpoint_pvalue": cp_pval,
            "mean_correlation": mean_r,
            "teacher_deltas": t_mat,
            "student_deltas": s_mat,
        }


# ======================================================================
# evaluate() — 向后兼容，符号一致性
# ======================================================================

def evaluate(
    model: nn.Module,
    dataset: DataLoader[dict[str, torch.Tensor]],
    *,
    teacher_model: nn.Module,
    checkpoints: list[tuple[int, int]],
    layer_mapping: dict[int, int],
    device: str | None = None,
) -> dict[str, float]:
    """符号一致性评估（向后兼容）。

    返回 {"consistency", "consistency_per_cp", "n_checkpoints"}。
    """
    dev = device or next(model.parameters()).device
    dev = torch.device(dev)
    model.eval()
    teacher_model.eval()

    cp_ok: dict[tuple[int, int], int] = defaultdict(int)
    cp_n: dict[tuple[int, int], int] = defaultdict(int)

    with torch.no_grad():
        for batch in dataset:
            clean = batch["clean_ids"].to(dev)
            corrupt = batch["corrupt_ids"].to(dev)
            ans = batch.get("answer_token_id")
            if ans is not None:
                ans = ans.to(dev)
            last = clean.shape[1] - 1

            t_base = teacher_model(clean).logits[:, last, :]
            s_base = model(clean).logits[:, last, :]

            for tl, tp in checkpoints:
                sl = layer_mapping[tl]
                t_p = _patched_logits(
                    teacher_model, clean, corrupt, tl, tp,
                )[:, last, :]
                s_p = _patched_logits(
                    model, clean, corrupt, sl, tp,
                )[:, last, :]

                if ans is not None:
                    td = _gather_logit_delta(t_base, t_p, ans)
                    sd = _gather_logit_delta(s_base, s_p, ans)
                else:
                    top = t_base.argmax(dim=-1)
                    td = _gather_logit_delta(t_base, t_p, top)
                    sd = _gather_logit_delta(s_base, s_p, top)

                cp_ok[(tl, tp)] += int(((td * sd) >= 0).sum().item())
                cp_n[(tl, tp)] += clean.shape[0]

    total = sum(cp_ok.values())
    count = sum(cp_n.values())
    overall = total / max(count, 1)

    scores = [cp_ok[c] / cp_n[c] for c in checkpoints if cp_n[c] > 0]
    avg = sum(scores) / len(scores) if scores else 0.0

    return {
        "consistency": overall,
        "consistency_per_cp": avg,
        "n_checkpoints": float(len(checkpoints)),
    }
