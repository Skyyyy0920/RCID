"""OOD 鲁棒性评估：衡量蒸馏方法在分布偏移下的性能保持率。

对每个任务，比较模型在 in-distribution (ID) 和各 OOD 变体上的准确率，
计算 degradation = (id_acc - ood_acc) / id_acc。

支持任务: ioi, greater_than, induction, sva。
"""

from __future__ import annotations

import logging

import torch
from torch import nn
from torch.utils.data import DataLoader
from transformers import GPT2Tokenizer

logger = logging.getLogger(__name__)


# ======================================================================
# 多任务准确率计算
# ======================================================================

def compute_accuracy(
    model: nn.Module,
    dataset: DataLoader[dict[str, torch.Tensor]],
    task: str,
    device: torch.device | str | None = None,
) -> float:
    """在给定 DataLoader 上计算任务准确率。

    Args:
        model: GPT-2 风格模型。
        dataset: DataLoader，batch 字典。
        task: "ioi" | "greater_than" | "induction" | "sva"。
        device: 目标设备。

    Returns:
        accuracy ∈ [0, 1]。
    """
    if device is None:
        device = next(model.parameters()).device
    device = torch.device(device)

    model.eval()
    correct, total = 0, 0

    with torch.no_grad():
        for batch in dataset:
            if task == "ioi":
                c, n = _acc_ioi(model, batch, device)
            elif task == "greater_than":
                c, n = _acc_gt(model, batch, device)
            elif task == "induction":
                c, n = _acc_induction(model, batch, device)
            elif task == "sva":
                c, n = _acc_sva(model, batch, device)
            else:
                raise ValueError(f"Unknown task: {task!r}")
            correct += c
            total += n

    return correct / max(total, 1)


def _acc_ioi(
    model: nn.Module, batch: dict[str, torch.Tensor], dev: torch.device,
) -> tuple[int, int]:
    """IOI: argmax == answer_token_id。"""
    ids = batch["clean_ids"].to(dev)                # (B, S)
    ans = batch["answer_token_id"].to(dev)          # (B,)
    logits = model(ids).logits                      # (B, S, V)
    last = ids.shape[1] - 1
    preds = logits[:, last, :].argmax(dim=-1)       # (B,)
    return int((preds == ans).sum().item()), ids.shape[0]


def _acc_gt(
    model: nn.Module, batch: dict[str, torch.Tensor], dev: torch.device,
) -> tuple[int, int]:
    """Greater-Than: 预测数字 > clean_threshold。"""
    from rcid.eval.task_accuracy import _get_digit_tokens

    ids = batch["clean_ids"].to(dev)                    # (B, S)
    thr = batch["clean_threshold"].to(dev)              # (B,)
    logits = model(ids).logits                          # (B, S, V)
    last = ids.shape[1] - 1
    last_logits = logits[:, last, :]                    # (B, V)

    dmap = _get_digit_tokens(model)
    d_ids = sorted(dmap.keys())
    d_vals = torch.tensor([dmap[k] for k in d_ids], device=dev)
    d_ids_t = torch.tensor(d_ids, device=dev)

    digit_logits = last_logits[:, d_ids_t]              # (B, n_digits)
    pred_vals = d_vals[digit_logits.argmax(dim=-1)]     # (B,)
    return int((pred_vals > thr).sum().item()), ids.shape[0]


def _acc_induction(
    model: nn.Module, batch: dict[str, torch.Tensor], dev: torch.device,
) -> tuple[int, int]:
    """Induction: argmax at target_pos == answer_token_id。"""
    ids = batch["clean_ids"].to(dev)                    # (B, S)
    tgt = batch["target_pos"].to(dev)                   # (B,)
    ans = batch["answer_token_id"].to(dev)              # (B,)
    logits = model(ids).logits                          # (B, S, V)

    B = ids.shape[0]
    # 逐样本取 target_pos 处的 logits
    tgt_logits = logits[torch.arange(B, device=dev), tgt, :]  # (B, V)
    preds = tgt_logits.argmax(dim=-1)                         # (B,)
    return int((preds == ans).sum().item()), B


def _acc_sva(
    model: nn.Module, batch: dict[str, torch.Tensor], dev: torch.device,
) -> tuple[int, int]:
    """SVA: 单数动词 logit 之和 > 复数动词 logit 之和（clean 为单数主语）。"""
    ids = batch["clean_ids"].to(dev)                    # (B, S)
    vpos = batch["verb_pos"].to(dev)                    # (B,)
    sg_ids = batch["singular_verb_ids"].to(dev)         # (n_verbs,)
    pl_ids = batch["plural_verb_ids"].to(dev)           # (n_verbs,)

    logits = model(ids).logits                          # (B, S, V)
    B = ids.shape[0]
    v_logits = logits[torch.arange(B, device=dev), vpos, :]  # (B, V)

    sg_score = v_logits[:, sg_ids].sum(dim=-1)          # (B,)
    pl_score = v_logits[:, pl_ids].sum(dim=-1)          # (B,)
    # Clean 是单数主语 → 应偏好单数动词
    return int((sg_score > pl_score).sum().item()), B


# ======================================================================
# RobustnessEvaluator
# ======================================================================

class RobustnessEvaluator:
    """OOD 鲁棒性评估器。

    比较模型在 ID 和 OOD 数据集上的准确率，计算性能下降率。
    """

    def __init__(self, device: str | torch.device | None = None) -> None:
        self.device = device

    def evaluate(
        self,
        model: nn.Module,
        id_dataset: DataLoader[dict[str, torch.Tensor]],
        ood_datasets: dict[str, DataLoader[dict[str, torch.Tensor]]],
        task: str,
    ) -> dict:
        """评估模型在 ID 和 OOD 上的准确率及性能下降。

        Args:
            model: GPT-2 风格模型。
            id_dataset: in-distribution DataLoader。
            ood_datasets: {variant_name: DataLoader}。
            task: 任务类型。

        Returns:
            dict:
                id_accuracy: float
                ood_accuracy: {str: float}
                degradation: {str: float}  (id - ood) / id
                mean_degradation: float
        """
        dev = self.device or next(model.parameters()).device
        dev = torch.device(dev)

        id_acc = compute_accuracy(model, id_dataset, task, dev)
        logger.info("ID accuracy (%s): %.4f", task, id_acc)

        ood_acc: dict[str, float] = {}
        degradation: dict[str, float] = {}

        for name, dl in ood_datasets.items():
            acc = compute_accuracy(model, dl, task, dev)
            ood_acc[name] = acc
            # degradation: 正值 = OOD 比 ID 差
            if id_acc > 0:
                degradation[name] = (id_acc - acc) / id_acc
            else:
                degradation[name] = 0.0
            logger.info(
                "OOD %s (%s): acc=%.4f, deg=%.4f",
                name, task, acc, degradation[name],
            )

        mean_deg = (
            sum(degradation.values()) / len(degradation)
            if degradation else 0.0
        )
        logger.info("Mean degradation (%s): %.4f", task, mean_deg)

        return {
            "id_accuracy": id_acc,
            "ood_accuracy": ood_acc,
            "degradation": degradation,
            "mean_degradation": mean_deg,
        }
