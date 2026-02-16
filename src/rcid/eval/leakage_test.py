"""因果痕迹信息泄露测试（线性探针法）。

验证 RCID 因果痕迹确实捕捉了任务特异性信息，而非通用激活模式。
1. 从因果痕迹训练线性分类器预测任务标签 → 准确率应 >> chance
2. 同一分类器预测控制标签（与任务无关） → 准确率应 ≈ chance
3. selectivity = task_acc - control_acc（越高越好）

线性探针使用纯 PyTorch 实现（避免 sklearn NumPy DLL 依赖问题）。
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import torch
from torch import nn

logger = logging.getLogger(__name__)


# ======================================================================
# 纯 PyTorch 线性探针
# ======================================================================

class _LinearProbe(nn.Module):
    """二分类线性探针：logistic regression。"""

    def __init__(self, d_input: int) -> None:
        super().__init__()
        self.linear = nn.Linear(d_input, 1)  # (d_input,) → scalar

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x).squeeze(-1)  # (B,)


def _train_linear_probe(
    X_train: torch.Tensor,   # (N_train, d)
    y_train: torch.Tensor,   # (N_train,) — 0/1
    X_test: torch.Tensor,    # (N_test, d)
    y_test: torch.Tensor,    # (N_test,) — 0/1
    *,
    lr: float = 1e-2,
    weight_decay: float = 1e-3,
    n_epochs: int = 200,
    seed: int = 0,
) -> float:
    """训练线性探针，返回 test accuracy ∈ [0, 1]。"""
    torch.manual_seed(seed)
    d = X_train.shape[1]
    probe = _LinearProbe(d)

    # 标准化特征（zero mean, unit variance）
    mean = X_train.mean(dim=0, keepdim=True)                 # (1, d)
    std = X_train.std(dim=0, keepdim=True).clamp(min=1e-8)   # (1, d)
    X_tr = (X_train - mean) / std   # (N_train, d)
    X_te = (X_test - mean) / std    # (N_test, d)

    y_tr = y_train.float()
    optimizer = torch.optim.AdamW(
        probe.parameters(), lr=lr, weight_decay=weight_decay,
    )
    loss_fn = nn.BCEWithLogitsLoss()

    probe.train()
    for _ in range(n_epochs):
        logits = probe(X_tr)          # (N_train,)
        loss = loss_fn(logits, y_tr)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    probe.eval()
    with torch.no_grad():
        preds = (probe(X_te) > 0.0).long()  # (N_test,)
    return (preds == y_test).float().mean().item()


# ======================================================================
# 标签生成工具
# ======================================================================

def make_binary_labels(raw_labels: torch.Tensor) -> torch.Tensor:
    """任意整数标签 → 二分类。按 median 分割: > median → 1。"""
    assert raw_labels.dim() == 1
    med = raw_labels.float().median()
    return (raw_labels.float() > med).long()  # (n,)


def make_control_even_length(seq_lengths: torch.Tensor) -> torch.Tensor:
    """控制标签：token 数是否为偶数。"""
    return (seq_lengths % 2 == 0).long()


def make_control_first_token_even(first_token_ids: torch.Tensor) -> torch.Tensor:
    """控制标签：第一个 token id 是否为偶数。"""
    return (first_token_ids % 2 == 0).long()


def make_control_random(n: int, seed: int = 12345) -> torch.Tensor:
    """控制标签：随机 0/1。"""
    rng = torch.Generator().manual_seed(seed)
    return torch.randint(0, 2, (n,), generator=rng)


# ======================================================================
# LeakageTestEvaluator
# ======================================================================

@dataclass
class ProbeResult:
    """单次探针测试的结果。"""
    accuracy: float
    n_train: int
    n_test: int
    class_balance: float  # 正类比例


class LeakageTestEvaluator:
    """因果痕迹信息泄露测试器。

    验证因果痕迹包含任务特异性信息（task probe 高准确率），
    而不包含无关信息（control probe ≈ chance）。
    """

    def __init__(
        self,
        test_ratio: float = 0.2,
        n_epochs: int = 200,
        lr: float = 1e-2,
        seed: int = 42,
    ) -> None:
        assert 0.0 < test_ratio < 1.0
        self.test_ratio = test_ratio
        self.n_epochs = n_epochs
        self.lr = lr
        self.seed = seed

    def _split(self, n: int) -> tuple[torch.Tensor, torch.Tensor]:
        """返回 (train_idx, test_idx)。"""
        rng = torch.Generator().manual_seed(self.seed)
        perm = torch.randperm(n, generator=rng)
        n_test = max(1, int(n * self.test_ratio))
        return perm[n_test:], perm[:n_test]

    def _run_probe(
        self,
        imprints: torch.Tensor,   # (n, d)
        labels: torch.Tensor,     # (n,) 0/1
        name: str,
    ) -> ProbeResult:
        """训练线性探针并返回结果。"""
        n = imprints.shape[0]
        assert labels.shape == (n,), f"Label shape {labels.shape} != ({n},)"

        train_idx, test_idx = self._split(n)
        X_tr, y_tr = imprints[train_idx], labels[train_idx]
        X_te, y_te = imprints[test_idx], labels[test_idx]
        balance = labels.float().mean().item()

        # 标签全部相同 → 无法训练
        if y_tr.unique().numel() < 2:
            logger.warning("Probe '%s': constant train labels", name)
            return ProbeResult(
                accuracy=max(balance, 1.0 - balance),
                n_train=len(train_idx), n_test=len(test_idx),
                class_balance=balance,
            )

        acc = _train_linear_probe(
            X_tr, y_tr, X_te, y_te,
            lr=self.lr, n_epochs=self.n_epochs, seed=self.seed,
        )
        logger.info(
            "Probe '%s': acc=%.4f (train=%d, test=%d, bal=%.2f)",
            name, acc, len(train_idx), len(test_idx), balance,
        )
        return ProbeResult(
            accuracy=acc, n_train=len(train_idx),
            n_test=len(test_idx), class_balance=balance,
        )

    def run_test(
        self,
        causal_imprints: torch.Tensor,  # (n_samples, d_model)
        task_labels: torch.Tensor,      # (n_samples,) 二分类
        control_labels: torch.Tensor,   # (n_samples,) 二分类
    ) -> dict:
        """运行泄露测试。

        Returns:
            task_probe_accuracy, control_probe_accuracy,
            selectivity (task - control), task_probe, control_probe。
        """
        assert causal_imprints.dim() == 2
        n = causal_imprints.shape[0]
        assert task_labels.shape == (n,) and control_labels.shape == (n,)

        X = causal_imprints.detach().float()
        task_r = self._run_probe(X, task_labels, "task")
        ctrl_r = self._run_probe(X, control_labels, "control")
        sel = task_r.accuracy - ctrl_r.accuracy

        logger.info(
            "Leakage: task=%.4f, control=%.4f, sel=%.4f",
            task_r.accuracy, ctrl_r.accuracy, sel,
        )
        return {
            "task_probe_accuracy": task_r.accuracy,
            "control_probe_accuracy": ctrl_r.accuracy,
            "selectivity": sel,
            "task_probe": task_r,
            "control_probe": ctrl_r,
        }

    def run_multi_control_test(
        self,
        causal_imprints: torch.Tensor,          # (n_samples, d_model)
        task_labels: torch.Tensor,               # (n_samples,) 二分类
        control_labels: dict[str, torch.Tensor], # {name: (n_samples,)}
    ) -> dict:
        """多个控制标签的泄露测试。

        Returns:
            task_probe_accuracy, control_accuracies {name: acc},
            mean_control_accuracy, selectivity。
        """
        assert causal_imprints.dim() == 2
        n = causal_imprints.shape[0]
        X = causal_imprints.detach().float()

        task_r = self._run_probe(X, task_labels, "task")

        ctrl_accs: dict[str, float] = {}
        for name, ctrl in control_labels.items():
            assert ctrl.shape == (n,), f"Control '{name}' shape mismatch"
            ctrl_accs[name] = self._run_probe(X, ctrl, f"ctrl_{name}").accuracy

        mean_ctrl = sum(ctrl_accs.values()) / max(len(ctrl_accs), 1)
        return {
            "task_probe_accuracy": task_r.accuracy,
            "control_accuracies": ctrl_accs,
            "mean_control_accuracy": mean_ctrl,
            "selectivity": task_r.accuracy - mean_ctrl,
        }
