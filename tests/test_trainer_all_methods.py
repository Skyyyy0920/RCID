"""DistillationTrainer 7 种方法统一测试。

测试覆盖:
- 7 种方法均能实例化训练器
- 7 种方法均能训练 5 步且不报错
- 每种方法返回正确的损失项名称
- loss_fn 的可学习参数被加入 optimizer
- 无效方法名被拒绝
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from transformers import GPT2Config, GPT2LMHeadModel

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from rcid.distillation.baselines import (
    FitNetsLoss,
    PrakashCKALoss,
    StandardKDLoss,
)
from rcid.distillation.informed_fitnets import InformedFitNetsLoss
from rcid.distillation.minilm import MiniLMStyleKD
from rcid.distillation.rcid_loss import RCIDLoss
from rcid.distillation.tinybert import TinyBERTStyleKD
from rcid.distillation.trainer import (
    DistillationTrainer,
    TrainConfig,
    TrainState,
    VALID_METHODS,
)


# ======================================================================
# Fixtures: tiny models + dummy data
# ======================================================================

T_LAYERS = 4    # 教师层数
S_LAYERS = 2    # 学生层数
D_T = 32        # 教师隐藏维度
D_S = 16        # 学生隐藏维度
H_T = 2         # 教师头数
H_S = 2         # 学生头数
VOCAB = 100
SEQ_LEN = 8
B = 4           # batch size
N_STEPS = 5     # 训练步数


def _make_gpt2(n_layer: int, n_embd: int, n_head: int) -> GPT2LMHeadModel:
    cfg = GPT2Config(
        vocab_size=VOCAB, n_positions=64,
        n_embd=n_embd, n_layer=n_layer, n_head=n_head,
        n_inner=n_embd * 4,
    )
    return GPT2LMHeadModel(cfg)


@pytest.fixture(scope="module")
def teacher() -> GPT2LMHeadModel:
    torch.manual_seed(42)
    m = _make_gpt2(T_LAYERS, D_T, H_T)
    m.eval()
    return m


@pytest.fixture(scope="module")
def make_student():
    """每次调用返回一个新的学生模型（训练需要独立副本）。"""
    def _fn():
        torch.manual_seed(99)
        return _make_gpt2(S_LAYERS, D_S, H_S)
    return _fn


class _RepeatingLoader:
    """重复同一 batch N 次的 DataLoader 替代品。"""

    def __init__(self, batch: dict[str, torch.Tensor], n: int) -> None:
        self.batch = batch
        self.n = n

    def __iter__(self):
        for _ in range(self.n):
            yield self.batch

    def __len__(self) -> int:
        return self.n


@pytest.fixture(scope="module")
def dummy_batch() -> dict[str, torch.Tensor]:
    """单个 batch (B, SEQ_LEN) 的 clean + corrupt ids。"""
    torch.manual_seed(42)
    return {
        "clean_ids": torch.randint(1, VOCAB, (B, SEQ_LEN)),
        "corrupt_ids": torch.randint(1, VOCAB, (B, SEQ_LEN)),
    }


@pytest.fixture(scope="module")
def dummy_loader(dummy_batch) -> _RepeatingLoader:
    """重复 dummy_batch N_STEPS 次。"""
    return _RepeatingLoader(dummy_batch, N_STEPS)


@pytest.fixture(scope="module")
def teacher_imprints(teacher, dummy_batch) -> dict[tuple[int, int], torch.Tensor]:
    """预计算教师痕迹 (匹配 dummy_batch)。"""
    checkpoints = [(0, 2), (1, 4)]
    clean = dummy_batch["clean_ids"]      # (B, SEQ_LEN)
    corrupt = dummy_batch["corrupt_ids"]  # (B, SEQ_LEN)

    imprints: dict[tuple[int, int], torch.Tensor] = {}
    layers = sorted(set(l for l, t in checkpoints))

    with torch.no_grad():
        for layer in layers:
            sc: dict[int, torch.Tensor] = {}
            sx: dict[int, torch.Tensor] = {}

            def _mh(l: int, s: dict):
                def hook(mod, inp, out):
                    s[l] = out[0]
                return hook

            h = teacher.transformer.h[layer].register_forward_hook(_mh(layer, sc))
            teacher(clean)
            h.remove()

            h = teacher.transformer.h[layer].register_forward_hook(_mh(layer, sx))
            teacher(corrupt)
            h.remove()

            for ll, tp in checkpoints:
                if ll == layer:
                    imprints[(ll, tp)] = sc[layer][:, tp, :] - sx[layer][:, tp, :]

    return imprints


# ======================================================================
# Loss fn factory per method
# ======================================================================

def _make_loss_fn(method: str) -> nn.Module:
    """根据方法名创建对应的损失函数。"""
    checkpoints = [(0, 2), (1, 4)]
    layer_mapping = {0: 0, 1: 1}

    if method == "standard_kd":
        return StandardKDLoss(temperature=4.0)
    if method == "fitnets":
        return FitNetsLoss(D_T, D_S, layer_pairs=[(0, 0), (2, 1)])
    if method == "prakash_cka":
        return PrakashCKALoss(D_T, D_S, layer_pairs=[(0, 0), (2, 1)])
    if method == "tinybert":
        return TinyBERTStyleKD(
            d_teacher=D_T, d_student=D_S,
            n_head_teacher=H_T, n_head_student=H_S,
            layer_pairs=[(0, 0), (2, 1)],
        )
    if method == "minilm":
        return MiniLMStyleKD(
            n_head_teacher=H_T, n_head_student=H_S,
            layer_pairs=[(1, 0)],
        )
    if method == "rcid":
        W = torch.randn(D_T, D_S)
        return RCIDLoss(W, checkpoints, layer_mapping)
    if method == "informed_fitnets":
        W = torch.randn(D_T, D_S)
        return InformedFitNetsLoss(W, checkpoints, layer_mapping)
    raise ValueError(f"Unknown method: {method}")


# ======================================================================
# Test: 基本构造
# ======================================================================

class TestTrainerConstruction:

    def test_all_methods_listed(self) -> None:
        assert len(VALID_METHODS) == 7

    def test_invalid_method_rejected(self, teacher, make_student) -> None:
        cfg = TrainConfig(method_name="bad_method")
        loss_fn = StandardKDLoss()
        with pytest.raises(AssertionError, match="Unknown method"):
            DistillationTrainer(teacher, make_student(), cfg, loss_fn, device="cpu")

    @pytest.mark.parametrize("method", VALID_METHODS)
    def test_construction(
        self, method, teacher, make_student, teacher_imprints,
    ) -> None:
        cfg = TrainConfig(method_name=method, epochs=1, lr=1e-3)
        loss_fn = _make_loss_fn(method)
        imps = teacher_imprints if method in ("rcid", "informed_fitnets") else None
        trainer = DistillationTrainer(
            teacher, make_student(), cfg, loss_fn,
            teacher_imprints=imps, device="cpu",
        )
        assert trainer.method == method
        assert isinstance(trainer.state, TrainState)


# ======================================================================
# Test: loss_fn 的参数是否被加入 optimizer
# ======================================================================

class TestOptimizerParams:

    def test_fitnets_projections_in_optimizer(
        self, teacher, make_student,
    ) -> None:
        cfg = TrainConfig(method_name="fitnets", lr=1e-3)
        loss_fn = _make_loss_fn("fitnets")
        student = make_student()
        trainer = DistillationTrainer(
            teacher, student, cfg, loss_fn, device="cpu",
        )
        n_student = sum(1 for _ in student.parameters())
        n_loss = sum(1 for _ in loss_fn.parameters())
        assert n_loss > 0, "FitNets should have learnable projections"
        # optimizer 应包含两者的参数
        n_opt = sum(len(pg["params"]) for pg in trainer.optimizer.param_groups)
        assert n_opt == n_student + n_loss

    def test_tinybert_projections_in_optimizer(
        self, teacher, make_student,
    ) -> None:
        cfg = TrainConfig(method_name="tinybert", lr=1e-3)
        loss_fn = _make_loss_fn("tinybert")
        student = make_student()
        trainer = DistillationTrainer(
            teacher, student, cfg, loss_fn, device="cpu",
        )
        n_loss = sum(1 for _ in loss_fn.parameters())
        assert n_loss > 0, "TinyBERT should have hidden projections"
        n_opt = sum(len(pg["params"]) for pg in trainer.optimizer.param_groups)
        n_student = sum(1 for _ in student.parameters())
        assert n_opt == n_student + n_loss

    def test_rcid_no_loss_params(self, teacher, make_student, teacher_imprints):
        cfg = TrainConfig(method_name="rcid", lr=1e-3)
        loss_fn = _make_loss_fn("rcid")
        student = make_student()
        trainer = DistillationTrainer(
            teacher, student, cfg, loss_fn,
            teacher_imprints=teacher_imprints, device="cpu",
        )
        n_loss = sum(1 for _ in loss_fn.parameters())
        assert n_loss == 0, "RCID W is a buffer, not parameter"
        n_opt = sum(len(pg["params"]) for pg in trainer.optimizer.param_groups)
        n_student = sum(1 for _ in student.parameters())
        assert n_opt == n_student


# ======================================================================
# Test: 训练 5 步 — 每种方法
# ======================================================================

class TestTrainFiveSteps:
    """每种方法训练 N_STEPS 步, 验证不崩溃且损失合理。"""

    @pytest.mark.parametrize("method", VALID_METHODS)
    def test_train_five_steps(
        self, method, teacher, make_student,
        dummy_loader, teacher_imprints,
    ) -> None:
        cfg = TrainConfig(
            method_name=method, epochs=1, lr=1e-3,
            lambda_kl=1.0, lambda_rcid=1.0, log_every=100,
        )
        loss_fn = _make_loss_fn(method)
        imps = teacher_imprints if method in ("rcid", "informed_fitnets") else None
        student = make_student()

        trainer = DistillationTrainer(
            teacher, student, cfg, loss_fn,
            teacher_imprints=imps, device="cpu",
        )
        state = trainer.train(dummy_loader)

        assert isinstance(state, TrainState)
        assert state.global_step == N_STEPS
        assert len(state.train_losses) == 1
        assert state.train_losses[0] > 0  # 损失 > 0（非退化）
        assert state.train_losses[0] < 1e6  # 无爆炸


# ======================================================================
# Test: 各方法返回的损失项名称
# ======================================================================

class TestLossParts:
    """验证 _compute_batch_loss 返回正确的损失项。"""

    def _run_one_step(self, method, teacher, student, loader, imprints):
        cfg = TrainConfig(
            method_name=method, epochs=1, lr=1e-3,
            lambda_kl=1.0, lambda_rcid=1.0,
        )
        loss_fn = _make_loss_fn(method)
        imps = imprints if method in ("rcid", "informed_fitnets") else None
        trainer = DistillationTrainer(
            teacher, student, cfg, loss_fn,
            teacher_imprints=imps, device="cpu",
        )
        batch = next(iter(loader))
        loss, parts = trainer._compute_batch_loss(batch)
        return loss, parts

    def test_standard_kd_parts(
        self, teacher, make_student, dummy_loader, teacher_imprints,
    ) -> None:
        loss, parts = self._run_one_step(
            "standard_kd", teacher, make_student(), dummy_loader, teacher_imprints,
        )
        assert loss.isfinite()
        assert "kl_loss" in parts

    def test_fitnets_parts(
        self, teacher, make_student, dummy_loader, teacher_imprints,
    ) -> None:
        loss, parts = self._run_one_step(
            "fitnets", teacher, make_student(), dummy_loader, teacher_imprints,
        )
        assert loss.isfinite()
        assert "kl_loss" in parts
        assert "aux_loss" in parts

    def test_prakash_parts(
        self, teacher, make_student, dummy_loader, teacher_imprints,
    ) -> None:
        loss, parts = self._run_one_step(
            "prakash_cka", teacher, make_student(), dummy_loader, teacher_imprints,
        )
        assert loss.isfinite()
        assert "kl_loss" in parts
        assert "aux_loss" in parts

    def test_rcid_parts(
        self, teacher, make_student, dummy_loader, teacher_imprints,
    ) -> None:
        loss, parts = self._run_one_step(
            "rcid", teacher, make_student(), dummy_loader, teacher_imprints,
        )
        assert loss.isfinite()
        assert "kl_loss" in parts
        assert "aux_loss" in parts

    def test_informed_fitnets_parts(
        self, teacher, make_student, dummy_loader, teacher_imprints,
    ) -> None:
        loss, parts = self._run_one_step(
            "informed_fitnets", teacher, make_student(), dummy_loader, teacher_imprints,
        )
        assert loss.isfinite()
        assert "kl_loss" in parts
        assert "aux_loss" in parts

    def test_tinybert_parts(
        self, teacher, make_student, dummy_loader, teacher_imprints,
    ) -> None:
        loss, parts = self._run_one_step(
            "tinybert", teacher, make_student(), dummy_loader, teacher_imprints,
        )
        assert loss.isfinite()
        assert "loss_hidden" in parts
        assert "loss_attn" in parts
        assert "loss_kl" in parts

    def test_minilm_parts(
        self, teacher, make_student, dummy_loader, teacher_imprints,
    ) -> None:
        loss, parts = self._run_one_step(
            "minilm", teacher, make_student(), dummy_loader, teacher_imprints,
        )
        assert loss.isfinite()
        assert "loss_vr" in parts
        assert "loss_kl" in parts


# ======================================================================
# Test: 评估和检查点保存
# ======================================================================

class TestEvalAndCheckpoint:

    def test_evaluate_returns_float(
        self, teacher, make_student, dummy_loader,
    ) -> None:
        cfg = TrainConfig(method_name="standard_kd", epochs=1, lr=1e-3)
        loss_fn = _make_loss_fn("standard_kd")
        trainer = DistillationTrainer(
            teacher, make_student(), cfg, loss_fn, device="cpu",
        )
        val_loss = trainer.evaluate(dummy_loader)
        assert isinstance(val_loss, float)
        assert val_loss > 0

    def test_checkpoint_saved(
        self, teacher, make_student, dummy_loader, tmp_path,
    ) -> None:
        cfg = TrainConfig(
            method_name="standard_kd", epochs=1, lr=1e-3,
            save_dir=str(tmp_path), log_every=100,
        )
        loss_fn = _make_loss_fn("standard_kd")
        trainer = DistillationTrainer(
            teacher, make_student(), cfg, loss_fn, device="cpu",
        )
        trainer.train(dummy_loader)
        assert (tmp_path / "final.pt").exists()

    def test_val_triggers_best_checkpoint(
        self, teacher, make_student, dummy_loader, tmp_path,
    ) -> None:
        cfg = TrainConfig(
            method_name="standard_kd", epochs=1, lr=1e-3,
            save_dir=str(tmp_path), log_every=100,
        )
        loss_fn = _make_loss_fn("standard_kd")
        trainer = DistillationTrainer(
            teacher, make_student(), cfg, loss_fn, device="cpu",
        )
        trainer.train(dummy_loader, val_loader=dummy_loader)
        assert (tmp_path / "best.pt").exists()
        assert (tmp_path / "final.pt").exists()


# ======================================================================
# Test: 教师模型冻结验证
# ======================================================================

class TestTeacherFrozen:

    def test_teacher_no_grad(self, teacher, make_student, dummy_loader) -> None:
        cfg = TrainConfig(method_name="standard_kd", epochs=1, lr=1e-3, log_every=100)
        loss_fn = _make_loss_fn("standard_kd")
        trainer = DistillationTrainer(
            teacher, make_student(), cfg, loss_fn, device="cpu",
        )
        # 验证教师所有参数 requires_grad=False
        for p in trainer.teacher.parameters():
            assert not p.requires_grad
