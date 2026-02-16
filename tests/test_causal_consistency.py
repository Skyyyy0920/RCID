"""因果一致性评估模块单元测试。

测试覆盖：
- _pearson_r: 完美正相关、完美负相关、零相关、常数输入、短输入
- _gather_logit_delta: 正确计算 logit 变化量
- CausalConsistencyEvaluator:
    - evaluate() 返回正确的键和形状
    - 教师自身 patching 产生非零 delta（Δ_T ≠ 0）
    - 教师 vs 教师应得到高相关
    - 未训练的随机学生相关应 ~0
    - p-value 合理范围
- evaluate() 向后兼容接口
- GPT-2 端到端测试（slow）
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

from rcid.eval.causal_consistency import (
    CausalConsistencyEvaluator,
    _gather_logit_delta,
    _pearson_r,
    evaluate,
)


# ======================================================================
# 辅助函数
# ======================================================================

def _make_tiny_gpt2(
    n_layer: int = 4,
    n_embd: int = 32,
    n_head: int = 2,
    vocab_size: int = 100,
    seed: int = 42,
) -> GPT2LMHeadModel:
    """创建极小的 GPT-2 模型用于单元测试。"""
    torch.manual_seed(seed)
    config = GPT2Config(
        vocab_size=vocab_size,
        n_positions=64,
        n_embd=n_embd,
        n_layer=n_layer,
        n_head=n_head,
        n_inner=n_embd * 4,
    )
    return GPT2LMHeadModel(config)


def _make_fake_dataloader(
    n_samples: int = 20,
    seq_len: int = 10,
    vocab_size: int = 100,
    batch_size: int = 5,
    seed: int = 42,
) -> DataLoader:
    """创建假的 DataLoader，包含 clean_ids, corrupt_ids, answer_token_id。"""
    rng = torch.Generator().manual_seed(seed)
    clean_ids = torch.randint(1, vocab_size, (n_samples, seq_len), generator=rng)
    corrupt_ids = clean_ids.clone()
    # 在位置 3 做最小修改
    corrupt_ids[:, 3] = torch.randint(1, vocab_size, (n_samples,), generator=rng)
    answer_ids = torch.randint(0, vocab_size, (n_samples,), generator=rng)

    def collate(batch: list) -> dict[str, torch.Tensor]:
        c = torch.stack([b[0] for b in batch])
        x = torch.stack([b[1] for b in batch])
        a = torch.stack([b[2] for b in batch])
        return {"clean_ids": c, "corrupt_ids": x, "answer_token_id": a}

    ds = TensorDataset(clean_ids, corrupt_ids, answer_ids)
    return DataLoader(ds, batch_size=batch_size, collate_fn=collate)


# ======================================================================
# TestPearsonR
# ======================================================================

class TestPearsonR:
    """测试 _pearson_r 函数。"""

    def test_perfect_positive(self) -> None:
        x = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])
        y = torch.tensor([2.0, 4.0, 6.0, 8.0, 10.0])
        r, p = _pearson_r(x, y)
        assert abs(r - 1.0) < 1e-6, f"r={r}"
        assert p < 0.01, f"p={p}"

    def test_perfect_negative(self) -> None:
        x = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])
        y = torch.tensor([10.0, 8.0, 6.0, 4.0, 2.0])
        r, p = _pearson_r(x, y)
        assert abs(r - (-1.0)) < 1e-6, f"r={r}"
        assert p < 0.01, f"p={p}"

    def test_zero_correlation(self) -> None:
        """正交向量 → r ≈ 0。"""
        x = torch.tensor([1.0, 0.0, -1.0, 0.0])
        y = torch.tensor([0.0, 1.0, 0.0, -1.0])
        r, p = _pearson_r(x, y)
        assert abs(r) < 0.01, f"r={r}"
        assert p > 0.5, f"p={p}"

    def test_constant_input_returns_zero(self) -> None:
        """常数向量的 Pearson r 应返回 (0.0, 1.0)。"""
        x = torch.tensor([3.0, 3.0, 3.0, 3.0])
        y = torch.tensor([1.0, 2.0, 3.0, 4.0])
        r, p = _pearson_r(x, y)
        assert r == 0.0
        assert p == 1.0

    def test_short_input_returns_zero(self) -> None:
        """n < 3 应返回 (0.0, 1.0)。"""
        x = torch.tensor([1.0, 2.0])
        y = torch.tensor([3.0, 4.0])
        r, p = _pearson_r(x, y)
        assert r == 0.0
        assert p == 1.0

    def test_large_n_significant(self) -> None:
        """大样本量下线性关系应非常显著。"""
        torch.manual_seed(42)
        x = torch.randn(200)
        y = 0.5 * x + 0.1 * torch.randn(200)
        r, p = _pearson_r(x, y)
        assert r > 0.8, f"r={r}"
        assert p < 1e-10, f"p={p}"

    def test_moderate_correlation(self) -> None:
        """中等相关：r 在合理范围。"""
        torch.manual_seed(123)
        x = torch.randn(50)
        y = 0.3 * x + torch.randn(50)
        r, p = _pearson_r(x, y)
        assert -1.0 <= r <= 1.0, f"r={r}"
        assert 0.0 <= p <= 1.0, f"p={p}"


# ======================================================================
# TestGatherLogitDelta
# ======================================================================

class TestGatherLogitDelta:
    """测试 _gather_logit_delta 函数。"""

    def test_basic(self) -> None:
        base = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        patched = torch.tensor([[1.5, 2.0, 3.0], [4.0, 5.5, 6.0]])
        tokens = torch.tensor([0, 1])
        delta = _gather_logit_delta(base, patched, tokens)
        # patched - base: [0.5, 0.5]
        assert delta.shape == (2,)
        assert abs(delta[0].item() - 0.5) < 1e-6
        assert abs(delta[1].item() - 0.5) < 1e-6

    def test_negative_delta(self) -> None:
        base = torch.tensor([[5.0, 3.0]])
        patched = torch.tensor([[2.0, 3.0]])
        tokens = torch.tensor([0])
        delta = _gather_logit_delta(base, patched, tokens)
        assert delta.item() == pytest.approx(-3.0)


# ======================================================================
# TestEvaluatorConstruction
# ======================================================================

class TestEvaluatorConstruction:
    """测试 CausalConsistencyEvaluator 构造。"""

    def test_default_device(self) -> None:
        e = CausalConsistencyEvaluator()
        assert e.device is None

    def test_explicit_device(self) -> None:
        e = CausalConsistencyEvaluator(device="cpu")
        assert e.device == "cpu"


# ======================================================================
# TestEvaluatorEvaluate
# ======================================================================

class TestEvaluatorEvaluate:
    """测试 CausalConsistencyEvaluator.evaluate()。"""

    @pytest.fixture()
    def setup(self):
        """创建教师、学生和 DataLoader。"""
        teacher = _make_tiny_gpt2(n_layer=4, seed=42)
        student = _make_tiny_gpt2(n_layer=4, seed=99)
        dl = _make_fake_dataloader(n_samples=20, batch_size=5)
        checkpoints = [(0, 3), (2, 5)]
        layer_mapping = {0: 0, 2: 2}
        return teacher, student, dl, checkpoints, layer_mapping

    def test_return_keys(self, setup) -> None:
        teacher, student, dl, cps, lm = setup
        evaluator = CausalConsistencyEvaluator(device="cpu")
        result = evaluator.evaluate(teacher, student, dl, cps, lm)
        assert "per_checkpoint_correlation" in result
        assert "per_checkpoint_pvalue" in result
        assert "mean_correlation" in result
        assert "teacher_deltas" in result
        assert "student_deltas" in result

    def test_delta_shapes(self, setup) -> None:
        teacher, student, dl, cps, lm = setup
        evaluator = CausalConsistencyEvaluator(device="cpu")
        result = evaluator.evaluate(teacher, student, dl, cps, lm)
        n_cp = len(cps)
        n_samples = 20
        assert result["teacher_deltas"].shape == (n_cp, n_samples)
        assert result["student_deltas"].shape == (n_cp, n_samples)

    def test_correlation_range(self, setup) -> None:
        teacher, student, dl, cps, lm = setup
        evaluator = CausalConsistencyEvaluator(device="cpu")
        result = evaluator.evaluate(teacher, student, dl, cps, lm)
        for cp, r in result["per_checkpoint_correlation"].items():
            assert -1.0 <= r <= 1.0, f"CP {cp}: r={r}"

    def test_pvalue_range(self, setup) -> None:
        teacher, student, dl, cps, lm = setup
        evaluator = CausalConsistencyEvaluator(device="cpu")
        result = evaluator.evaluate(teacher, student, dl, cps, lm)
        for cp, p in result["per_checkpoint_pvalue"].items():
            assert 0.0 <= p <= 1.0, f"CP {cp}: p={p}"

    def test_mean_is_average(self, setup) -> None:
        teacher, student, dl, cps, lm = setup
        evaluator = CausalConsistencyEvaluator(device="cpu")
        result = evaluator.evaluate(teacher, student, dl, cps, lm)
        corrs = list(result["per_checkpoint_correlation"].values())
        expected_mean = sum(corrs) / len(corrs)
        assert abs(result["mean_correlation"] - expected_mean) < 1e-6

    def test_teacher_deltas_nonzero(self, setup) -> None:
        """教师在 clean 上 patching 应产生非零 delta。"""
        teacher, _, dl, cps, lm = setup
        evaluator = CausalConsistencyEvaluator(device="cpu")
        result = evaluator.evaluate(teacher, teacher, dl, cps, lm)
        t_delta = result["teacher_deltas"]
        # 至少某些 delta 应非零
        assert t_delta.abs().sum() > 0, "Teacher deltas all zero"

    def test_teacher_self_high_correlation(self, setup) -> None:
        """教师 vs 自身应得到 r=1.0。"""
        teacher, _, dl, cps, lm = setup
        evaluator = CausalConsistencyEvaluator(device="cpu")
        result = evaluator.evaluate(teacher, teacher, dl, cps, lm)
        # 教师与自身：delta 完全相同 → r = 1.0
        for cp, r in result["per_checkpoint_correlation"].items():
            assert abs(r - 1.0) < 1e-4, f"CP {cp}: r={r}, expected 1.0"


# ======================================================================
# TestBackwardCompatEvaluate
# ======================================================================

class TestBackwardCompatEvaluate:
    """测试向后兼容的 evaluate() 函数。"""

    def test_return_keys(self) -> None:
        teacher = _make_tiny_gpt2(n_layer=4, seed=42)
        student = _make_tiny_gpt2(n_layer=4, seed=99)
        dl = _make_fake_dataloader(n_samples=10, batch_size=5)
        cps = [(1, 3)]
        lm = {1: 1}
        result = evaluate(
            student, dl,
            teacher_model=teacher,
            checkpoints=cps,
            layer_mapping=lm,
            device="cpu",
        )
        assert "consistency" in result
        assert "consistency_per_cp" in result
        assert "n_checkpoints" in result

    def test_consistency_range(self) -> None:
        teacher = _make_tiny_gpt2(n_layer=4, seed=42)
        student = _make_tiny_gpt2(n_layer=4, seed=99)
        dl = _make_fake_dataloader(n_samples=10, batch_size=5)
        cps = [(1, 3)]
        lm = {1: 1}
        result = evaluate(
            student, dl,
            teacher_model=teacher,
            checkpoints=cps,
            layer_mapping=lm,
        )
        assert 0.0 <= result["consistency"] <= 1.0
        assert 0.0 <= result["consistency_per_cp"] <= 1.0

    def test_teacher_self_perfect_consistency(self) -> None:
        """教师 vs 自身应得到 consistency = 1.0。"""
        teacher = _make_tiny_gpt2(n_layer=4, seed=42)
        dl = _make_fake_dataloader(n_samples=10, batch_size=5)
        cps = [(0, 3), (2, 5)]
        lm = {0: 0, 2: 2}
        result = evaluate(
            teacher, dl,
            teacher_model=teacher,
            checkpoints=cps,
            layer_mapping=lm,
        )
        assert result["consistency"] == 1.0, (
            f"Teacher self-consistency should be 1.0, got {result['consistency']}"
        )

    def test_n_checkpoints(self) -> None:
        teacher = _make_tiny_gpt2(n_layer=4, seed=42)
        dl = _make_fake_dataloader(n_samples=10, batch_size=5)
        cps = [(0, 3), (2, 5), (3, 7)]
        lm = {0: 0, 2: 2, 3: 3}
        result = evaluate(
            teacher, dl,
            teacher_model=teacher,
            checkpoints=cps,
            layer_mapping=lm,
        )
        assert result["n_checkpoints"] == 3.0


# ======================================================================
# TestWithoutAnswerTokenId
# ======================================================================

class TestWithoutAnswerTokenId:
    """测试没有 answer_token_id 时的 fallback（用 argmax）。"""

    def test_no_answer_token_evaluator(self) -> None:
        teacher = _make_tiny_gpt2(n_layer=4, seed=42)
        student = _make_tiny_gpt2(n_layer=4, seed=99)

        rng = torch.Generator().manual_seed(42)
        clean = torch.randint(1, 100, (10, 8), generator=rng)
        corrupt = clean.clone()
        corrupt[:, 2] = torch.randint(1, 100, (10,), generator=rng)

        def collate(batch):
            c = torch.stack([b[0] for b in batch])
            x = torch.stack([b[1] for b in batch])
            return {"clean_ids": c, "corrupt_ids": x}

        ds = TensorDataset(clean, corrupt)
        dl = DataLoader(ds, batch_size=5, collate_fn=collate)

        evaluator = CausalConsistencyEvaluator(device="cpu")
        result = evaluator.evaluate(
            teacher, student, dl,
            checkpoints=[(1, 2)], layer_mapping={1: 1},
        )
        assert "mean_correlation" in result
        assert -1.0 <= result["mean_correlation"] <= 1.0

    def test_no_answer_token_evaluate_compat(self) -> None:
        teacher = _make_tiny_gpt2(n_layer=4, seed=42)

        rng = torch.Generator().manual_seed(42)
        clean = torch.randint(1, 100, (10, 8), generator=rng)
        corrupt = clean.clone()
        corrupt[:, 2] = torch.randint(1, 100, (10,), generator=rng)

        def collate(batch):
            c = torch.stack([b[0] for b in batch])
            x = torch.stack([b[1] for b in batch])
            return {"clean_ids": c, "corrupt_ids": x}

        ds = TensorDataset(clean, corrupt)
        dl = DataLoader(ds, batch_size=5, collate_fn=collate)

        result = evaluate(
            teacher, dl,
            teacher_model=teacher,
            checkpoints=[(1, 2)], layer_mapping={1: 1},
        )
        assert 0.0 <= result["consistency"] <= 1.0


# ======================================================================
# TestGPT2EndToEnd (slow)
# ======================================================================

@pytest.mark.slow
class TestGPT2EndToEnd:
    """端到端测试：加载真实 GPT-2。"""

    @pytest.fixture(scope="class")
    def gpt2_model(self):
        from transformers import GPT2LMHeadModel as GPT2
        model = GPT2.from_pretrained("gpt2")
        model.eval()
        return model

    @pytest.fixture(scope="class")
    def gpt2_dataloader(self):
        """创建简单的测试数据。"""
        rng = torch.Generator().manual_seed(42)
        clean = torch.randint(1000, 40000, (30, 15), generator=rng)
        corrupt = clean.clone()
        corrupt[:, 5] = torch.randint(1000, 40000, (30,), generator=rng)
        answer = torch.randint(1000, 40000, (30,), generator=rng)

        def collate(batch):
            c = torch.stack([b[0] for b in batch])
            x = torch.stack([b[1] for b in batch])
            a = torch.stack([b[2] for b in batch])
            return {"clean_ids": c, "corrupt_ids": x, "answer_token_id": a}

        ds = TensorDataset(clean, corrupt, answer)
        return DataLoader(ds, batch_size=10, collate_fn=collate)

    def test_teacher_self_correlation(
        self, gpt2_model, gpt2_dataloader,
    ) -> None:
        """GPT-2 vs 自身：r ≈ 1.0。"""
        evaluator = CausalConsistencyEvaluator(device="cpu")
        result = evaluator.evaluate(
            gpt2_model, gpt2_model, gpt2_dataloader,
            checkpoints=[(5, 5), (9, 5)],
            layer_mapping={5: 5, 9: 9},
        )
        for cp, r in result["per_checkpoint_correlation"].items():
            assert abs(r - 1.0) < 1e-4, f"CP {cp}: r={r}"

    def test_teacher_deltas_significant(
        self, gpt2_model, gpt2_dataloader,
    ) -> None:
        """GPT-2 patching 产生的 teacher delta 应非零。"""
        evaluator = CausalConsistencyEvaluator(device="cpu")
        result = evaluator.evaluate(
            gpt2_model, gpt2_model, gpt2_dataloader,
            checkpoints=[(5, 5)],
            layer_mapping={5: 5},
        )
        t_delta = result["teacher_deltas"]
        # 至少 80% 的 delta 非零
        nonzero_frac = (t_delta.abs() > 1e-6).float().mean().item()
        assert nonzero_frac > 0.8, f"Only {nonzero_frac:.1%} nonzero"

    def test_random_student_low_correlation(
        self, gpt2_model, gpt2_dataloader,
    ) -> None:
        """未训练随机学生 vs GPT-2：相关应较低（|r| < 0.7）。"""
        random_student = _make_tiny_gpt2(
            n_layer=12, n_embd=768, n_head=12,
            vocab_size=50257, seed=99,
        )
        evaluator = CausalConsistencyEvaluator(device="cpu")
        result = evaluator.evaluate(
            gpt2_model, random_student, gpt2_dataloader,
            checkpoints=[(5, 5)],
            layer_mapping={5: 5},
        )
        r = result["per_checkpoint_correlation"][(5, 5)]
        assert abs(r) < 0.7, f"Random student r={r}, expected low"

    def test_print_delta_stats(
        self, gpt2_model, gpt2_dataloader, capsys,
    ) -> None:
        """打印 delta 统计信息（不断言，仅输出）。"""
        evaluator = CausalConsistencyEvaluator(device="cpu")
        result = evaluator.evaluate(
            gpt2_model, gpt2_model, gpt2_dataloader,
            checkpoints=[(5, 5), (9, 5)],
            layer_mapping={5: 5, 9: 9},
        )
        print("\n--- Delta Statistics ---")
        for i, cp in enumerate([(5, 5), (9, 5)]):
            td = result["teacher_deltas"][i]
            print(
                f"CP {cp}: "
                f"mean={td.mean():.4f}, std={td.std():.4f}, "
                f"min={td.min():.4f}, max={td.max():.4f}, "
                f"r={result['per_checkpoint_correlation'][cp]:.4f}"
            )
