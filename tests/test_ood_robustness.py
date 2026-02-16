"""OOD 鲁棒性模块单元测试。

测试覆盖：
- OODTestGenerator: 为 4 个任务生成 OOD 变体
- 每种变体生成有效的 DataLoader（正确 batch 键和形状）
- OOD 数据与 ID 数据不同但格式兼容
- compute_accuracy: 4 种任务的准确率计算
- RobustnessEvaluator: 返回正确键、degradation 范围
- GPT-2 端到端（slow）: 教师 ID > OOD、随机学生 ≈ chance
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch
from torch import nn
from transformers import GPT2Config, GPT2LMHeadModel, GPT2Tokenizer

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from rcid.eval.ood_generators import (
    OODTestGenerator,
    make_gt_ood_datasets,
    make_induction_ood_datasets,
    make_ioi_ood_datasets,
    make_sva_ood_datasets,
)
from rcid.eval.ood_robustness import RobustnessEvaluator, compute_accuracy


# ======================================================================
# 辅助函数
# ======================================================================

def _make_tiny_gpt2(
    n_layer: int = 2,
    n_embd: int = 32,
    n_head: int = 2,
    vocab_size: int = 50257,
    seed: int = 42,
) -> GPT2LMHeadModel:
    torch.manual_seed(seed)
    config = GPT2Config(
        vocab_size=vocab_size,
        n_positions=128,
        n_embd=n_embd,
        n_layer=n_layer,
        n_head=n_head,
        n_inner=n_embd * 4,
    )
    return GPT2LMHeadModel(config)


@pytest.fixture(scope="module")
def tokenizer() -> GPT2Tokenizer:
    return GPT2Tokenizer.from_pretrained("gpt2")


# ======================================================================
# TestOODGeneratorIOI
# ======================================================================

class TestOODGeneratorIOI:
    """IOI OOD 变体测试。"""

    def test_generates_two_variants(self, tokenizer) -> None:
        ood = make_ioi_ood_datasets(n_samples=10, tokenizer=tokenizer)
        assert "rare_names" in ood
        assert "new_templates" in ood

    def test_batch_keys(self, tokenizer) -> None:
        ood = make_ioi_ood_datasets(n_samples=10, tokenizer=tokenizer)
        for name, dl in ood.items():
            batch = next(iter(dl))
            assert "clean_ids" in batch, f"{name}: missing clean_ids"
            assert "corrupt_ids" in batch, f"{name}: missing corrupt_ids"
            assert "answer_token_id" in batch, f"{name}: missing answer_token_id"

    def test_batch_shapes(self, tokenizer) -> None:
        ood = make_ioi_ood_datasets(
            n_samples=10, tokenizer=tokenizer, seed=42,
        )
        for name, dl in ood.items():
            batch = next(iter(dl))
            B = batch["clean_ids"].shape[0]
            assert B <= 32, f"{name}: batch_size={B}"
            assert batch["clean_ids"].dim() == 2
            assert batch["answer_token_id"].shape == (B,)


# ======================================================================
# TestOODGeneratorGT
# ======================================================================

class TestOODGeneratorGT:
    """Greater-Than OOD 变体测试。"""

    def test_generates_two_variants(self, tokenizer) -> None:
        ood = make_gt_ood_datasets(n_samples=10, tokenizer=tokenizer)
        assert "alt_year_range" in ood
        assert "new_templates" in ood

    def test_batch_keys(self, tokenizer) -> None:
        ood = make_gt_ood_datasets(n_samples=10, tokenizer=tokenizer)
        for name, dl in ood.items():
            batch = next(iter(dl))
            assert "clean_ids" in batch
            assert "clean_threshold" in batch


# ======================================================================
# TestOODGeneratorInduction
# ======================================================================

class TestOODGeneratorInduction:
    """Induction Heads OOD 变体测试。"""

    def test_generates_two_variants(self, tokenizer) -> None:
        ood = make_induction_ood_datasets(
            n_samples=10, tokenizer=tokenizer,
        )
        assert "long_seq" in ood
        assert "very_long_seq" in ood

    def test_longer_sequences(self, tokenizer) -> None:
        ood = make_induction_ood_datasets(
            n_samples=10, tokenizer=tokenizer,
        )
        batch_long = next(iter(ood["long_seq"]))
        batch_vlong = next(iter(ood["very_long_seq"]))
        assert batch_long["clean_ids"].shape[1] == 60
        assert batch_vlong["clean_ids"].shape[1] == 90

    def test_batch_keys(self, tokenizer) -> None:
        ood = make_induction_ood_datasets(
            n_samples=10, tokenizer=tokenizer,
        )
        for name, dl in ood.items():
            batch = next(iter(dl))
            assert "clean_ids" in batch
            assert "target_pos" in batch
            assert "answer_token_id" in batch


# ======================================================================
# TestOODGeneratorSVA
# ======================================================================

class TestOODGeneratorSVA:
    """SVA OOD 变体测试。"""

    def test_generates_two_variants(self, tokenizer) -> None:
        ood = make_sva_ood_datasets(n_samples=10, tokenizer=tokenizer)
        assert "multi_attractor" in ood
        assert "rare_nouns" in ood

    def test_batch_keys(self, tokenizer) -> None:
        ood = make_sva_ood_datasets(n_samples=10, tokenizer=tokenizer)
        for name, dl in ood.items():
            batch = next(iter(dl))
            assert "clean_ids" in batch
            assert "singular_verb_ids" in batch
            assert "plural_verb_ids" in batch

    def test_multi_attractor_longer(self, tokenizer) -> None:
        """multi_attractor 变体的序列应比基础 SVA 长。"""
        from rcid.data.sva import SVADataset
        base = SVADataset(
            n_samples=10, tokenizer=tokenizer, seed=42, n_attractors=0,
        )
        ood = make_sva_ood_datasets(n_samples=10, tokenizer=tokenizer)
        batch_base = next(iter(base.to_dataloader(32, shuffle=False)))
        batch_multi = next(iter(ood["multi_attractor"]))
        assert batch_multi["clean_ids"].shape[1] > batch_base["clean_ids"].shape[1]


# ======================================================================
# TestOODTestGeneratorUnified
# ======================================================================

class TestOODTestGeneratorUnified:
    """OODTestGenerator 统一接口测试。"""

    def test_all_tasks(self, tokenizer) -> None:
        gen = OODTestGenerator(
            n_samples=5, tokenizer=tokenizer, seed=42,
        )
        for task in ("ioi", "greater_than", "induction", "sva"):
            ood = gen.generate(task)
            assert len(ood) >= 2, f"Task {task}: only {len(ood)} variants"

    def test_unknown_task_raises(self, tokenizer) -> None:
        gen = OODTestGenerator(n_samples=5, tokenizer=tokenizer)
        with pytest.raises(AssertionError, match="Unknown task"):
            gen.generate("nonexistent_task")

    def test_deterministic(self, tokenizer) -> None:
        """相同 seed 产生相同 OOD 数据。"""
        gen1 = OODTestGenerator(n_samples=5, tokenizer=tokenizer, seed=99)
        gen2 = OODTestGenerator(n_samples=5, tokenizer=tokenizer, seed=99)
        ood1 = gen1.generate("induction")
        ood2 = gen2.generate("induction")
        b1 = next(iter(ood1["long_seq"]))
        b2 = next(iter(ood2["long_seq"]))
        assert torch.equal(b1["clean_ids"], b2["clean_ids"])


# ======================================================================
# TestComputeAccuracy
# ======================================================================

class TestComputeAccuracy:
    """compute_accuracy 函数测试（用 tiny 模型）。"""

    def test_ioi_returns_valid_range(self, tokenizer) -> None:
        model = _make_tiny_gpt2()
        from rcid.data.ioi import IOIDataset
        ds = IOIDataset(n_samples=10, tokenizer=tokenizer, seed=42)
        dl = ds.to_dataloader(batch_size=10, shuffle=False)
        acc = compute_accuracy(model, dl, "ioi", "cpu")
        assert 0.0 <= acc <= 1.0

    def test_induction_returns_valid_range(self, tokenizer) -> None:
        model = _make_tiny_gpt2()
        from rcid.data.induction import InductionDataset
        ds = InductionDataset(
            n_samples=10, tokenizer=tokenizer, seed=42, seq_len=20,
        )
        dl = ds.to_dataloader(batch_size=10, shuffle=False)
        acc = compute_accuracy(model, dl, "induction", "cpu")
        assert 0.0 <= acc <= 1.0

    def test_sva_returns_valid_range(self, tokenizer) -> None:
        model = _make_tiny_gpt2()
        from rcid.data.sva import SVADataset
        ds = SVADataset(n_samples=10, tokenizer=tokenizer, seed=42)
        dl = ds.to_dataloader(batch_size=10, shuffle=False)
        acc = compute_accuracy(model, dl, "sva", "cpu")
        assert 0.0 <= acc <= 1.0

    def test_gt_returns_valid_range(self, tokenizer) -> None:
        model = _make_tiny_gpt2()
        from rcid.data.greater_than import GreaterThanDataset
        ds = GreaterThanDataset(
            n_samples=10, tokenizer=tokenizer, seed=42,
        )
        dl = ds.to_dataloader(batch_size=10, shuffle=False)
        acc = compute_accuracy(model, dl, "greater_than", "cpu")
        assert 0.0 <= acc <= 1.0

    def test_unknown_task_raises(self, tokenizer) -> None:
        model = _make_tiny_gpt2()
        from rcid.data.ioi import IOIDataset
        ds = IOIDataset(n_samples=5, tokenizer=tokenizer, seed=42)
        dl = ds.to_dataloader(batch_size=5, shuffle=False)
        with pytest.raises(ValueError, match="Unknown task"):
            compute_accuracy(model, dl, "bogus", "cpu")


# ======================================================================
# TestRobustnessEvaluator
# ======================================================================

class TestRobustnessEvaluator:
    """RobustnessEvaluator 测试。"""

    def test_return_keys(self, tokenizer) -> None:
        model = _make_tiny_gpt2()
        from rcid.data.ioi import IOIDataset
        id_ds = IOIDataset(
            n_samples=10, tokenizer=tokenizer, seed=42,
        ).to_dataloader(10, shuffle=False)
        ood = make_ioi_ood_datasets(
            n_samples=10, tokenizer=tokenizer, seed=7777,
        )
        evaluator = RobustnessEvaluator(device="cpu")
        result = evaluator.evaluate(model, id_ds, ood, task="ioi")
        assert "id_accuracy" in result
        assert "ood_accuracy" in result
        assert "degradation" in result
        assert "mean_degradation" in result

    def test_ood_accuracy_keys_match(self, tokenizer) -> None:
        model = _make_tiny_gpt2()
        from rcid.data.ioi import IOIDataset
        id_ds = IOIDataset(
            n_samples=10, tokenizer=tokenizer, seed=42,
        ).to_dataloader(10, shuffle=False)
        ood = make_ioi_ood_datasets(
            n_samples=10, tokenizer=tokenizer,
        )
        evaluator = RobustnessEvaluator(device="cpu")
        result = evaluator.evaluate(model, id_ds, ood, task="ioi")
        assert set(result["ood_accuracy"].keys()) == set(ood.keys())
        assert set(result["degradation"].keys()) == set(ood.keys())

    def test_degradation_range(self, tokenizer) -> None:
        """degradation 应在合理范围（允许负值 = OOD 更好）。"""
        model = _make_tiny_gpt2()
        from rcid.data.induction import InductionDataset
        id_ds = InductionDataset(
            n_samples=10, tokenizer=tokenizer, seed=42, seq_len=20,
        ).to_dataloader(10, shuffle=False)
        ood = make_induction_ood_datasets(
            n_samples=10, tokenizer=tokenizer,
        )
        evaluator = RobustnessEvaluator(device="cpu")
        result = evaluator.evaluate(model, id_ds, ood, task="induction")
        for name, deg in result["degradation"].items():
            # degradation 不应超过 ±2（极端情况）
            assert -2.0 <= deg <= 2.0, f"{name}: deg={deg}"

    def test_mean_degradation_is_average(self, tokenizer) -> None:
        model = _make_tiny_gpt2()
        from rcid.data.sva import SVADataset
        id_ds = SVADataset(
            n_samples=10, tokenizer=tokenizer, seed=42,
        ).to_dataloader(10, shuffle=False)
        ood = make_sva_ood_datasets(
            n_samples=10, tokenizer=tokenizer,
        )
        evaluator = RobustnessEvaluator(device="cpu")
        result = evaluator.evaluate(model, id_ds, ood, task="sva")
        degs = list(result["degradation"].values())
        expected = sum(degs) / len(degs)
        assert abs(result["mean_degradation"] - expected) < 1e-6


# ======================================================================
# TestGPT2EndToEnd (slow)
# ======================================================================

@pytest.mark.slow
class TestGPT2EndToEnd:
    """端到端测试：真实 GPT-2 模型。"""

    @pytest.fixture(scope="class")
    def gpt2(self):
        model = GPT2LMHeadModel.from_pretrained("gpt2")
        model.eval()
        return model

    @pytest.fixture(scope="class")
    def tok(self) -> GPT2Tokenizer:
        return GPT2Tokenizer.from_pretrained("gpt2")

    def test_ioi_teacher_ood_performance(self, gpt2, tok) -> None:
        """GPT-2 在 IOI OOD 上准确率应 > 0 但可能低于 ID。"""
        from rcid.data.ioi import IOIDataset
        id_ds = IOIDataset(
            n_samples=50, tokenizer=tok, seed=42,
        ).to_dataloader(25, shuffle=False)
        ood = make_ioi_ood_datasets(n_samples=50, tokenizer=tok)

        evaluator = RobustnessEvaluator(device="cpu")
        result = evaluator.evaluate(gpt2, id_ds, ood, task="ioi")

        print(f"\nIOI ID accuracy: {result['id_accuracy']:.4f}")
        for name, acc in result["ood_accuracy"].items():
            print(f"IOI OOD {name}: {acc:.4f} (deg={result['degradation'][name]:.4f})")

        # 教师 ID 准确率应 > 0.5
        assert result["id_accuracy"] > 0.5, (
            f"GPT-2 IOI ID acc = {result['id_accuracy']:.4f}"
        )
        # OOD 准确率应 > 0（模型不是完全失效）
        for name, acc in result["ood_accuracy"].items():
            assert acc > 0.0, f"OOD {name} acc = 0"

    def test_induction_teacher_ood_performance(self, gpt2, tok) -> None:
        """GPT-2 在 Induction OOD 上应仍有一定准确率。"""
        from rcid.data.induction import InductionDataset
        id_ds = InductionDataset(
            n_samples=50, tokenizer=tok, seed=42, seq_len=30,
        ).to_dataloader(25, shuffle=False)
        ood = make_induction_ood_datasets(n_samples=50, tokenizer=tok)

        evaluator = RobustnessEvaluator(device="cpu")
        result = evaluator.evaluate(gpt2, id_ds, ood, task="induction")

        print(f"\nInduction ID accuracy: {result['id_accuracy']:.4f}")
        for name, acc in result["ood_accuracy"].items():
            print(f"Induction OOD {name}: {acc:.4f}")

        # Induction ID: 应 > 40% (top-1)
        assert result["id_accuracy"] > 0.4
        # OOD long_seq: 仍应 > 20%
        assert result["ood_accuracy"]["long_seq"] > 0.2

    def test_random_student_near_chance(self, gpt2, tok) -> None:
        """未训练学生在 IOI ID 和 OOD 上都应接近随机。"""
        random_student = _make_tiny_gpt2(
            n_layer=2, n_embd=32, n_head=2,
            vocab_size=50257, seed=99,
        )
        from rcid.data.ioi import IOIDataset
        id_ds = IOIDataset(
            n_samples=50, tokenizer=tok, seed=42,
        ).to_dataloader(25, shuffle=False)
        ood = make_ioi_ood_datasets(n_samples=50, tokenizer=tok)

        evaluator = RobustnessEvaluator(device="cpu")
        result = evaluator.evaluate(random_student, id_ds, ood, task="ioi")

        print(f"\nRandom student IOI ID: {result['id_accuracy']:.4f}")
        for name, acc in result["ood_accuracy"].items():
            print(f"Random student IOI OOD {name}: {acc:.4f}")

        # Random model: ID 准确率应 < 20%（24 个名字，chance ~4%）
        assert result["id_accuracy"] < 0.2

    def test_sva_teacher_ood(self, gpt2, tok) -> None:
        """GPT-2 在 SVA OOD 上应保持一定性能。"""
        from rcid.data.sva import SVADataset
        id_ds = SVADataset(
            n_samples=50, tokenizer=tok, seed=42,
        ).to_dataloader(25, shuffle=False)
        ood = make_sva_ood_datasets(n_samples=50, tokenizer=tok)

        evaluator = RobustnessEvaluator(device="cpu")
        result = evaluator.evaluate(gpt2, id_ds, ood, task="sva")

        print(f"\nSVA ID accuracy: {result['id_accuracy']:.4f}")
        for name, acc in result["ood_accuracy"].items():
            print(f"SVA OOD {name}: {acc:.4f} (deg={result['degradation'][name]:.4f})")

        # GPT-2 SVA ID > 60%
        assert result["id_accuracy"] > 0.6
        # OOD 不应完全失效
        for name, acc in result["ood_accuracy"].items():
            assert acc > 0.3, f"SVA OOD {name} = {acc:.4f}"
