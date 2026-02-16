"""评估模块单元测试。

测试覆盖：
- task_accuracy: IOI 准确率 / Greater-Than 准确率
- causal_consistency: 教师-学生因果干预一致性
- perplexity: 困惑度计算（使用合成数据）
"""

from __future__ import annotations

import math

import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from transformers import GPT2Config, GPT2LMHeadModel


# ======================================================================
# 辅助函数
# ======================================================================

def _make_tiny_gpt2(
    n_layer: int = 2,
    n_embd: int = 32,
    n_head: int = 2,
    vocab_size: int = 100,
) -> GPT2LMHeadModel:
    """创建极小 GPT-2 模型用于测试。"""
    config = GPT2Config(
        vocab_size=vocab_size,
        n_positions=64,
        n_embd=n_embd,
        n_layer=n_layer,
        n_head=n_head,
        n_inner=n_embd * 4,
    )
    return GPT2LMHeadModel(config)


def _make_ioi_batch(
    batch_size: int = 4,
    seq_len: int = 10,
    vocab_size: int = 100,
    answer_token: int = 42,
) -> dict[str, torch.Tensor]:
    """构造 IOI 风格的 batch 字典。"""
    return {
        "clean_ids": torch.randint(1, vocab_size, (batch_size, seq_len)),
        "corrupt_ids": torch.randint(1, vocab_size, (batch_size, seq_len)),
        "io_token_pos": torch.zeros(batch_size, dtype=torch.long),
        "s2_token_pos": torch.ones(batch_size, dtype=torch.long),
        "answer_token_id": torch.full((batch_size,), answer_token, dtype=torch.long),
    }


def _make_gt_batch(
    batch_size: int = 4,
    seq_len: int = 10,
    vocab_size: int = 100,
) -> dict[str, torch.Tensor]:
    """构造 Greater-Than 风格的 batch 字典。"""
    return {
        "clean_ids": torch.randint(1, vocab_size, (batch_size, seq_len)),
        "corrupt_ids": torch.randint(1, vocab_size, (batch_size, seq_len)),
        "year_token_pos": torch.full((batch_size,), 5, dtype=torch.long),
        "clean_threshold": torch.full((batch_size,), 50, dtype=torch.long),
        "corrupt_threshold": torch.full((batch_size,), 80, dtype=torch.long),
    }


class _SingleBatchLoader:
    """包装单个 batch 为可迭代 DataLoader 替代品。"""

    def __init__(self, batch: dict[str, torch.Tensor]) -> None:
        self.batch = batch

    def __iter__(self):
        yield self.batch


# ======================================================================
# task_accuracy 测试
# ======================================================================

class TestTaskAccuracyIOI:
    """IOI 任务准确率测试。"""

    def test_perfect_accuracy(self) -> None:
        """当模型 argmax 恰好命中 answer_token 时，准确率为 1。"""
        from rcid.eval.task_accuracy import evaluate

        model = _make_tiny_gpt2()
        model.eval()

        batch = _make_ioi_batch(batch_size=4, answer_token=42)

        # 劫持 lm_head bias 使 token 42 的 logit 极大
        with torch.no_grad():
            model.lm_head.bias = nn.Parameter(
                torch.zeros(100), requires_grad=False,
            )
            model.lm_head.bias[42] = 1000.0

        loader = _SingleBatchLoader(batch)
        result = evaluate(model, loader, task="ioi")

        assert "accuracy" in result
        assert result["accuracy"] == 1.0

    def test_zero_accuracy(self) -> None:
        """当模型 argmax 永远不命中 answer_token 时，准确率为 0。"""
        from rcid.eval.task_accuracy import evaluate

        model = _make_tiny_gpt2()
        model.eval()

        batch = _make_ioi_batch(batch_size=4, answer_token=42)

        # 使 token 42 的 logit 极小
        with torch.no_grad():
            model.lm_head.bias = nn.Parameter(
                torch.zeros(100), requires_grad=False,
            )
            model.lm_head.bias[42] = -1000.0
            model.lm_head.bias[0] = 1000.0  # token 0 wins

        loader = _SingleBatchLoader(batch)
        result = evaluate(model, loader, task="ioi")

        assert result["accuracy"] == 0.0

    def test_accuracy_range(self) -> None:
        """准确率在 [0, 1] 范围内。"""
        from rcid.eval.task_accuracy import evaluate

        model = _make_tiny_gpt2()
        model.eval()

        batch = _make_ioi_batch(batch_size=8)
        loader = _SingleBatchLoader(batch)
        result = evaluate(model, loader, task="ioi")

        assert 0.0 <= result["accuracy"] <= 1.0

    def test_invalid_task(self) -> None:
        """未知任务类型应 raise AssertionError。"""
        from rcid.eval.task_accuracy import evaluate

        model = _make_tiny_gpt2()
        batch = _make_ioi_batch()
        loader = _SingleBatchLoader(batch)

        with pytest.raises(AssertionError, match="Unknown task"):
            evaluate(model, loader, task="nonexistent_task")


class TestTaskAccuracyGT:
    """Greater-Than 任务准确率测试。"""

    def test_accuracy_range(self) -> None:
        """准确率在 [0, 1] 范围内。"""
        from rcid.eval.task_accuracy import evaluate

        # 需要 vocab_size=50257 以匹配 GPT-2 tokenizer
        model = _make_tiny_gpt2(vocab_size=50257)
        model.eval()

        batch = _make_gt_batch(batch_size=4, vocab_size=50257)
        loader = _SingleBatchLoader(batch)
        result = evaluate(model, loader, task="greater_than")

        assert 0.0 <= result["accuracy"] <= 1.0

    def test_return_keys(self) -> None:
        """返回字典包含 accuracy 键。"""
        from rcid.eval.task_accuracy import evaluate

        model = _make_tiny_gpt2(vocab_size=50257)
        model.eval()

        batch = _make_gt_batch(batch_size=2, vocab_size=50257)
        loader = _SingleBatchLoader(batch)
        result = evaluate(model, loader, task="greater_than")

        assert "accuracy" in result


# ======================================================================
# causal_consistency 测试
# ======================================================================

class TestCausalConsistency:
    """因果干预一致性测试。"""

    def test_same_model_perfect_consistency(self) -> None:
        """教师=学生时，因果一致性应为 1.0。"""
        from rcid.eval.causal_consistency import evaluate

        model = _make_tiny_gpt2(n_layer=4)
        model.eval()

        batch = _make_ioi_batch(batch_size=2, seq_len=8)
        loader = _SingleBatchLoader(batch)

        checkpoints = [(0, 7), (1, 7)]
        layer_mapping = {0: 0, 1: 1}

        result = evaluate(
            model, loader,
            teacher_model=model,
            checkpoints=checkpoints,
            layer_mapping=layer_mapping,
        )

        assert result["consistency"] == 1.0
        assert result["consistency_per_cp"] == 1.0
        assert result["n_checkpoints"] == 2.0

    def test_return_keys(self) -> None:
        """返回字典包含所有必要键。"""
        from rcid.eval.causal_consistency import evaluate

        teacher = _make_tiny_gpt2(n_layer=4)
        student = _make_tiny_gpt2(n_layer=4)
        teacher.eval()
        student.eval()

        batch = _make_ioi_batch(batch_size=2, seq_len=8)
        loader = _SingleBatchLoader(batch)

        checkpoints = [(0, 7)]
        layer_mapping = {0: 0}

        result = evaluate(
            student, loader,
            teacher_model=teacher,
            checkpoints=checkpoints,
            layer_mapping=layer_mapping,
        )

        assert "consistency" in result
        assert "consistency_per_cp" in result
        assert "n_checkpoints" in result

    def test_consistency_range(self) -> None:
        """一致性在 [0, 1] 范围内。"""
        from rcid.eval.causal_consistency import evaluate

        teacher = _make_tiny_gpt2(n_layer=4)
        student = _make_tiny_gpt2(n_layer=4)
        teacher.eval()
        student.eval()

        batch = _make_ioi_batch(batch_size=4, seq_len=8)
        loader = _SingleBatchLoader(batch)

        checkpoints = [(0, 7), (2, 7)]
        layer_mapping = {0: 0, 2: 2}

        result = evaluate(
            student, loader,
            teacher_model=teacher,
            checkpoints=checkpoints,
            layer_mapping=layer_mapping,
        )

        assert 0.0 <= result["consistency"] <= 1.0
        assert 0.0 <= result["consistency_per_cp"] <= 1.0

    def test_no_answer_id_fallback(self) -> None:
        """当 batch 无 answer_token_id 时，使用 argmax 回退。"""
        from rcid.eval.causal_consistency import evaluate

        model = _make_tiny_gpt2(n_layer=4)
        model.eval()

        # 创建无 answer_token_id 的 batch
        batch = {
            "clean_ids": torch.randint(1, 100, (2, 8)),
            "corrupt_ids": torch.randint(1, 100, (2, 8)),
        }
        loader = _SingleBatchLoader(batch)

        checkpoints = [(0, 7)]
        layer_mapping = {0: 0}

        result = evaluate(
            model, loader,
            teacher_model=model,
            checkpoints=checkpoints,
            layer_mapping=layer_mapping,
        )

        # 同模型 + argmax 回退 → 应为 1.0
        assert result["consistency"] == 1.0


# ======================================================================
# perplexity 测试
# ======================================================================

class TestPerplexity:
    """困惑度测试。"""

    def test_perplexity_from_dataloader(self) -> None:
        """从 DataLoader 计算困惑度应返回合理值。"""
        from rcid.eval.perplexity import evaluate

        model = _make_tiny_gpt2(vocab_size=100)
        model.eval()

        # 构造简单的 input/label batch
        batch = {
            "input_ids": torch.randint(0, 100, (2, 16)),
            "labels": torch.randint(0, 100, (2, 16)),
        }
        loader = _SingleBatchLoader(batch)

        result = evaluate(model, loader)

        assert "perplexity" in result
        assert "avg_loss" in result
        assert "n_tokens" in result
        assert result["perplexity"] > 0.0
        assert result["n_tokens"] == 32.0  # 2 * 16

    def test_perplexity_clean_ids_fallback(self) -> None:
        """当 batch 含 clean_ids 时，使用 autoregressive 方式。"""
        from rcid.eval.perplexity import evaluate

        model = _make_tiny_gpt2(vocab_size=100)
        model.eval()

        batch = {
            "clean_ids": torch.randint(0, 100, (2, 16)),
        }
        loader = _SingleBatchLoader(batch)

        result = evaluate(model, loader)

        assert result["perplexity"] > 0.0
        # autoregressive: seq_len-1 = 15 tokens per sample
        assert result["n_tokens"] == 30.0  # 2 * 15

    def test_perplexity_positive(self) -> None:
        """困惑度必须 > 1（对于随机模型通常远大于 1）。"""
        from rcid.eval.perplexity import evaluate

        model = _make_tiny_gpt2(vocab_size=100)
        model.eval()

        batch = {
            "input_ids": torch.randint(0, 100, (4, 32)),
            "labels": torch.randint(0, 100, (4, 32)),
        }
        loader = _SingleBatchLoader(batch)

        result = evaluate(model, loader)

        # 随机模型对 100-token 词表：perplexity ≈ 100
        assert result["perplexity"] > 1.0

    def test_uniform_model_perplexity(self) -> None:
        """均匀分布模型的困惑度应接近词表大小。"""
        from rcid.eval.perplexity import evaluate

        vocab_size = 50
        model = _make_tiny_gpt2(vocab_size=vocab_size)
        model.eval()

        # 将 lm_head 输出设为均匀分布
        with torch.no_grad():
            model.lm_head.weight.zero_()
            model.lm_head.bias = nn.Parameter(
                torch.zeros(vocab_size), requires_grad=False,
            )

        batch = {
            "input_ids": torch.randint(0, vocab_size, (2, 32)),
            "labels": torch.randint(0, vocab_size, (2, 32)),
        }
        loader = _SingleBatchLoader(batch)

        result = evaluate(model, loader)

        # 均匀分布 → perplexity ≈ vocab_size
        assert abs(result["perplexity"] - vocab_size) < vocab_size * 0.3

    def test_text_chunk_dataset(self) -> None:
        """TextChunkDataset 正确切分 token 序列。"""
        from rcid.eval.perplexity import TextChunkDataset

        tokens = list(range(100))
        ds = TextChunkDataset(tokens, chunk_size=32)

        assert len(ds) > 0

        sample = ds[0]
        assert "input_ids" in sample
        assert "labels" in sample
        assert sample["input_ids"].shape[0] == sample["labels"].shape[0]
        # input 和 label 应该偏移 1
        assert torch.equal(sample["labels"][:-1], sample["input_ids"][1:])

    def test_missing_keys_raises(self) -> None:
        """batch 缺少必要键应 raise KeyError。"""
        from rcid.eval.perplexity import evaluate

        model = _make_tiny_gpt2(vocab_size=100)
        model.eval()

        batch = {"random_key": torch.randn(2, 8)}
        loader = _SingleBatchLoader(batch)

        with pytest.raises(KeyError, match="input_ids"):
            evaluate(model, loader)


# ======================================================================
# patched_logits 内部函数测试
# ======================================================================

class TestPatchingInternals:
    """activation patching 内部函数测试。"""

    def test_patched_logits_differs(self) -> None:
        """patch 后的 logits 应与基线不同。"""
        from rcid.eval.causal_consistency import _patched_logits

        model = _make_tiny_gpt2(n_layer=4)
        model.eval()

        clean = torch.randint(1, 100, (2, 8))
        corrupt = torch.randint(1, 100, (2, 8))

        with torch.no_grad():
            base_logits = model(clean).logits         # (2, 8, 100)
            patched = _patched_logits(
                model, clean, corrupt, layer=1, token_pos=5,
            )  # (2, 8, 100)

        # 如果 clean ≠ corrupt，patched 应 ≠ base
        if not torch.equal(clean, corrupt):
            assert not torch.allclose(base_logits, patched, atol=1e-5)

    def test_same_input_no_change(self) -> None:
        """clean = corrupt 时，patch 不应改变 logits。"""
        from rcid.eval.causal_consistency import _patched_logits

        model = _make_tiny_gpt2(n_layer=4)
        model.eval()

        ids = torch.randint(1, 100, (2, 8))

        with torch.no_grad():
            base_logits = model(ids).logits
            patched = _patched_logits(
                model, ids, ids, layer=0, token_pos=3,
            )

        assert torch.allclose(base_logits, patched, atol=1e-5)

    def test_gather_logit_delta(self) -> None:
        """_gather_logit_delta 正确计算 logit 差值。"""
        from rcid.eval.causal_consistency import _gather_logit_delta

        base = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        patched = torch.tensor([[1.5, 2.0, 3.0], [4.0, 5.0, 7.0]])
        tokens = torch.tensor([0, 2])

        delta = _gather_logit_delta(base, patched, tokens)

        assert torch.allclose(delta, torch.tensor([0.5, 1.0]))
