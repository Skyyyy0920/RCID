"""Induction Heads 数据集单元测试。"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from rcid.circuit.checkpoint_selection import collect_key_positions_induction
from rcid.data.induction import (
    SAFE_TOKEN_MIN,
    SAFE_TOKEN_MAX,
    SPECIAL_TOKEN_ID,
    InductionDataset,
    InductionSample,
)


# ======================================================================
# Fixtures
# ======================================================================

@pytest.fixture(scope="module")
def dataset() -> InductionDataset:
    return InductionDataset(n_samples=50, seed=42, seq_len=30)


@pytest.fixture(scope="module")
def batch(dataset: InductionDataset) -> dict[str, torch.Tensor]:
    loader = dataset.to_dataloader(batch_size=50, shuffle=False)
    return next(iter(loader))


# ======================================================================
# TestInductionSampleGeneration
# ======================================================================

class TestInductionSampleGeneration:
    """样本生成的正确性测试。"""

    def test_dataset_creation(self, dataset: InductionDataset) -> None:
        assert len(dataset) == 50

    def test_sample_fields(self, dataset: InductionDataset) -> None:
        sample = dataset[0]
        assert isinstance(sample, InductionSample)
        assert isinstance(sample.clean_text, str)
        assert isinstance(sample.corrupt_text, str)
        assert sample.clean_ids.dim() == 1
        assert sample.clean_ids.dtype == torch.long
        assert sample.corrupt_ids.dim() == 1
        assert sample.corrupt_ids.dtype == torch.long
        assert isinstance(sample.trigger_pos, int)
        assert isinstance(sample.target_pos, int)
        assert isinstance(sample.answer_token_id, int)

    def test_clean_corrupt_differ_at_one_position(
        self, dataset: InductionDataset,
    ) -> None:
        for sample in dataset.samples:
            diffs = (sample.clean_ids != sample.corrupt_ids).nonzero(
                as_tuple=True
            )[0]
            assert len(diffs) == 1, (
                f"Expected 1 diff, got {len(diffs)} at positions {diffs.tolist()}"
            )
            # 差异位置应是 second_a_pos = target_pos
            assert diffs[0].item() == sample.target_pos

    def test_a_appears_twice_in_clean(
        self, dataset: InductionDataset,
    ) -> None:
        for sample in dataset.samples:
            a_id = sample.clean_ids[sample.trigger_pos].item()
            a_count = (sample.clean_ids == a_id).sum().item()
            assert a_count == 2, f"A (id={a_id}) appears {a_count} times in clean"

    def test_a_appears_once_in_corrupt(
        self, dataset: InductionDataset,
    ) -> None:
        for sample in dataset.samples:
            a_id = sample.clean_ids[sample.trigger_pos].item()
            a_count = (sample.corrupt_ids == a_id).sum().item()
            assert a_count == 1, f"A (id={a_id}) appears {a_count} times in corrupt"

    def test_trigger_pair_intact(
        self, dataset: InductionDataset,
    ) -> None:
        for sample in dataset.samples:
            a_id = sample.clean_ids[sample.trigger_pos].item()
            b_id = sample.clean_ids[sample.trigger_pos + 1].item()
            assert b_id == sample.answer_token_id
            # Trigger pair also intact in corrupt
            assert sample.corrupt_ids[sample.trigger_pos].item() == a_id
            assert sample.corrupt_ids[sample.trigger_pos + 1].item() == b_id

    def test_target_pos_is_second_a(
        self, dataset: InductionDataset,
    ) -> None:
        for sample in dataset.samples:
            a_id = sample.clean_ids[sample.trigger_pos].item()
            # target_pos = second_a_pos (logits here predict next token = B)
            assert sample.clean_ids[sample.target_pos].item() == a_id
            # subseq_len >= 3, so gap >= 3
            assert sample.target_pos >= sample.trigger_pos + 3

    def test_answer_token_is_b(
        self, dataset: InductionDataset,
    ) -> None:
        for sample in dataset.samples:
            b_at_trigger = sample.clean_ids[sample.trigger_pos + 1].item()
            assert sample.answer_token_id == b_at_trigger

    def test_no_special_tokens(
        self, dataset: InductionDataset,
    ) -> None:
        for sample in dataset.samples:
            assert (sample.clean_ids != SPECIAL_TOKEN_ID).all()
            assert (sample.corrupt_ids != SPECIAL_TOKEN_ID).all()

    def test_all_tokens_in_safe_range(
        self, dataset: InductionDataset,
    ) -> None:
        for sample in dataset.samples:
            assert (sample.clean_ids >= SAFE_TOKEN_MIN).all()
            assert (sample.clean_ids < SAFE_TOKEN_MAX).all()

    def test_seed_reproducibility(self) -> None:
        ds1 = InductionDataset(n_samples=10, seed=123, seq_len=20)
        ds2 = InductionDataset(n_samples=10, seed=123, seq_len=20)
        for s1, s2 in zip(ds1.samples, ds2.samples):
            assert torch.equal(s1.clean_ids, s2.clean_ids)
            assert torch.equal(s1.corrupt_ids, s2.corrupt_ids)
            assert s1.trigger_pos == s2.trigger_pos
            assert s1.target_pos == s2.target_pos
            assert s1.answer_token_id == s2.answer_token_id

    def test_fixed_sequence_length(
        self, dataset: InductionDataset,
    ) -> None:
        for sample in dataset.samples:
            assert sample.clean_ids.shape[0] == 30
            assert sample.corrupt_ids.shape[0] == 30


# ======================================================================
# TestInductionCollate
# ======================================================================

class TestInductionCollate:
    """DataLoader / collate 函数测试。"""

    def test_batch_keys(self, batch: dict[str, torch.Tensor]) -> None:
        expected = {"clean_ids", "corrupt_ids", "trigger_pos",
                    "target_pos", "answer_token_id"}
        assert set(batch.keys()) == expected

    def test_batch_shapes(self, batch: dict[str, torch.Tensor]) -> None:
        B = 50
        seq_len = 30
        assert batch["clean_ids"].shape == (B, seq_len)
        assert batch["corrupt_ids"].shape == (B, seq_len)
        assert batch["trigger_pos"].shape == (B,)
        assert batch["target_pos"].shape == (B,)
        assert batch["answer_token_id"].shape == (B,)

    def test_fixed_length_no_zero_padding(
        self, batch: dict[str, torch.Tensor],
    ) -> None:
        # All tokens should be in safe range (no pad zeros)
        assert (batch["clean_ids"] > 0).all()
        assert (batch["corrupt_ids"] > 0).all()


# ======================================================================
# TestCheckpointSelection
# ======================================================================

class TestCheckpointSelection:
    """collect_key_positions_induction 测试。"""

    def test_collect_key_positions(
        self, batch: dict[str, torch.Tensor],
    ) -> None:
        positions = collect_key_positions_induction(batch)
        assert isinstance(positions, list)
        assert len(positions) > 0
        # Should contain trigger_pos and target_pos values
        trigger_vals = set(batch["trigger_pos"].tolist())
        target_vals = set(batch["target_pos"].tolist())
        for p in trigger_vals:
            assert p in positions
        for p in target_vals:
            assert p in positions

    def test_key_positions_sorted_and_deduped(
        self, batch: dict[str, torch.Tensor],
    ) -> None:
        positions = collect_key_positions_induction(batch)
        assert positions == sorted(set(positions))


# ======================================================================
# TestTeacherVerification — 需要加载 GPT-2
# ======================================================================

@pytest.mark.slow
class TestTeacherVerification:
    """用 GPT-2 教师模型验证数据集的因果结构。"""

    @pytest.fixture(scope="class")
    def teacher_and_data(self):
        from rcid.models.teacher import TeacherModel
        teacher = TeacherModel("gpt2")
        ds = InductionDataset(n_samples=50, seed=42, seq_len=30)
        loader = ds.to_dataloader(batch_size=50, shuffle=False)
        batch = next(iter(loader))
        return teacher, ds, batch

    def test_teacher_prefers_b_at_target(self, teacher_and_data) -> None:
        """clean 样本中，教师在 target_pos 对 B 的 logit 显著高于平均。"""
        teacher, ds, batch = teacher_and_data
        clean_ids = batch["clean_ids"].to(teacher.device)
        target_pos = batch["target_pos"]
        answer_ids = batch["answer_token_id"]

        with torch.no_grad():
            logits = teacher.model(clean_ids).logits  # (B, seq, V)

        b_logits = []
        mean_logits = []
        for i in range(clean_ids.shape[0]):
            tp = target_pos[i].item()
            logit_vec = logits[i, tp, :]                  # (V,)
            b_logits.append(logit_vec[answer_ids[i]].item())
            mean_logits.append(logit_vec.mean().item())

        # B 的 logit 应显著高于平均（至少对多数样本）
        above_mean = sum(1 for b, m in zip(b_logits, mean_logits) if b > m)
        ratio = above_mean / len(b_logits)
        assert ratio > 0.5, (
            f"Only {ratio:.0%} samples have B logit > mean logit"
        )

    def test_teacher_preference_reduced_in_corrupt(
        self, teacher_and_data,
    ) -> None:
        """corrupt 样本中，对 B 的偏好应大幅降低。"""
        teacher, ds, batch = teacher_and_data
        clean_ids = batch["clean_ids"].to(teacher.device)
        corrupt_ids = batch["corrupt_ids"].to(teacher.device)
        target_pos = batch["target_pos"]
        answer_ids = batch["answer_token_id"]

        with torch.no_grad():
            clean_logits = teacher.model(clean_ids).logits
            corrupt_logits = teacher.model(corrupt_ids).logits

        clean_b_ranks, corrupt_b_ranks = [], []
        for i in range(clean_ids.shape[0]):
            tp = target_pos[i].item()
            b_id = answer_ids[i].item()
            # Clean: B 的排名
            clean_rank = (clean_logits[i, tp, :] > clean_logits[i, tp, b_id]).sum().item()
            clean_b_ranks.append(clean_rank)
            # Corrupt: B 的排名
            corrupt_rank = (corrupt_logits[i, tp, :] > corrupt_logits[i, tp, b_id]).sum().item()
            corrupt_b_ranks.append(corrupt_rank)

        # B 在 corrupt 中排名应普遍更差（排名数值更大）
        mean_clean = sum(clean_b_ranks) / len(clean_b_ranks)
        mean_corrupt = sum(corrupt_b_ranks) / len(corrupt_b_ranks)
        assert mean_corrupt > mean_clean, (
            f"Corrupt mean rank ({mean_corrupt:.1f}) should be worse than "
            f"clean mean rank ({mean_clean:.1f})"
        )

    def test_teacher_accuracy(self, teacher_and_data) -> None:
        """教师在 clean 样本上 top-1 准确率 > 50%，top-5 > 70%。"""
        teacher, ds, batch = teacher_and_data
        clean_ids = batch["clean_ids"].to(teacher.device)
        target_pos = batch["target_pos"]
        answer_ids = batch["answer_token_id"]

        with torch.no_grad():
            logits = teacher.model(clean_ids).logits

        top1, top5 = 0, 0
        for i in range(clean_ids.shape[0]):
            tp = target_pos[i].item()
            b_id = answer_ids[i].item()
            rank = (logits[i, tp, :] > logits[i, tp, b_id]).sum().item()
            if rank == 0:
                top1 += 1
            if rank < 5:
                top5 += 1

        n = clean_ids.shape[0]
        acc1, acc5 = top1 / n, top5 / n
        print(f"\nTeacher induction: top-1={acc1:.0%}, top-5={acc5:.0%}")
        assert acc1 > 0.5, f"Top-1 accuracy {acc1:.1%} < 50%"
        assert acc5 > 0.7, f"Top-5 accuracy {acc5:.1%} < 70%"

    def test_print_samples(self, teacher_and_data) -> None:
        """打印 10 个样本展示格式（不断言，仅输出）。"""
        _, ds, _ = teacher_and_data
        print("\n" + "=" * 60)
        print("Induction Heads — Sample Display (10 samples)")
        print("=" * 60)
        for i, sample in enumerate(ds.samples[:10]):
            a_id = sample.clean_ids[sample.trigger_pos].item()
            b_id = sample.answer_token_id
            # Use ascii() to avoid Windows GBK encoding errors
            clean_snip = ascii(sample.clean_text[20:80])
            corrupt_snip = ascii(sample.corrupt_text[20:80])
            print(f"\n--- Sample {i+1} ---")
            print(f"  trigger_pos={sample.trigger_pos}, "
                  f"target_pos={sample.target_pos}")
            print(f"  A={a_id}, B={b_id} (answer)")
            print(f"  clean:   ...{clean_snip}...")
            print(f"  corrupt: ...{corrupt_snip}...")
