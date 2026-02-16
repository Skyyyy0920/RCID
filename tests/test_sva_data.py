"""Subject-Verb Agreement 数据集单元测试。"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from rcid.circuit.checkpoint_selection import collect_key_positions_sva
from rcid.data.sva import (
    NOUN_PAIRS,
    PP_TEMPLATES,
    VERB_PAIRS,
    SVADataset,
    SVASample,
)


# ======================================================================
# Fixtures
# ======================================================================

@pytest.fixture(scope="module")
def dataset() -> SVADataset:
    return SVADataset(n_samples=50, seed=42)


@pytest.fixture(scope="module")
def batch(dataset: SVADataset) -> dict[str, torch.Tensor]:
    loader = dataset.to_dataloader(batch_size=50, shuffle=False)
    return next(iter(loader))


# ======================================================================
# TestSVASampleGeneration
# ======================================================================

class TestSVASampleGeneration:
    """样本生成的正确性测试。"""

    def test_dataset_creation(self, dataset: SVADataset) -> None:
        assert len(dataset) == 50

    def test_sample_fields(self, dataset: SVADataset) -> None:
        sample = dataset[0]
        assert isinstance(sample, SVASample)
        assert isinstance(sample.clean_text, str)
        assert isinstance(sample.corrupt_text, str)
        assert sample.clean_ids.dim() == 1
        assert sample.clean_ids.dtype == torch.long
        assert sample.corrupt_ids.dim() == 1
        assert sample.corrupt_ids.dtype == torch.long
        assert isinstance(sample.subject_pos, int)
        assert isinstance(sample.verb_pos, int)
        assert isinstance(sample.singular_verb_ids, list)
        assert isinstance(sample.plural_verb_ids, list)
        assert isinstance(sample.clean_is_singular, bool)

    def test_clean_corrupt_differ_at_one_position(
        self, dataset: SVADataset,
    ) -> None:
        for sample in dataset.samples:
            diffs = (sample.clean_ids != sample.corrupt_ids).nonzero(
                as_tuple=True
            )[0]
            assert len(diffs) == 1, (
                f"Expected 1 diff, got {len(diffs)} at positions {diffs.tolist()}"
            )
            # 差异位置应是 subject_pos
            assert diffs[0].item() == sample.subject_pos

    def test_subject_pos_is_always_1(
        self, dataset: SVADataset,
    ) -> None:
        """主语始终在位置 1（'The' 占位置 0）。"""
        for sample in dataset.samples:
            assert sample.subject_pos == 1

    def test_verb_pos_is_last_token(
        self, dataset: SVADataset,
    ) -> None:
        for sample in dataset.samples:
            assert sample.verb_pos == sample.clean_ids.shape[0] - 1

    def test_clean_is_singular(
        self, dataset: SVADataset,
    ) -> None:
        """所有 clean 样本的主语应为单数形式。"""
        singular_set = {
            dataset.tokenizer.encode(sg)[0] for sg, _ in NOUN_PAIRS
        }
        for sample in dataset.samples:
            subj_id = sample.clean_ids[sample.subject_pos].item()
            assert subj_id in singular_set, (
                f"Clean subject id {subj_id} not in singular noun set"
            )
            assert sample.clean_is_singular is True

    def test_corrupt_is_plural(
        self, dataset: SVADataset,
    ) -> None:
        """所有 corrupt 样本的主语应为复数形式。"""
        plural_set = {
            dataset.tokenizer.encode(pl)[0] for _, pl in NOUN_PAIRS
        }
        for sample in dataset.samples:
            subj_id = sample.corrupt_ids[sample.subject_pos].item()
            assert subj_id in plural_set, (
                f"Corrupt subject id {subj_id} not in plural noun set"
            )

    def test_clean_corrupt_same_length(
        self, dataset: SVADataset,
    ) -> None:
        for sample in dataset.samples:
            assert sample.clean_ids.shape == sample.corrupt_ids.shape

    def test_verb_id_lists_match(
        self, dataset: SVADataset,
    ) -> None:
        """所有样本的动词 id 列表应一致。"""
        first = dataset.samples[0]
        for sample in dataset.samples[1:]:
            assert sample.singular_verb_ids == first.singular_verb_ids
            assert sample.plural_verb_ids == first.plural_verb_ids

    def test_verb_pairs_count(
        self, dataset: SVADataset,
    ) -> None:
        sample = dataset.samples[0]
        assert len(sample.singular_verb_ids) == len(VERB_PAIRS)
        assert len(sample.plural_verb_ids) == len(VERB_PAIRS)

    def test_seed_reproducibility(self) -> None:
        ds1 = SVADataset(n_samples=10, seed=123)
        ds2 = SVADataset(n_samples=10, seed=123)
        for s1, s2 in zip(ds1.samples, ds2.samples):
            assert torch.equal(s1.clean_ids, s2.clean_ids)
            assert torch.equal(s1.corrupt_ids, s2.corrupt_ids)
            assert s1.subject_pos == s2.subject_pos
            assert s1.verb_pos == s2.verb_pos

    def test_n_attractors_fixed(self) -> None:
        """固定 n_attractors 参数时，所有样本序列长度应一致。"""
        for n_att in [0, 1, 2]:
            ds = SVADataset(n_samples=10, seed=42, n_attractors=n_att)
            lengths = [s.clean_ids.shape[0] for s in ds.samples]
            # 不同模板/名词可能有不同长度但差异很小
            assert max(lengths) - min(lengths) <= 2, (
                f"n_attractors={n_att}: lengths vary too much: {lengths}"
            )

    def test_text_starts_with_the(
        self, dataset: SVADataset,
    ) -> None:
        for sample in dataset.samples:
            assert sample.clean_text.startswith("The")
            assert sample.corrupt_text.startswith("The")

    def test_noun_pairs_sufficient(self) -> None:
        assert len(NOUN_PAIRS) >= 20

    def test_pp_templates_sufficient(self) -> None:
        assert len(PP_TEMPLATES) >= 10


# ======================================================================
# TestSVACollate
# ======================================================================

class TestSVACollate:
    """DataLoader / collate 函数测试。"""

    def test_batch_keys(self, batch: dict[str, torch.Tensor]) -> None:
        expected = {
            "clean_ids", "corrupt_ids", "subject_pos",
            "verb_pos", "singular_verb_ids", "plural_verb_ids",
        }
        assert set(batch.keys()) == expected

    def test_batch_shapes(self, batch: dict[str, torch.Tensor]) -> None:
        B = 50
        assert batch["clean_ids"].dim() == 2
        assert batch["clean_ids"].shape[0] == B
        assert batch["corrupt_ids"].shape[0] == B
        assert batch["subject_pos"].shape == (B,)
        assert batch["verb_pos"].shape == (B,)
        # 动词 id 不按 batch 维度展开
        assert batch["singular_verb_ids"].dim() == 1
        assert batch["plural_verb_ids"].dim() == 1
        assert len(batch["singular_verb_ids"]) == len(VERB_PAIRS)

    def test_no_negative_ids(self, batch: dict[str, torch.Tensor]) -> None:
        assert (batch["clean_ids"] >= 0).all()
        assert (batch["corrupt_ids"] >= 0).all()


# ======================================================================
# TestCheckpointSelection
# ======================================================================

class TestCheckpointSelection:
    """collect_key_positions_sva 测试。"""

    def test_collect_key_positions(
        self, batch: dict[str, torch.Tensor],
    ) -> None:
        positions = collect_key_positions_sva(batch)
        assert isinstance(positions, list)
        assert len(positions) > 0
        subject_vals = set(batch["subject_pos"].tolist())
        verb_vals = set(batch["verb_pos"].tolist())
        for p in subject_vals:
            assert p in positions
        for p in verb_vals:
            assert p in positions

    def test_key_positions_sorted_and_deduped(
        self, batch: dict[str, torch.Tensor],
    ) -> None:
        positions = collect_key_positions_sva(batch)
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
        ds = SVADataset(n_samples=50, seed=42)
        loader = ds.to_dataloader(batch_size=50, shuffle=False)
        batch = next(iter(loader))
        return teacher, ds, batch

    def test_teacher_prefers_correct_number(
        self, teacher_and_data,
    ) -> None:
        """clean 中教师偏好单数动词，corrupt 中偏好复数动词。"""
        teacher, ds, batch = teacher_and_data
        clean_ids = batch["clean_ids"].to(teacher.device)
        corrupt_ids = batch["corrupt_ids"].to(teacher.device)
        verb_pos = batch["verb_pos"]
        sg_ids = batch["singular_verb_ids"].to(teacher.device)  # (n_verbs,)
        pl_ids = batch["plural_verb_ids"].to(teacher.device)    # (n_verbs,)

        with torch.no_grad():
            clean_logits = teacher.model(clean_ids).logits   # (B, seq, V)
            corrupt_logits = teacher.model(corrupt_ids).logits

        clean_correct, corrupt_correct = 0, 0
        for i in range(clean_ids.shape[0]):
            vp = verb_pos[i].item()
            # Clean (singular subject): sum of singular verb logits > plural
            cl = clean_logits[i, vp, :]   # (V,)
            sg_sum = cl[sg_ids].sum().item()
            pl_sum = cl[pl_ids].sum().item()
            if sg_sum > pl_sum:
                clean_correct += 1
            # Corrupt (plural subject): sum of plural verb logits > singular
            co = corrupt_logits[i, vp, :]
            sg_sum_c = co[sg_ids].sum().item()
            pl_sum_c = co[pl_ids].sum().item()
            if pl_sum_c > sg_sum_c:
                corrupt_correct += 1

        n = clean_ids.shape[0]
        clean_acc = clean_correct / n
        corrupt_acc = corrupt_correct / n
        print(f"\nSVA clean accuracy: {clean_acc:.0%} ({clean_correct}/{n})")
        print(f"SVA corrupt accuracy: {corrupt_acc:.0%} ({corrupt_correct}/{n})")
        assert clean_acc > 0.6, f"Clean accuracy {clean_acc:.1%} < 60%"
        assert corrupt_acc > 0.6, f"Corrupt accuracy {corrupt_acc:.1%} < 60%"

    def test_teacher_number_preference_flips(
        self, teacher_and_data,
    ) -> None:
        """单数/复数动词 logit 差 (sg - pl) 在 clean 中应为正，corrupt 中应为负。"""
        teacher, ds, batch = teacher_and_data
        clean_ids = batch["clean_ids"].to(teacher.device)
        corrupt_ids = batch["corrupt_ids"].to(teacher.device)
        verb_pos = batch["verb_pos"]
        sg_ids = batch["singular_verb_ids"].to(teacher.device)
        pl_ids = batch["plural_verb_ids"].to(teacher.device)

        with torch.no_grad():
            clean_logits = teacher.model(clean_ids).logits
            corrupt_logits = teacher.model(corrupt_ids).logits

        clean_diffs, corrupt_diffs = [], []
        for i in range(clean_ids.shape[0]):
            vp = verb_pos[i].item()
            cl = clean_logits[i, vp, :]
            co = corrupt_logits[i, vp, :]
            # sg - pl logit difference (positive = prefers singular)
            clean_diffs.append(
                (cl[sg_ids].mean() - cl[pl_ids].mean()).item()
            )
            corrupt_diffs.append(
                (co[sg_ids].mean() - co[pl_ids].mean()).item()
            )

        mean_clean = sum(clean_diffs) / len(clean_diffs)
        mean_corrupt = sum(corrupt_diffs) / len(corrupt_diffs)
        print(f"\nSVA logit diff: clean={mean_clean:.3f} corrupt={mean_corrupt:.3f}")
        # Clean should have positive diff, corrupt negative
        assert mean_clean > mean_corrupt, (
            f"Clean diff ({mean_clean:.3f}) should be > corrupt diff ({mean_corrupt:.3f})"
        )

    def test_teacher_accuracy(self, teacher_and_data) -> None:
        """教师在 clean 样本上的 SVA 判断准确率。"""
        teacher, ds, batch = teacher_and_data
        clean_ids = batch["clean_ids"].to(teacher.device)
        verb_pos = batch["verb_pos"]
        # Use is/are pair (strongest signal)
        is_id = ds.tokenizer.encode(" is")[0]
        are_id = ds.tokenizer.encode(" are")[0]

        with torch.no_grad():
            logits = teacher.model(clean_ids).logits

        correct = 0
        for i in range(clean_ids.shape[0]):
            vp = verb_pos[i].item()
            # Clean is singular → should prefer " is" over " are"
            if logits[i, vp, is_id] > logits[i, vp, are_id]:
                correct += 1

        n = clean_ids.shape[0]
        accuracy = correct / n
        print(f"\nSVA is/are accuracy: {accuracy:.0%} ({correct}/{n})")
        assert accuracy > 0.6, f"SVA accuracy {accuracy:.1%} < 60%"

    def test_print_samples(self, teacher_and_data) -> None:
        """打印 10 个样本展示格式（不断言，仅输出）。"""
        _, ds, _ = teacher_and_data
        print("\n" + "=" * 60)
        print("SVA — Sample Display (10 samples)")
        print("=" * 60)
        for i, sample in enumerate(ds.samples[:10]):
            print(f"\n--- Sample {i+1} ---")
            print(f"  subject_pos={sample.subject_pos}, "
                  f"verb_pos={sample.verb_pos}")
            # Use ascii() to avoid Windows GBK encoding errors
            print(f"  clean:   {ascii(sample.clean_text)}")
            print(f"  corrupt: {ascii(sample.corrupt_text)}")
