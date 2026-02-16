"""Docstring 参数补全数据集单元测试。"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from rcid.circuit.checkpoint_selection import collect_key_positions_docstring
from rcid.data.docstring import (
    DOC_DESCS,
    FUNC_NAMES,
    PARAM_DESCS,
    PARAM_NAMES,
    DocstringDataset,
    DocstringSample,
)


# ======================================================================
# Fixtures
# ======================================================================

@pytest.fixture(scope="module")
def dataset() -> DocstringDataset:
    return DocstringDataset(n_samples=50, seed=42)


@pytest.fixture(scope="module")
def batch(dataset: DocstringDataset) -> dict[str, torch.Tensor]:
    loader = dataset.to_dataloader(batch_size=50, shuffle=False)
    return next(iter(loader))


# ======================================================================
# TestDocstringSampleGeneration
# ======================================================================

class TestDocstringSampleGeneration:
    """样本生成的正确性测试。"""

    def test_dataset_creation(self, dataset: DocstringDataset) -> None:
        assert len(dataset) == 50

    def test_sample_fields(self, dataset: DocstringDataset) -> None:
        sample = dataset[0]
        assert isinstance(sample, DocstringSample)
        assert isinstance(sample.clean_text, str)
        assert isinstance(sample.corrupt_text, str)
        assert sample.clean_ids.dim() == 1
        assert sample.clean_ids.dtype == torch.long
        assert sample.corrupt_ids.dim() == 1
        assert sample.corrupt_ids.dtype == torch.long
        assert isinstance(sample.target_pos, int)
        assert isinstance(sample.answer_token_id, int)
        assert isinstance(sample.corrupt_token_id, int)
        assert isinstance(sample.param_name_pos, int)

    def test_clean_corrupt_differ_at_one_position(
        self, dataset: DocstringDataset,
    ) -> None:
        for sample in dataset.samples:
            diffs = (sample.clean_ids != sample.corrupt_ids).nonzero(
                as_tuple=True
            )[0]
            assert len(diffs) == 1, (
                f"Expected 1 diff, got {len(diffs)} at {diffs.tolist()}"
            )
            assert diffs[0].item() == sample.param_name_pos

    def test_clean_corrupt_same_length(
        self, dataset: DocstringDataset,
    ) -> None:
        for sample in dataset.samples:
            assert sample.clean_ids.shape == sample.corrupt_ids.shape

    def test_target_pos_is_last_token(
        self, dataset: DocstringDataset,
    ) -> None:
        for sample in dataset.samples:
            assert sample.target_pos == sample.clean_ids.shape[0] - 1

    def test_answer_and_corrupt_token_differ(
        self, dataset: DocstringDataset,
    ) -> None:
        for sample in dataset.samples:
            assert sample.answer_token_id != sample.corrupt_token_id

    def test_answer_token_in_param_names(
        self, dataset: DocstringDataset,
    ) -> None:
        """answer_token_id 应对应某个已知参数名。"""
        param_ids = {
            dataset.tokenizer.encode(p)[0] for p in PARAM_NAMES
        }
        for sample in dataset.samples:
            assert sample.answer_token_id in param_ids
            assert sample.corrupt_token_id in param_ids

    def test_diff_position_matches_signature_param(
        self, dataset: DocstringDataset,
    ) -> None:
        """差异位置的 clean token 应是 answer_token_id。"""
        for sample in dataset.samples:
            pos = sample.param_name_pos
            assert sample.clean_ids[pos].item() == sample.answer_token_id
            assert sample.corrupt_ids[pos].item() == sample.corrupt_token_id

    def test_text_starts_with_def(
        self, dataset: DocstringDataset,
    ) -> None:
        for sample in dataset.samples:
            assert sample.clean_text.startswith("def ")
            assert sample.corrupt_text.startswith("def ")

    def test_text_contains_param(
        self, dataset: DocstringDataset,
    ) -> None:
        for sample in dataset.samples:
            assert ":param" in sample.clean_text
            assert sample.clean_text.endswith(":param")

    def test_seed_reproducibility(self) -> None:
        ds1 = DocstringDataset(n_samples=10, seed=123)
        ds2 = DocstringDataset(n_samples=10, seed=123)
        for s1, s2 in zip(ds1.samples, ds2.samples):
            assert torch.equal(s1.clean_ids, s2.clean_ids)
            assert torch.equal(s1.corrupt_ids, s2.corrupt_ids)
            assert s1.target_pos == s2.target_pos
            assert s1.answer_token_id == s2.answer_token_id

    def test_param_names_sufficient(self) -> None:
        assert len(PARAM_NAMES) >= 20

    def test_func_names_sufficient(self) -> None:
        assert len(FUNC_NAMES) >= 10

    def test_doc_descs_sufficient(self) -> None:
        assert len(DOC_DESCS) >= 5

    def test_param_descs_sufficient(self) -> None:
        assert len(PARAM_DESCS) >= 3

    def test_three_params_in_signature(
        self, dataset: DocstringDataset,
    ) -> None:
        """签名中应有 3 个参数（2 个逗号）。"""
        for sample in dataset.samples:
            # Extract signature line
            sig_line = sample.clean_text.split("\n")[0]
            # Count commas inside parentheses
            paren_content = sig_line.split("(")[1].split(")")[0]
            params = paren_content.split(",")
            assert len(params) == 3, (
                f"Expected 3 params, got {len(params)}: {sig_line}"
            )


# ======================================================================
# TestDocstringCollate
# ======================================================================

class TestDocstringCollate:
    """DataLoader / collate 函数测试。"""

    def test_batch_keys(self, batch: dict[str, torch.Tensor]) -> None:
        expected = {
            "clean_ids", "corrupt_ids", "target_pos",
            "answer_token_id", "corrupt_token_id", "param_name_pos",
        }
        assert set(batch.keys()) == expected

    def test_batch_shapes(self, batch: dict[str, torch.Tensor]) -> None:
        B = 50
        assert batch["clean_ids"].dim() == 2
        assert batch["clean_ids"].shape[0] == B
        assert batch["corrupt_ids"].shape[0] == B
        assert batch["target_pos"].shape == (B,)
        assert batch["answer_token_id"].shape == (B,)
        assert batch["corrupt_token_id"].shape == (B,)
        assert batch["param_name_pos"].shape == (B,)

    def test_no_negative_ids(self, batch: dict[str, torch.Tensor]) -> None:
        assert (batch["clean_ids"] >= 0).all()
        assert (batch["corrupt_ids"] >= 0).all()


# ======================================================================
# TestCheckpointSelection
# ======================================================================

class TestCheckpointSelection:
    """collect_key_positions_docstring 测试。"""

    def test_collect_key_positions(
        self, batch: dict[str, torch.Tensor],
    ) -> None:
        positions = collect_key_positions_docstring(batch)
        assert isinstance(positions, list)
        assert len(positions) > 0
        param_vals = set(batch["param_name_pos"].tolist())
        target_vals = set(batch["target_pos"].tolist())
        for p in param_vals:
            assert p in positions
        for p in target_vals:
            assert p in positions

    def test_key_positions_sorted_and_deduped(
        self, batch: dict[str, torch.Tensor],
    ) -> None:
        positions = collect_key_positions_docstring(batch)
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
        ds = DocstringDataset(n_samples=50, seed=42)
        loader = ds.to_dataloader(batch_size=50, shuffle=False)
        batch = next(iter(loader))
        return teacher, ds, batch

    def test_teacher_accuracy_clean(self, teacher_and_data) -> None:
        """教师在 clean 样本上 top-1 准确率应 > 60%。"""
        teacher, ds, batch = teacher_and_data
        clean_ids = batch["clean_ids"].to(teacher.device)
        target_pos = batch["target_pos"]
        answer_ids = batch["answer_token_id"]

        with torch.no_grad():
            logits = teacher.model(clean_ids).logits

        correct = 0
        for i in range(clean_ids.shape[0]):
            tp = target_pos[i].item()
            pred = logits[i, tp, :].argmax().item()
            if pred == answer_ids[i].item():
                correct += 1

        n = clean_ids.shape[0]
        accuracy = correct / n
        print(f"\nDocstring clean top-1 accuracy: {accuracy:.0%} ({correct}/{n})")
        assert accuracy > 0.6, f"Clean accuracy {accuracy:.1%} < 60%"

    def test_teacher_accuracy_corrupt(self, teacher_and_data) -> None:
        """教师在 corrupt 样本上应预测替换后的参数名。"""
        teacher, ds, batch = teacher_and_data
        corrupt_ids = batch["corrupt_ids"].to(teacher.device)
        target_pos = batch["target_pos"]
        corrupt_answer = batch["corrupt_token_id"]

        with torch.no_grad():
            logits = teacher.model(corrupt_ids).logits

        correct = 0
        for i in range(corrupt_ids.shape[0]):
            tp = target_pos[i].item()
            pred = logits[i, tp, :].argmax().item()
            if pred == corrupt_answer[i].item():
                correct += 1

        n = corrupt_ids.shape[0]
        accuracy = correct / n
        print(f"Docstring corrupt top-1 accuracy: {accuracy:.0%} ({correct}/{n})")
        assert accuracy > 0.5, f"Corrupt accuracy {accuracy:.1%} < 50%"

    def test_teacher_prefers_correct_param(self, teacher_and_data) -> None:
        """clean 中 answer logit > corrupt answer logit；反之亦然。"""
        teacher, ds, batch = teacher_and_data
        clean_ids = batch["clean_ids"].to(teacher.device)
        corrupt_ids = batch["corrupt_ids"].to(teacher.device)
        target_pos = batch["target_pos"]
        answer_ids = batch["answer_token_id"]
        corrupt_answer = batch["corrupt_token_id"]

        with torch.no_grad():
            cl = teacher.model(clean_ids).logits
            co = teacher.model(corrupt_ids).logits

        # In clean: answer param logit should beat corrupt param logit
        clean_correct = 0
        for i in range(clean_ids.shape[0]):
            tp = target_pos[i].item()
            a_id = answer_ids[i].item()
            c_id = corrupt_answer[i].item()
            if cl[i, tp, a_id] > cl[i, tp, c_id]:
                clean_correct += 1

        n = clean_ids.shape[0]
        ratio = clean_correct / n
        print(f"Docstring clean prefers correct param: {ratio:.0%}")
        assert ratio > 0.7, f"Only {ratio:.1%} prefer correct param in clean"

    def test_print_samples(self, teacher_and_data) -> None:
        """打印 10 个样本展示格式（不断言，仅输出）。"""
        _, ds, _ = teacher_and_data
        print("\n" + "=" * 60)
        print("Docstring — Sample Display (10 samples)")
        print("=" * 60)
        for i, sample in enumerate(ds.samples[:10]):
            print(f"\n--- Sample {i+1} ---")
            print(f"  target_pos={sample.target_pos}, "
                  f"param_name_pos={sample.param_name_pos}")
            # Use ascii() to avoid Windows GBK encoding errors
            print(f"  answer={sample.answer_token_id}, "
                  f"corrupt={sample.corrupt_token_id}")
            # Show first 2 lines of clean text
            lines = sample.clean_text.split("\n")
            print(f"  sig:    {ascii(lines[0])}")
            print(f"  end:    ...{ascii(lines[-1])}")
