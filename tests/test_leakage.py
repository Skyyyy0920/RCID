"""因果痕迹信息泄露测试模块单元测试。

测试覆盖：
- _train_linear_probe: 可分离数据 → 高准确率，随机数据 → ≈ chance
- make_binary_labels: 中位数分割正确
- make_control_*: 控制标签生成
- LeakageTestEvaluator.run_test: 返回正确键、范围
- 信息性痕迹 → task 高 / control 低（selectivity > 0）
- 随机痕迹 → task ≈ control ≈ chance
- run_multi_control_test: 多控制标签
- GPT-2 端到端（slow）: 真实因果痕迹的 task 探针
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from rcid.eval.leakage_test import (
    LeakageTestEvaluator,
    ProbeResult,
    _train_linear_probe,
    make_binary_labels,
    make_control_even_length,
    make_control_first_token_even,
    make_control_random,
)


# ======================================================================
# TestTrainLinearProbe
# ======================================================================

class TestTrainLinearProbe:
    """_train_linear_probe 单元测试。"""

    def test_linearly_separable(self) -> None:
        """线性可分数据 → 准确率接近 1.0。"""
        torch.manual_seed(42)
        n_train, n_test, d = 200, 50, 10
        w_true = torch.randn(d)  # (d,)

        X_train = torch.randn(n_train, d)
        y_train = (X_train @ w_true > 0).long()
        X_test = torch.randn(n_test, d)
        y_test = (X_test @ w_true > 0).long()

        acc = _train_linear_probe(
            X_train, y_train, X_test, y_test,
            n_epochs=300, lr=0.05,
        )
        assert acc > 0.85, f"Expected >0.85, got {acc:.4f}"

    def test_random_labels_near_chance(self) -> None:
        """随机标签 → 准确率 ≈ 0.5。"""
        torch.manual_seed(42)
        n_train, n_test, d = 200, 50, 10

        X_train = torch.randn(n_train, d)
        y_train = torch.randint(0, 2, (n_train,))
        X_test = torch.randn(n_test, d)
        y_test = torch.randint(0, 2, (n_test,))

        acc = _train_linear_probe(
            X_train, y_train, X_test, y_test,
            n_epochs=200, lr=0.01,
        )
        assert 0.2 <= acc <= 0.8, f"Expected near 0.5, got {acc:.4f}"

    def test_high_dimensional(self) -> None:
        """高维输入也应工作。"""
        torch.manual_seed(42)
        d = 768  # GPT-2 维度
        X_train = torch.randn(100, d)
        w = torch.randn(d)
        y_train = (X_train @ w > 0).long()
        X_test = torch.randn(20, d)
        y_test = (X_test @ w > 0).long()

        acc = _train_linear_probe(
            X_train, y_train, X_test, y_test,
            n_epochs=300, lr=0.01,
        )
        assert acc >= 0.65, f"High-dim: acc={acc:.4f}"


# ======================================================================
# TestLabelGenerators
# ======================================================================

class TestLabelGenerators:
    """标签生成工具测试。"""

    def test_make_binary_labels_balanced(self) -> None:
        labels = torch.tensor([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
        binary = make_binary_labels(labels)
        assert binary.shape == (10,)
        assert set(binary.tolist()).issubset({0, 1})
        # median = 5.5, 所以 >5.5 → 1 (6-10), ≤5.5 → 0 (1-5)
        assert binary.sum().item() == 5

    def test_make_binary_labels_constant(self) -> None:
        """全部相同 → 全为 0。"""
        labels = torch.tensor([5, 5, 5, 5])
        binary = make_binary_labels(labels)
        assert binary.sum().item() == 0

    def test_control_even_length(self) -> None:
        lengths = torch.tensor([10, 11, 12, 13])
        ctrl = make_control_even_length(lengths)
        assert ctrl.tolist() == [1, 0, 1, 0]

    def test_control_first_token_even(self) -> None:
        ids = torch.tensor([100, 101, 200, 201])
        ctrl = make_control_first_token_even(ids)
        assert ctrl.tolist() == [1, 0, 1, 0]

    def test_control_random(self) -> None:
        ctrl = make_control_random(100, seed=42)
        assert ctrl.shape == (100,)
        assert set(ctrl.tolist()).issubset({0, 1})
        # 不应全 0 或全 1
        assert 20 < ctrl.sum().item() < 80

    def test_control_random_deterministic(self) -> None:
        c1 = make_control_random(50, seed=99)
        c2 = make_control_random(50, seed=99)
        assert torch.equal(c1, c2)


# ======================================================================
# TestLeakageTestEvaluator
# ======================================================================

class TestLeakageTestEvaluator:
    """LeakageTestEvaluator 测试。"""

    def test_return_keys(self) -> None:
        evaluator = LeakageTestEvaluator(
            test_ratio=0.3, n_epochs=50, seed=42,
        )
        X = torch.randn(100, 16)
        task_labels = torch.randint(0, 2, (100,))
        ctrl_labels = torch.randint(0, 2, (100,))

        result = evaluator.run_test(X, task_labels, ctrl_labels)
        assert "task_probe_accuracy" in result
        assert "control_probe_accuracy" in result
        assert "selectivity" in result
        assert "task_probe" in result
        assert "control_probe" in result

    def test_accuracy_range(self) -> None:
        evaluator = LeakageTestEvaluator(n_epochs=50)
        X = torch.randn(100, 16)
        task_labels = torch.randint(0, 2, (100,))
        ctrl_labels = torch.randint(0, 2, (100,))

        result = evaluator.run_test(X, task_labels, ctrl_labels)
        assert 0.0 <= result["task_probe_accuracy"] <= 1.0
        assert 0.0 <= result["control_probe_accuracy"] <= 1.0

    def test_selectivity_is_difference(self) -> None:
        evaluator = LeakageTestEvaluator(n_epochs=50)
        X = torch.randn(100, 16)
        task_labels = torch.randint(0, 2, (100,))
        ctrl_labels = torch.randint(0, 2, (100,))

        result = evaluator.run_test(X, task_labels, ctrl_labels)
        expected = result["task_probe_accuracy"] - result["control_probe_accuracy"]
        assert abs(result["selectivity"] - expected) < 1e-6

    def test_probe_result_type(self) -> None:
        evaluator = LeakageTestEvaluator(n_epochs=50)
        X = torch.randn(100, 16)
        task_labels = torch.randint(0, 2, (100,))
        ctrl_labels = torch.randint(0, 2, (100,))

        result = evaluator.run_test(X, task_labels, ctrl_labels)
        assert isinstance(result["task_probe"], ProbeResult)
        assert isinstance(result["control_probe"], ProbeResult)

    def test_informative_imprints_high_selectivity(self) -> None:
        """痕迹包含任务信息 → task 高, control 低 → selectivity > 0。"""
        torch.manual_seed(42)
        n, d = 300, 32
        w = torch.randn(d)

        # 痕迹与 task 标签线性相关
        X = torch.randn(n, d)
        task_labels = (X @ w > 0).long()
        # 控制标签与痕迹无关
        ctrl_labels = make_control_random(n, seed=999)

        evaluator = LeakageTestEvaluator(
            n_epochs=300, lr=0.05, seed=42,
        )
        result = evaluator.run_test(X, task_labels, ctrl_labels)

        assert result["task_probe_accuracy"] > 0.75, (
            f"Task acc too low: {result['task_probe_accuracy']:.4f}"
        )
        assert result["selectivity"] > 0.15, (
            f"Selectivity too low: {result['selectivity']:.4f}"
        )

    def test_random_imprints_low_task_accuracy(self) -> None:
        """随机痕迹 → task 准确率 ≈ chance。"""
        torch.manual_seed(42)
        n, d = 200, 32

        X = torch.randn(n, d)
        task_labels = torch.randint(0, 2, (n,))
        ctrl_labels = torch.randint(0, 2, (n,))

        evaluator = LeakageTestEvaluator(n_epochs=100, seed=42)
        result = evaluator.run_test(X, task_labels, ctrl_labels)

        # 随机 → 两者都接近 0.5
        assert 0.2 <= result["task_probe_accuracy"] <= 0.8
        assert 0.2 <= result["control_probe_accuracy"] <= 0.8


# ======================================================================
# TestMultiControlTest
# ======================================================================

class TestMultiControlTest:
    """run_multi_control_test 测试。"""

    def test_return_keys(self) -> None:
        evaluator = LeakageTestEvaluator(n_epochs=50)
        X = torch.randn(100, 16)
        task_labels = torch.randint(0, 2, (100,))
        controls = {
            "even_len": make_control_random(100, seed=1),
            "first_token": make_control_random(100, seed=2),
            "random": make_control_random(100, seed=3),
        }

        result = evaluator.run_multi_control_test(X, task_labels, controls)
        assert "task_probe_accuracy" in result
        assert "control_accuracies" in result
        assert "mean_control_accuracy" in result
        assert "selectivity" in result

    def test_control_accuracies_keys(self) -> None:
        evaluator = LeakageTestEvaluator(n_epochs=50)
        X = torch.randn(100, 16)
        task_labels = torch.randint(0, 2, (100,))
        controls = {
            "a": make_control_random(100, seed=1),
            "b": make_control_random(100, seed=2),
        }

        result = evaluator.run_multi_control_test(X, task_labels, controls)
        assert set(result["control_accuracies"].keys()) == {"a", "b"}

    def test_mean_is_average(self) -> None:
        evaluator = LeakageTestEvaluator(n_epochs=50)
        X = torch.randn(100, 16)
        task_labels = torch.randint(0, 2, (100,))
        controls = {
            "a": make_control_random(100, seed=1),
            "b": make_control_random(100, seed=2),
        }

        result = evaluator.run_multi_control_test(X, task_labels, controls)
        accs = list(result["control_accuracies"].values())
        expected = sum(accs) / len(accs)
        assert abs(result["mean_control_accuracy"] - expected) < 1e-6


# ======================================================================
# TestConstantLabels
# ======================================================================

class TestConstantLabels:
    """边界情况：标签全部相同。"""

    def test_constant_labels_handled(self) -> None:
        evaluator = LeakageTestEvaluator(n_epochs=50, seed=42)
        X = torch.randn(100, 16)
        # 全为 1 的标签
        task_labels = torch.ones(100, dtype=torch.long)
        ctrl_labels = torch.randint(0, 2, (100,))

        result = evaluator.run_test(X, task_labels, ctrl_labels)
        # 应不崩溃；task_probe_accuracy 为 majority class accuracy
        assert 0.0 <= result["task_probe_accuracy"] <= 1.0


# ======================================================================
# TestGPT2EndToEnd (slow)
# ======================================================================

@pytest.mark.slow
class TestGPT2EndToEnd:
    """端到端测试：用真实 GPT-2 提取因果痕迹。"""

    @pytest.fixture(scope="class")
    def gpt2_model(self):
        from transformers import GPT2LMHeadModel
        model = GPT2LMHeadModel.from_pretrained("gpt2")
        model.eval()
        return model

    @pytest.fixture(scope="class")
    def gpt2_tokenizer(self):
        from transformers import GPT2Tokenizer
        return GPT2Tokenizer.from_pretrained("gpt2")

    def _extract_imprints(
        self, model, clean_ids, corrupt_ids, layer, token_pos,
    ) -> torch.Tensor:
        """提取因果痕迹 d = r(clean) - r(corrupt) at (layer, token_pos)。"""
        storage_c: dict[str, torch.Tensor] = {}
        storage_x: dict[str, torch.Tensor] = {}

        def make_hook(s: dict):
            def hook(mod, inp, out):
                s["r"] = out[0]
            return hook

        h1 = model.transformer.h[layer].register_forward_hook(make_hook(storage_c))
        with torch.no_grad():
            model(clean_ids)
        h1.remove()

        h2 = model.transformer.h[layer].register_forward_hook(make_hook(storage_x))
        with torch.no_grad():
            model(corrupt_ids)
        h2.remove()

        d = storage_c["r"][:, token_pos, :] - storage_x["r"][:, token_pos, :]
        return d  # (B, d_model)

    def test_ioi_task_probe_high(
        self, gpt2_model, gpt2_tokenizer,
    ) -> None:
        """IOI 因果痕迹应包含 IO identity 信息。"""
        from rcid.data.ioi import IOIDataset

        ds = IOIDataset(n_samples=200, tokenizer=gpt2_tokenizer, seed=42)
        dl = ds.to_dataloader(batch_size=200, shuffle=False)
        batch = next(iter(dl))

        # 提取 layer=9 (Name Mover heads 附近) 的痕迹
        clean = batch["clean_ids"]
        corrupt = batch["corrupt_ids"]
        # 用最后位置
        last_pos = clean.shape[1] - 1
        imprints = self._extract_imprints(
            gpt2_model, clean, corrupt, layer=9, token_pos=last_pos,
        )  # (200, 768)

        # 任务标签：answer_token_id 的 median 分割
        task_labels = make_binary_labels(batch["answer_token_id"])
        ctrl_labels = make_control_random(200, seed=42)

        evaluator = LeakageTestEvaluator(
            n_epochs=300, lr=0.01, test_ratio=0.2, seed=42,
        )
        result = evaluator.run_test(imprints, task_labels, ctrl_labels)

        print(f"\nIOI task probe: {result['task_probe_accuracy']:.4f}")
        print(f"IOI ctrl probe: {result['control_probe_accuracy']:.4f}")
        print(f"IOI selectivity: {result['selectivity']:.4f}")

        # 任务探针应显著 > chance
        assert result["task_probe_accuracy"] > 0.6, (
            f"IOI task probe too low: {result['task_probe_accuracy']:.4f}"
        )
        # selectivity > 0
        assert result["selectivity"] > 0.0

    def test_random_imprints_baseline(self, gpt2_model) -> None:
        """随机向量替代因果痕迹 → task ≈ chance。"""
        torch.manual_seed(42)
        n, d = 200, 768
        X = torch.randn(n, d)
        task_labels = torch.randint(0, 2, (n,))
        ctrl_labels = make_control_random(n, seed=42)

        evaluator = LeakageTestEvaluator(
            n_epochs=200, lr=0.01, seed=42,
        )
        result = evaluator.run_test(X, task_labels, ctrl_labels)

        print(f"\nRandom task probe: {result['task_probe_accuracy']:.4f}")
        print(f"Random ctrl probe: {result['control_probe_accuracy']:.4f}")

        # 随机 → task ≈ chance
        assert result["task_probe_accuracy"] < 0.7
