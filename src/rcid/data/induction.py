"""Induction Heads 任务数据与对比对构建。

Induction Heads 执行 [A][B]...[A] → [B] 的复制模式 (Olsson et al., 2022)。
这是 Transformer 中最基本的 in-context learning 电路。

对比对构造（最小修改原则）：
    clean:   [...random... A B ...random... A _]  → 模型应在 _ 位置预测 B
    corrupt: [...random... A B ...random... C _]  → 仅将第二个 A 替换为 C（C≠A）
    差值捕捉 induction head 的因果效应。
"""

from __future__ import annotations

import logging
import random
from dataclasses import dataclass

import torch
from torch.utils.data import DataLoader, Dataset
from transformers import GPT2Tokenizer

logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────────────────
# 安全 token 池：排除特殊 token 和罕见编码
# GPT-2 vocab_size=50257, <|endoftext|>=50256, pad 使用 0
# 使用 [1000, 40000) 避免控制字符、罕见字节 token 和尾部特殊 token
# ──────────────────────────────────────────────────────────────────────
SAFE_TOKEN_MIN: int = 1000
SAFE_TOKEN_MAX: int = 40000
SAFE_POOL: list[int] = list(range(SAFE_TOKEN_MIN, SAFE_TOKEN_MAX))

# GPT-2 特殊 token
SPECIAL_TOKEN_ID: int = 50256  # <|endoftext|>


@dataclass
class InductionSample:
    """单个 Induction Heads 对比对的全部信息。"""

    clean_text: str
    corrupt_text: str
    clean_ids: torch.Tensor     # (seq_len,)
    corrupt_ids: torch.Tensor   # (seq_len,)
    trigger_pos: int            # 第一个 [A] 的位置
    target_pos: int             # 第二个 [A] 的位置（读取此位置的 logits 来预测 B）
    answer_token_id: int        # 正确答案 B 的 token id


class InductionDataset(Dataset[InductionSample]):
    """Induction Heads 对比数据集。

    生成 [A][B]...[A] → [B] 模式的对比样本对。
    所有序列等长（固定 seq_len），无需 padding。
    """

    def __init__(
        self,
        n_samples: int = 500,
        tokenizer: GPT2Tokenizer | None = None,
        seed: int = 42,
        seq_len: int = 30,
    ) -> None:
        """生成 n_samples 个 Induction Heads 对比对。

        Args:
            n_samples: 对比对数量。
            tokenizer: GPT-2 tokenizer；若为 None 则自动加载。
            seed: 随机种子，保证可复现。
            seq_len: 固定序列长度（所有样本等长）。
        """
        assert seq_len >= 12, f"seq_len must be >= 12, got {seq_len}"
        assert len(SAFE_POOL) >= 100, (
            f"Safe token pool too small: {len(SAFE_POOL)}"
        )

        self.tokenizer = tokenizer or GPT2Tokenizer.from_pretrained("gpt2")
        self.n_samples = n_samples
        self.seq_len = seq_len

        rng = random.Random(seed)
        self.samples: list[InductionSample] = []

        for _ in range(n_samples):
            sample = self._generate_one(rng)
            self.samples.append(sample)

        logger.info(
            "InductionDataset: generated %d contrastive pairs (seq_len=%d)",
            len(self.samples), seq_len,
        )

    # ------------------------------------------------------------------
    # 内部方法
    # ------------------------------------------------------------------

    def _generate_one(self, rng: random.Random) -> InductionSample:
        """生成单个 Induction Heads 对比对。

        使用 "重复随机序列" 模式（Olsson et al., 2022）：
          [prefix] [subseq] [subseq_repeat...]
        整个子序列被重复，模型通过 induction head 在看到第二个 [A] 后预测 [B]。

        构造逻辑：
        1. 生成随机前缀 + 随机子序列 [A B X1 X2 ...]
        2. Clean: prefix + subseq + subseq（重复），截断到 seq_len
        3. Corrupt: 将第二个 [A] 替换为 [C]（C≠A），破坏 induction 信号
        """
        sl = self.seq_len

        # 前缀长度和子序列长度
        prefix_len = rng.randint(2, max(3, sl // 4))
        # 子序列至少 3 token（A, B, ≥1 filler），重复后不超过 seq_len
        max_subseq = (sl - prefix_len) // 2
        subseq_len = rng.randint(3, min(8, max_subseq))

        # 生成随机 token（全部不重复以确保 A 只出现两次）
        n_unique = prefix_len + subseq_len + 1  # +1 for C
        tokens = rng.sample(SAFE_POOL, n_unique)
        prefix_tokens = tokens[:prefix_len]
        subseq_tokens = tokens[prefix_len:prefix_len + subseq_len]
        c_id = tokens[-1]  # corrupt 替换 token

        a_id = subseq_tokens[0]  # A = 子序列第一个 token
        b_id = subseq_tokens[1]  # B = 子序列第二个 token

        trigger_pos = prefix_len  # 第一个 [A] 的位置
        second_a_pos = prefix_len + subseq_len  # 第二个 [A] 的位置
        target_pos = second_a_pos  # 读取此位置的 logits → 预测 B

        # 构建 clean: [prefix] [subseq] [subseq_repeat] + filler
        # 重复子序列以激活 induction heads
        repeat_tokens = list(subseq_tokens)  # 完整重复
        clean_ids = list(prefix_tokens) + list(subseq_tokens) + repeat_tokens
        # 用随机 filler（不含 A）补齐到 seq_len
        filler_pool = [t for t in SAFE_POOL if t != a_id]
        while len(clean_ids) < sl:
            clean_ids.append(rng.choice(filler_pool))
        clean_ids = clean_ids[:sl]

        # 验证 A 恰好出现两次（prefix 和 filler 都不含 A）
        a_count = clean_ids.count(a_id)
        assert a_count == 2, (
            f"A (id={a_id}) appears {a_count} times, expected 2"
        )

        # 构建 corrupt 序列（最小修改：第二个 A → C）
        corrupt_ids = list(clean_ids)
        corrupt_ids[second_a_pos] = c_id

        # 验证最小修改
        diffs = [i for i in range(sl) if clean_ids[i] != corrupt_ids[i]]
        assert diffs == [second_a_pos], (
            f"Minimal modification violated: diffs at {diffs}, "
            f"expected [{second_a_pos}]"
        )

        # 转为文本（用于调试展示）
        clean_text = self.tokenizer.decode(clean_ids)
        corrupt_text = self.tokenizer.decode(corrupt_ids)

        return InductionSample(
            clean_text=clean_text,
            corrupt_text=corrupt_text,
            clean_ids=torch.tensor(clean_ids, dtype=torch.long),      # (seq_len,)
            corrupt_ids=torch.tensor(corrupt_ids, dtype=torch.long),  # (seq_len,)
            trigger_pos=trigger_pos,
            target_pos=target_pos,
            answer_token_id=b_id,
        )

    # ------------------------------------------------------------------
    # Dataset 接口
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return self.n_samples

    def __getitem__(self, idx: int) -> InductionSample:
        return self.samples[idx]

    # ------------------------------------------------------------------
    # DataLoader 工厂
    # ------------------------------------------------------------------

    def to_dataloader(
        self,
        batch_size: int = 32,
        shuffle: bool = True,
    ) -> DataLoader[dict[str, torch.Tensor]]:
        """创建 DataLoader，将 InductionSample 整理为 batch 字典。

        返回的每个 batch 是一个字典:
            clean_ids:        (B, seq_len)
            corrupt_ids:      (B, seq_len)
            trigger_pos:      (B,)
            target_pos:       (B,)
            answer_token_id:  (B,)

        Args:
            batch_size: 批大小。
            shuffle: 是否打乱顺序。
        """
        return DataLoader(
            self,
            batch_size=batch_size,
            shuffle=shuffle,
            collate_fn=_collate_induction,
        )


def _collate_induction(
    samples: list[InductionSample],
) -> dict[str, torch.Tensor]:
    """将 InductionSample 列表整理为 batch 字典。

    序列固定等长，无需 padding。若序列长度不同则 pad 到 batch 最大长度。
    """
    max_len = max(s.clean_ids.shape[0] for s in samples)
    batch_size = len(samples)

    clean_ids = torch.zeros(batch_size, max_len, dtype=torch.long)      # (B, max_len)
    corrupt_ids = torch.zeros(batch_size, max_len, dtype=torch.long)    # (B, max_len)
    trigger_pos = torch.zeros(batch_size, dtype=torch.long)             # (B,)
    target_pos = torch.zeros(batch_size, dtype=torch.long)              # (B,)
    answer_token_id = torch.zeros(batch_size, dtype=torch.long)         # (B,)

    for i, sample in enumerate(samples):
        seq_len = sample.clean_ids.shape[0]
        clean_ids[i, :seq_len] = sample.clean_ids             # (seq_len,)
        corrupt_ids[i, :seq_len] = sample.corrupt_ids         # (seq_len,)
        trigger_pos[i] = sample.trigger_pos
        target_pos[i] = sample.target_pos
        answer_token_id[i] = sample.answer_token_id

    return {
        "clean_ids": clean_ids,              # (B, max_len)
        "corrupt_ids": corrupt_ids,          # (B, max_len)
        "trigger_pos": trigger_pos,          # (B,)
        "target_pos": target_pos,            # (B,)
        "answer_token_id": answer_token_id,  # (B,)
    }
