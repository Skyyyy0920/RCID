"""Greater-Than 数字比较任务数据与对比对构建。

Greater-Than 任务格式：
    "The war lasted from 1412 to 15" → 模型应预测 > 12 的两位数字

对比对构造（最小修改原则）：
    clean:   "The war lasted from 1412 to 15"  （阈值 = 12）
    corrupt: "The war lasted from 1489 to 15"  （阈值 = 89）
    仅替换起始年份的后两位，改变 "大于" 约束的阈值。
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
# 模板池：{YEAR} 占位符将被替换为完整四位数年份
# 句子在 "to XX" 前截断，XX 的世纪部分由模板固定
# ──────────────────────────────────────────────────────────────────────
TEMPLATES: list[dict[str, str]] = [
    {
        "prefix": "The war lasted from the year",
        "mid": " to",
    },
    {
        "prefix": "The event ran from the year",
        "mid": " to",
    },
    {
        "prefix": "The dynasty ruled from the year",
        "mid": " to",
    },
    {
        "prefix": "The project spanned from the year",
        "mid": " to",
    },
    {
        "prefix": "The conflict extended from the year",
        "mid": " to",
    },
    {
        "prefix": "The expedition took place from the year",
        "mid": " to",
    },
    {
        "prefix": "The construction lasted from the year",
        "mid": " to",
    },
]

# 世纪前缀：仅使用 "安全" 世纪——所有后两位 01-98 在 GPT-2 中
# tokenize 为相同 token 数。排除 10/12/14/18（部分年份被 BPE 合并
# 为单 token，导致 clean/corrupt 长度不一致）。
CENTURY_PREFIXES: list[int] = [11, 13, 15, 16, 17]


@dataclass
class GreaterThanSample:
    """单个 Greater-Than 对比对的全部信息。"""

    clean_text: str
    corrupt_text: str
    clean_ids: torch.Tensor     # (seq_len,)
    corrupt_ids: torch.Tensor   # (seq_len,)
    year_token_pos: int         # 起始年份后两位 token 在序列中的位置
    clean_threshold: int        # clean 版本的阈值（后两位数字）
    corrupt_threshold: int      # corrupt 版本的阈值（后两位数字）


class GreaterThanDataset(Dataset[GreaterThanSample]):
    """Greater-Than 对比数据集，可配置生成 N 对 (clean, corrupt) 样本。

    每对数据严格遵循最小修改原则：
    clean 和 corrupt 仅在起始年份的后两位不同。
    """

    def __init__(
        self,
        n_samples: int = 500,
        tokenizer: GPT2Tokenizer | None = None,
        seed: int = 42,
    ) -> None:
        """生成 n_samples 个 Greater-Than 对比对。

        Args:
            n_samples: 对比对数量。
            tokenizer: GPT-2 tokenizer；若为 None 则自动加载。
            seed: 随机种子，保证可复现。
        """
        self.tokenizer = tokenizer or GPT2Tokenizer.from_pretrained("gpt2")
        self.n_samples = n_samples

        rng = random.Random(seed)
        self.samples: list[GreaterThanSample] = []

        for _ in range(n_samples):
            sample = self._generate_one(rng)
            self.samples.append(sample)

        logger.info(
            "GreaterThanDataset: generated %d contrastive pairs",
            len(self.samples),
        )

    # ------------------------------------------------------------------
    # 内部方法
    # ------------------------------------------------------------------

    def _generate_one(self, rng: random.Random) -> GreaterThanSample:
        """生成单个 Greater-Than 对比对。

        构造逻辑：
        1. 随机选世纪前缀 (10-18) 和两个不同的后两位数字
        2. 构建 clean = "... from the year XXYY to XX"
        3. 构建 corrupt = "... from the year XXZZ to XX"
        4. 仅 YY ≠ ZZ，其余完全相同
        """
        template = rng.choice(TEMPLATES)
        century = rng.choice(CENTURY_PREFIXES)

        # 选两个不同的后两位（01-98），确保阈值有意义
        clean_suffix, corrupt_suffix = rng.sample(range(1, 99), 2)

        century_str = str(century)                          # e.g. "14"
        clean_year = f"{century}{clean_suffix:02d}"         # e.g. "1412"
        corrupt_year = f"{century}{corrupt_suffix:02d}"     # e.g. "1489"

        # 构建文本：截断在 "to XX" — XX 是世纪前缀
        # 模型需要预测世纪内的后两位，且应 > 阈值
        clean_text = (
            f"{template['prefix']} {clean_year}"
            f"{template['mid']} {century_str}"
        )
        corrupt_text = (
            f"{template['prefix']} {corrupt_year}"
            f"{template['mid']} {century_str}"
        )

        # tokenize
        clean_ids = self.tokenizer.encode(clean_text)       # list[int]
        corrupt_ids = self.tokenizer.encode(corrupt_text)   # list[int]

        assert len(clean_ids) == len(corrupt_ids), (
            f"Token length mismatch: clean={len(clean_ids)}, "
            f"corrupt={len(corrupt_ids)}.\n"
            f"  clean:   {clean_text!r}\n"
            f"  corrupt: {corrupt_text!r}"
        )

        # 定位年份后两位 token 的位置
        # GPT-2 会把 " 1412" 编码为多个 token，我们需要找到后两位的位置
        # 策略：找到 clean 和 corrupt 的第一个差异位置
        diff_positions = [
            i for i in range(len(clean_ids))
            if clean_ids[i] != corrupt_ids[i]
        ]

        assert len(diff_positions) >= 1, (
            f"No difference found between clean and corrupt.\n"
            f"  clean:   {clean_text!r}\n"
            f"  corrupt: {corrupt_text!r}"
        )

        # 年份后两位对应的 token 位置（取第一个差异位置）
        year_token_pos = diff_positions[0]

        # 验证：差异应只出现在年份区域（连续位置）
        if len(diff_positions) > 1:
            assert diff_positions[-1] - diff_positions[0] < 3, (
                f"Differences span too wide: {diff_positions}.\n"
                f"  clean:   {self.tokenizer.convert_ids_to_tokens(clean_ids)}\n"
                f"  corrupt: {self.tokenizer.convert_ids_to_tokens(corrupt_ids)}"
            )

        return GreaterThanSample(
            clean_text=clean_text,
            corrupt_text=corrupt_text,
            clean_ids=torch.tensor(clean_ids, dtype=torch.long),      # (seq_len,)
            corrupt_ids=torch.tensor(corrupt_ids, dtype=torch.long),  # (seq_len,)
            year_token_pos=year_token_pos,
            clean_threshold=clean_suffix,
            corrupt_threshold=corrupt_suffix,
        )

    # ------------------------------------------------------------------
    # Dataset 接口
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return self.n_samples

    def __getitem__(self, idx: int) -> GreaterThanSample:
        return self.samples[idx]

    # ------------------------------------------------------------------
    # DataLoader 工厂
    # ------------------------------------------------------------------

    def to_dataloader(
        self,
        batch_size: int = 32,
        shuffle: bool = True,
    ) -> DataLoader[dict[str, torch.Tensor]]:
        """创建 DataLoader，将 GreaterThanSample 整理为 batch 字典。

        返回的每个 batch 是一个字典:
            clean_ids:         (B, seq_len)
            corrupt_ids:       (B, seq_len)
            year_token_pos:    (B,)
            clean_threshold:   (B,)
            corrupt_threshold: (B,)

        Args:
            batch_size: 批大小。
            shuffle: 是否打乱顺序。
        """
        return DataLoader(
            self,
            batch_size=batch_size,
            shuffle=shuffle,
            collate_fn=_collate_greater_than,
        )


def _collate_greater_than(
    samples: list[GreaterThanSample],
) -> dict[str, torch.Tensor]:
    """将 GreaterThanSample 列表整理为 padded batch 字典。

    所有序列 pad 到 batch 内最长长度，pad token 使用 0。
    """
    max_len = max(s.clean_ids.shape[0] for s in samples)
    batch_size = len(samples)

    clean_ids = torch.zeros(batch_size, max_len, dtype=torch.long)        # (B, max_len)
    corrupt_ids = torch.zeros(batch_size, max_len, dtype=torch.long)      # (B, max_len)
    year_token_pos = torch.zeros(batch_size, dtype=torch.long)            # (B,)
    clean_threshold = torch.zeros(batch_size, dtype=torch.long)           # (B,)
    corrupt_threshold = torch.zeros(batch_size, dtype=torch.long)         # (B,)

    for i, sample in enumerate(samples):
        seq_len = sample.clean_ids.shape[0]
        clean_ids[i, :seq_len] = sample.clean_ids             # (seq_len,)
        corrupt_ids[i, :seq_len] = sample.corrupt_ids         # (seq_len,)
        year_token_pos[i] = sample.year_token_pos
        clean_threshold[i] = sample.clean_threshold
        corrupt_threshold[i] = sample.corrupt_threshold

    return {
        "clean_ids": clean_ids,                # (B, max_len)
        "corrupt_ids": corrupt_ids,            # (B, max_len)
        "year_token_pos": year_token_pos,      # (B,)
        "clean_threshold": clean_threshold,    # (B,)
        "corrupt_threshold": corrupt_threshold,  # (B,)
    }
