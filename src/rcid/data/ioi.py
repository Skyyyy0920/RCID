"""Indirect Object Identification (IOI) 任务数据与对比对构建。

IOI 任务格式：
    "When [IO] and [S] went to the store, [S] gave a drink to" → 模型应预测 [IO]

对比对构造（最小修改原则）：
    clean:   "When Mary and John went to the store, John gave a drink to"
    corrupt: "When Mary and John went to the store, Mary gave a drink to"
    仅将第二次出现的名字 John → Mary，破坏 "重复名字" 信号。
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
# 名字池：选取在 GPT-2 tokenizer 中编码为单 token 的常见英文名
# 名字前带空格，因为在句中出现时 GPT-2 BPE 会包含前导空格
# ──────────────────────────────────────────────────────────────────────
NAMES: list[str] = [
    " Mary", " John", " Alice", " Bob", " Sarah", " David",
    " James", " Emma", " Tom", " Lisa", " Mark", " Anna",
    " Paul", " Jane", " Luke", " Kate", " Jack", " Grace",
    " Sam", " Claire", " Dan", " Emily", " Mike", " Amy",
]

# 场景模板：{IO} = indirect object, {S} = subject (重复名字)
# 格式：(前半句含两名字, 后半句含 S 的第二次出现)
TEMPLATES: list[dict[str, str]] = [
    {
        "setup": "When{IO} and{S} went to the store,",
        "action": "{S} gave a drink to",
    },
    {
        "setup": "When{IO} and{S} went to the park,",
        "action": "{S} gave a book to",
    },
    {
        "setup": "When{IO} and{S} were at the library,",
        "action": "{S} handed a letter to",
    },
    {
        "setup": "When{IO} and{S} arrived at the office,",
        "action": "{S} passed a note to",
    },
    {
        "setup": "When{IO} and{S} met at the cafe,",
        "action": "{S} offered a gift to",
    },
    {
        "setup": "When{IO} and{S} sat in the classroom,",
        "action": "{S} gave a pencil to",
    },
    {
        "setup": "When{IO} and{S} stood in the kitchen,",
        "action": "{S} handed a plate to",
    },
]


@dataclass
class IOISample:
    """单个 IOI 对比对的全部信息。"""

    clean_text: str
    corrupt_text: str
    clean_ids: torch.Tensor     # (seq_len,)
    corrupt_ids: torch.Tensor   # (seq_len,)
    io_token_pos: int           # indirect object 在 token 序列中的位置
    s2_token_pos: int           # subject 第二次出现在 token 序列中的位置
    answer_token_id: int        # 正确答案（IO 名字）的 token id


class IOIDataset(Dataset[IOISample]):
    """IOI 对比数据集，可配置生成 N 对 (clean, corrupt) 样本。

    每对数据严格遵循最小修改原则：
    clean 和 corrupt 仅在 subject 第二次出现的位置不同。
    """

    def __init__(
        self,
        n_samples: int = 500,
        tokenizer: GPT2Tokenizer | None = None,
        seed: int = 42,
    ) -> None:
        """生成 n_samples 个 IOI 对比对。

        Args:
            n_samples: 对比对数量。
            tokenizer: GPT-2 tokenizer；若为 None 则自动加载。
            seed: 随机种子，保证可复现。
        """
        self.tokenizer = tokenizer or GPT2Tokenizer.from_pretrained("gpt2")
        self.n_samples = n_samples

        self._validate_names()

        rng = random.Random(seed)
        self.samples: list[IOISample] = []

        for _ in range(n_samples):
            sample = self._generate_one(rng)
            self.samples.append(sample)

        logger.info("IOIDataset: generated %d contrastive pairs", len(self.samples))

    # ------------------------------------------------------------------
    # 内部方法
    # ------------------------------------------------------------------

    def _validate_names(self) -> None:
        """验证所有名字在 GPT-2 tokenizer 中编码为单 token。"""
        for name in NAMES:
            token_ids = self.tokenizer.encode(name)
            assert len(token_ids) == 1, (
                f"Name '{name}' tokenizes to {len(token_ids)} tokens "
                f"({self.tokenizer.convert_ids_to_tokens(token_ids)}), "
                f"expected exactly 1. Remove it from NAMES."
            )

    def _generate_one(self, rng: random.Random) -> IOISample:
        """生成单个 IOI 对比对。

        构造逻辑：
        1. 随机选两个不同名字 IO, S
        2. 随机选模板
        3. clean = template(IO, S)       → 答案是 IO
        4. corrupt = 后半句的 {S} 替换为 IO → 破坏重复信号
        """
        # 选两个不同的名字
        io_name, s_name = rng.sample(NAMES, 2)

        # 选模板
        template = rng.choice(TEMPLATES)

        # 构建 clean 文本
        clean_text = (
            template["setup"].format(IO=io_name, S=s_name)
            + " "
            + template["action"].format(S=s_name)
        )

        # 构建 corrupt 文本：仅替换后半句中的 S → IO（最小修改）
        corrupt_text = (
            template["setup"].format(IO=io_name, S=s_name)
            + " "
            + template["action"].format(S=io_name)
        )

        # tokenize
        clean_ids = self.tokenizer.encode(clean_text)    # list[int]
        corrupt_ids = self.tokenizer.encode(corrupt_text)  # list[int]

        assert len(clean_ids) == len(corrupt_ids), (
            f"Token length mismatch: clean={len(clean_ids)}, "
            f"corrupt={len(corrupt_ids)}.\n"
            f"  clean:   {clean_text!r}\n"
            f"  corrupt: {corrupt_text!r}"
        )

        # 定位关键 token 位置
        io_token_id = self.tokenizer.encode(io_name)[0]
        s_token_id = self.tokenizer.encode(s_name)[0]

        # IO 在 clean 中的位置（第一次出现）
        io_token_pos = self._find_token_pos(clean_ids, io_token_id, occurrence=1)
        # S 在 clean 中第二次出现的位置（后半句）
        s2_token_pos = self._find_token_pos(clean_ids, s_token_id, occurrence=2)

        # 验证最小修改：clean 和 corrupt 应只在 s2_token_pos 位置不同
        diff_positions = [
            i for i in range(len(clean_ids))
            if clean_ids[i] != corrupt_ids[i]
        ]
        assert diff_positions == [s2_token_pos], (
            f"Minimal modification violated: diffs at {diff_positions}, "
            f"expected only [{s2_token_pos}].\n"
            f"  clean:   {self.tokenizer.convert_ids_to_tokens(clean_ids)}\n"
            f"  corrupt: {self.tokenizer.convert_ids_to_tokens(corrupt_ids)}"
        )

        return IOISample(
            clean_text=clean_text,
            corrupt_text=corrupt_text,
            clean_ids=torch.tensor(clean_ids, dtype=torch.long),      # (seq_len,)
            corrupt_ids=torch.tensor(corrupt_ids, dtype=torch.long),  # (seq_len,)
            io_token_pos=io_token_pos,
            s2_token_pos=s2_token_pos,
            answer_token_id=io_token_id,
        )

    @staticmethod
    def _find_token_pos(
        token_ids: list[int],
        target_id: int,
        occurrence: int,
    ) -> int:
        """在 token 序列中找到目标 token 第 N 次出现的位置。

        Args:
            token_ids: token id 列表。
            target_id: 要查找的 token id。
            occurrence: 第几次出现（1-indexed）。

        Returns:
            token 位置索引。
        """
        count = 0
        for i, tid in enumerate(token_ids):
            if tid == target_id:
                count += 1
                if count == occurrence:
                    return i

        raise ValueError(
            f"Token id {target_id} does not appear {occurrence} time(s) "
            f"in sequence of length {len(token_ids)}"
        )

    # ------------------------------------------------------------------
    # Dataset 接口
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return self.n_samples

    def __getitem__(self, idx: int) -> IOISample:
        return self.samples[idx]

    # ------------------------------------------------------------------
    # DataLoader 工厂
    # ------------------------------------------------------------------

    def to_dataloader(
        self,
        batch_size: int = 32,
        shuffle: bool = True,
    ) -> DataLoader[dict[str, torch.Tensor]]:
        """创建 DataLoader，将 IOISample 整理为 batch 字典。

        返回的每个 batch 是一个字典:
            clean_ids:       (B, seq_len)
            corrupt_ids:     (B, seq_len)
            io_token_pos:    (B,)
            s2_token_pos:    (B,)
            answer_token_id: (B,)

        Args:
            batch_size: 批大小。
            shuffle: 是否打乱顺序。
        """
        return DataLoader(
            self,
            batch_size=batch_size,
            shuffle=shuffle,
            collate_fn=_collate_ioi,
        )


def _collate_ioi(
    samples: list[IOISample],
) -> dict[str, torch.Tensor]:
    """将 IOISample 列表整理为 padded batch 字典。

    所有序列 pad 到 batch 内最长长度，pad token 使用 0。
    """
    max_len = max(s.clean_ids.shape[0] for s in samples)
    batch_size = len(samples)

    clean_ids = torch.zeros(batch_size, max_len, dtype=torch.long)      # (B, max_len)
    corrupt_ids = torch.zeros(batch_size, max_len, dtype=torch.long)    # (B, max_len)
    io_token_pos = torch.zeros(batch_size, dtype=torch.long)            # (B,)
    s2_token_pos = torch.zeros(batch_size, dtype=torch.long)            # (B,)
    answer_token_id = torch.zeros(batch_size, dtype=torch.long)         # (B,)

    for i, sample in enumerate(samples):
        seq_len = sample.clean_ids.shape[0]
        clean_ids[i, :seq_len] = sample.clean_ids           # (seq_len,)
        corrupt_ids[i, :seq_len] = sample.corrupt_ids       # (seq_len,)
        io_token_pos[i] = sample.io_token_pos
        s2_token_pos[i] = sample.s2_token_pos
        answer_token_id[i] = sample.answer_token_id

    return {
        "clean_ids": clean_ids,              # (B, max_len)
        "corrupt_ids": corrupt_ids,          # (B, max_len)
        "io_token_pos": io_token_pos,        # (B,)
        "s2_token_pos": s2_token_pos,        # (B,)
        "answer_token_id": answer_token_id,  # (B,)
    }
