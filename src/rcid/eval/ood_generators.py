"""分布外 (OOD) 测试集生成器。

为每个任务生成 OOD 变体数据集，用于衡量蒸馏方法
在分布偏移下的性能保持率。

OOD 变体设计原则：
- 仅改变任务的表面特征（名字、模板、长度），不改变底层逻辑
- 所有方法使用完全相同的 OOD 测试集（固定种子）
- 保持最小修改原则不变
"""

from __future__ import annotations

import logging
import random
from typing import Sequence

import torch
from torch.utils.data import DataLoader
from transformers import GPT2Tokenizer

from rcid.data.greater_than import GreaterThanDataset
from rcid.data.induction import InductionDataset
from rcid.data.ioi import IOIDataset
from rcid.data.sva import SVADataset

logger = logging.getLogger(__name__)

# ======================================================================
# IOI OOD 变体
# ======================================================================

# 训练用常见名字在 ioi.py 的 NAMES 中；OOD 用不常见名字
IOI_OOD_NAMES: list[str] = [
    " Finn", " Dale", " Ruth", " Hugo", " Carl", " Glen",
    " Dean", " Cole", " Kent", " Ross", " Dawn", " Jade",
    " Troy", " Seth", " Drew", " Chad", " Brad", " Lane",
    " Clay", " Kurt", " Neal", " Kyle", " Wade", " Brent",
]

IOI_OOD_TEMPLATES: list[dict[str, str]] = [
    {
        "setup": "{IO} and{S} were at the shop.",
        "action": "{S} handed a bag to",
    },
    {
        "setup": "{IO} and{S} visited the market.",
        "action": "{S} passed a box to",
    },
    {
        "setup": "At the mall,{IO} and{S} stopped.",
        "action": "{S} gave a card to",
    },
    {
        "setup": "In the garden,{IO} and{S} sat down.",
        "action": "{S} offered a flower to",
    },
]


def make_ioi_ood_datasets(
    n_samples: int = 100,
    tokenizer: GPT2Tokenizer | None = None,
    seed: int = 7777,
) -> dict[str, DataLoader]:
    """生成 IOI 任务的 OOD 变体。

    变体:
        rare_names: 不常见英文名
        new_templates: 不同句式模板
    """
    tok = tokenizer or GPT2Tokenizer.from_pretrained("gpt2")
    ood: dict[str, DataLoader] = {}

    # --- rare_names: 注入不常见名字 ---
    import rcid.data.ioi as ioi_mod
    orig_names = ioi_mod.NAMES
    try:
        ioi_mod.NAMES = IOI_OOD_NAMES
        ds = IOIDataset(n_samples=n_samples, tokenizer=tok, seed=seed)
        ood["rare_names"] = ds.to_dataloader(batch_size=32, shuffle=False)
    finally:
        ioi_mod.NAMES = orig_names

    # --- new_templates: 不同句式模板 ---
    orig_templates = ioi_mod.TEMPLATES
    try:
        ioi_mod.TEMPLATES = IOI_OOD_TEMPLATES
        ds = IOIDataset(n_samples=n_samples, tokenizer=tok, seed=seed + 1)
        ood["new_templates"] = ds.to_dataloader(batch_size=32, shuffle=False)
    finally:
        ioi_mod.TEMPLATES = orig_templates

    return ood


# ======================================================================
# Greater-Than OOD 变体
# ======================================================================

GT_OOD_TEMPLATES: list[dict[str, str]] = [
    {"prefix": "The kingdom flourished from the year", "mid": " to"},
    {"prefix": "The plague spread from the year", "mid": " to"},
    {"prefix": "The empire declined from the year", "mid": " to"},
    {"prefix": "The reform began in the year", "mid": " to"},
]

# 安全世纪前缀：所有后两位 01-98 在 GPT-2 中 tokenize 为相同 token 数
# 19 → " 19xx" 全部 1 token; 21 → " 21xx" 全部 2 tokens
GT_OOD_CENTURIES: list[int] = [19, 21]


def make_gt_ood_datasets(
    n_samples: int = 100,
    tokenizer: GPT2Tokenizer | None = None,
    seed: int = 7777,
) -> dict[str, DataLoader]:
    """生成 Greater-Than 的 OOD 变体。

    变体:
        alt_year_range: 使用 19/21 世纪（训练用 10-18）
        new_templates: 不同上下文模板（用安全世纪前缀）
    """
    tok = tokenizer or GPT2Tokenizer.from_pretrained("gpt2")
    ood: dict[str, DataLoader] = {}
    import rcid.data.greater_than as gt_mod

    # --- alt_year_range: 使用 19/21 世纪 ---
    orig_centuries = gt_mod.CENTURY_PREFIXES
    try:
        gt_mod.CENTURY_PREFIXES = GT_OOD_CENTURIES
        ds = GreaterThanDataset(n_samples=n_samples, tokenizer=tok, seed=seed)
        ood["alt_year_range"] = ds.to_dataloader(batch_size=32, shuffle=False)
    finally:
        gt_mod.CENTURY_PREFIXES = orig_centuries

    # --- new_templates + 安全世纪 ---
    orig_templates = gt_mod.TEMPLATES
    try:
        gt_mod.TEMPLATES = GT_OOD_TEMPLATES
        gt_mod.CENTURY_PREFIXES = GT_OOD_CENTURIES
        ds = GreaterThanDataset(
            n_samples=n_samples, tokenizer=tok, seed=seed + 1,
        )
        ood["new_templates"] = ds.to_dataloader(batch_size=32, shuffle=False)
    finally:
        gt_mod.TEMPLATES = orig_templates
        gt_mod.CENTURY_PREFIXES = orig_centuries

    return ood


# ======================================================================
# Induction Heads OOD 变体
# ======================================================================

def make_induction_ood_datasets(
    n_samples: int = 100,
    tokenizer: GPT2Tokenizer | None = None,
    seed: int = 7777,
) -> dict[str, DataLoader]:
    """生成 Induction Heads 的 OOD 变体。

    变体:
        long_seq: 更长序列 (60 tokens, 训练用 30)
        very_long_seq: 非常长序列 (90 tokens)
    """
    tok = tokenizer or GPT2Tokenizer.from_pretrained("gpt2")
    ood: dict[str, DataLoader] = {}

    ds_long = InductionDataset(
        n_samples=n_samples, tokenizer=tok, seed=seed, seq_len=60,
    )
    ood["long_seq"] = ds_long.to_dataloader(batch_size=32, shuffle=False)

    ds_vlong = InductionDataset(
        n_samples=n_samples, tokenizer=tok, seed=seed + 1, seq_len=90,
    )
    ood["very_long_seq"] = ds_vlong.to_dataloader(batch_size=32, shuffle=False)

    return ood


# ======================================================================
# SVA OOD 变体
# ======================================================================

SVA_OOD_NOUN_PAIRS: list[tuple[str, str]] = [
    (" monk", " monks"), (" clerk", " clerks"), (" fool", " fools"),
    (" scout", " scouts"), (" priest", " priests"), (" knight", " knights"),
    (" pig", " pigs"), (" lord", " lords"), (" rat", " rats"),
    (" saint", " saints"), (" cow", " cows"), (" chief", " chiefs"),
    (" wolf", " wolves"), (" bear", " bears"), (" goat", " goats"),
    (" whale", " whales"), (" witch", " witches"), (" prince", " princes"),
    (" queen", " queens"), (" thief", " thieves"),
    (" frog", " frogs"), (" guest", " guests"),
    (" coach", " coaches"), (" dwarf", " dwarves"),
]

SVA_OOD_PP_TEMPLATES: list[str] = [
    " underneath the{ATTR}",
    " alongside the{ATTR}",
    " before the{ATTR}",
    " against the{ATTR}",
    " within the{ATTR}",
    " between the{ATTR}",
    " towards the{ATTR}",
    " despite the{ATTR}",
    " without the{ATTR}",
    " throughout the{ATTR}",
]


def make_sva_ood_datasets(
    n_samples: int = 100,
    tokenizer: GPT2Tokenizer | None = None,
    seed: int = 7777,
) -> dict[str, DataLoader]:
    """生成 SVA 的 OOD 变体。

    变体:
        multi_attractor: 2-3 个 attractor（训练用 0-2）
        rare_nouns: 不常见名词
    """
    tok = tokenizer or GPT2Tokenizer.from_pretrained("gpt2")
    ood: dict[str, DataLoader] = {}

    # --- multi_attractor: 固定 2 个 attractor ---
    ds_multi = SVADataset(
        n_samples=n_samples, tokenizer=tok, seed=seed, n_attractors=2,
    )
    ood["multi_attractor"] = ds_multi.to_dataloader(
        batch_size=32, shuffle=False,
    )

    # --- rare_nouns: 不常见名词 + 不同 PP 模板 ---
    import rcid.data.sva as sva_mod
    orig_nouns = sva_mod.NOUN_PAIRS
    orig_pp = sva_mod.PP_TEMPLATES
    try:
        sva_mod.NOUN_PAIRS = SVA_OOD_NOUN_PAIRS
        sva_mod.PP_TEMPLATES = SVA_OOD_PP_TEMPLATES
        ds = SVADataset(n_samples=n_samples, tokenizer=tok, seed=seed + 1)
        ood["rare_nouns"] = ds.to_dataloader(batch_size=32, shuffle=False)
    finally:
        sva_mod.NOUN_PAIRS = orig_nouns
        sva_mod.PP_TEMPLATES = orig_pp

    return ood


# ======================================================================
# OODTestGenerator — 统一接口
# ======================================================================

class OODTestGenerator:
    """为所有任务生成 OOD 变体数据集的统一接口。"""

    def __init__(
        self,
        n_samples: int = 100,
        tokenizer: GPT2Tokenizer | None = None,
        seed: int = 7777,
    ) -> None:
        self.n_samples = n_samples
        self.tokenizer = tokenizer
        self.seed = seed

    def generate(self, task: str) -> dict[str, DataLoader]:
        """为指定任务生成所有 OOD 变体。

        Args:
            task: "ioi" | "greater_than" | "induction" | "sva"

        Returns:
            {variant_name: DataLoader}
        """
        makers = {
            "ioi": make_ioi_ood_datasets,
            "greater_than": make_gt_ood_datasets,
            "induction": make_induction_ood_datasets,
            "sva": make_sva_ood_datasets,
        }
        assert task in makers, f"Unknown task: {task!r}"
        return makers[task](
            n_samples=self.n_samples,
            tokenizer=self.tokenizer,
            seed=self.seed,
        )
