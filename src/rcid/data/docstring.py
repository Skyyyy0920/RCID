"""Docstring 参数补全任务数据与对比对构建。

Docstring 任务测试模型是否能跟踪 Python 函数签名中的参数名，
并在 docstring 的 :param 段落中正确补全（Heimersheim & Janiak, 2023）。

对比对构造（最小修改原则）：
    clean:   def f(a, b, c):\n  '''\n  :param a: ...\n  :param b: ...\n  :param
             → 模型应预测 " c"
    corrupt: def f(a, b, X):\n  '''\n  :param a: ...\n  :param b: ...\n  :param
             → 仅替换签名中第三个参数名 c → X
    差值捕捉 signature-tracking 回路的因果效应。
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
# 参数名池：每个必须在 GPT-2 中编码为 " name" → 单 token（含前导空格）
# ──────────────────────────────────────────────────────────────────────
PARAM_NAMES: list[str] = [
    " data", " config", " name", " value", " count",
    " size", " path", " mode", " rate", " text",
    " source", " target", " level", " index", " key",
    " file", " output", " input", " limit", " offset",
    " weight", " color", " shape", " alpha", " beta",
    " radius", " angle", " axis", " label", " status",
    " result", " query", " token", " batch", " width",
    " height", " depth", " port", " message", " address",
]

# 函数名池（用于 "def {func}(...)" ）
FUNC_NAMES: list[str] = [
    "process", "calculate", "send", "load", "train",
    "draw", "search", "build", "parse", "update",
    "create", "delete", "fetch", "save", "check",
    "convert", "validate", "handle", "filter", "sort",
]

# Docstring 描述模板：{FUNC} 占位符
DOC_DESCS: list[str] = [
    "Process the input.",
    "Calculate the result.",
    "Send the request.",
    "Load the resource.",
    "Train the model.",
    "Draw the element.",
    "Search the items.",
    "Build the output.",
    "Parse the content.",
    "Update the record.",
]

# 参数描述模板：{PARAM} 占位符
PARAM_DESCS: list[str] = [
    "the {PARAM}",
    "input {PARAM}",
    "{PARAM} value",
    "a {PARAM}",
]


@dataclass
class DocstringSample:
    """单个 Docstring 对比对的全部信息。"""

    clean_text: str
    corrupt_text: str
    clean_ids: torch.Tensor     # (seq_len,)
    corrupt_ids: torch.Tensor   # (seq_len,)
    target_pos: int             # 末尾位置（logits 在此预测参数名）
    answer_token_id: int        # clean 中应预测的参数名 token id
    corrupt_token_id: int       # corrupt 中签名里替换后的参数名 token id
    param_name_pos: int         # 签名中被替换的参数名在 token 序列中的位置


class DocstringDataset(Dataset[DocstringSample]):
    """Docstring 参数补全对比数据集。

    生成 Python 函数定义 + docstring，clean/corrupt 仅在签名中
    第三个参数名不同（最小修改），末尾处模型应预测对应参数名。
    """

    def __init__(
        self,
        n_samples: int = 500,
        tokenizer: GPT2Tokenizer | None = None,
        seed: int = 42,
    ) -> None:
        """生成 n_samples 个 Docstring 对比对。

        Args:
            n_samples: 对比对数量。
            tokenizer: GPT-2 tokenizer；若为 None 则自动加载。
            seed: 随机种子，保证可复现。
        """
        self.tokenizer = tokenizer or GPT2Tokenizer.from_pretrained("gpt2")
        self.n_samples = n_samples

        self._validate_params()

        rng = random.Random(seed)
        self.samples: list[DocstringSample] = []

        for _ in range(n_samples):
            sample = self._generate_one(rng)
            self.samples.append(sample)

        logger.info(
            "DocstringDataset: generated %d contrastive pairs",
            len(self.samples),
        )

    # ------------------------------------------------------------------
    # 内部方法
    # ------------------------------------------------------------------

    def _validate_params(self) -> None:
        """验证所有参数名在 GPT-2 tokenizer 中编码为单 token。"""
        for param in PARAM_NAMES:
            ids = self.tokenizer.encode(param)
            assert len(ids) == 1, (
                f"Param '{param}' encodes to {len(ids)} tokens, expected 1"
            )

    def _generate_one(self, rng: random.Random) -> DocstringSample:
        """生成单个 Docstring 对比对。

        构造逻辑（3 参数函数）：
        1. 选 3 个不同参数名 (p1, p2, p3) + 1 个替换参数 p_alt
        2. Clean 签名: def func(p1, p2, p3):
        3. Corrupt 签名: def func(p1, p2, p_alt):
        4. 共同 docstring: :param p1: desc\\n    :param p2: desc\\n    :param
        5. 模型在末尾应预测 clean→p3, corrupt→p_alt
        """
        # 选 4 个不同参数名
        params = rng.sample(PARAM_NAMES, 4)
        p1, p2, p3, p_alt = params

        # 选函数名和描述
        func = rng.choice(FUNC_NAMES)
        doc_desc = rng.choice(DOC_DESCS)

        # 参数描述（去掉前导空格得到裸名）
        desc_tmpl = rng.choice(PARAM_DESCS)
        p1_bare, p2_bare = p1.strip(), p2.strip()
        desc1 = desc_tmpl.format(PARAM=p1_bare)
        desc2 = desc_tmpl.format(PARAM=p2_bare)

        # 构建文本
        sig_clean = f"def {func}({p1.strip()},{p2},{p3}):"
        sig_corrupt = f"def {func}({p1.strip()},{p2},{p_alt}):"
        body = (
            f'\n    """{doc_desc}\n'
            f"    :param{p1}: {desc1}\n"
            f"    :param{p2}: {desc2}\n"
            f"    :param"
        )
        clean_text = sig_clean + body
        corrupt_text = sig_corrupt + body

        # tokenize
        clean_ids = self.tokenizer.encode(clean_text)
        corrupt_ids = self.tokenizer.encode(corrupt_text)

        assert len(clean_ids) == len(corrupt_ids), (
            f"Token length mismatch: clean={len(clean_ids)} vs "
            f"corrupt={len(corrupt_ids)}.\n"
            f"  clean:   {clean_text!r}\n  corrupt: {corrupt_text!r}"
        )

        # 找差异位置（应恰好 1 处，即签名中第三个参数名）
        diff_positions = [
            i for i in range(len(clean_ids))
            if clean_ids[i] != corrupt_ids[i]
        ]
        assert len(diff_positions) == 1, (
            f"Expected 1 diff, got {len(diff_positions)} at {diff_positions}.\n"
            f"  clean:   {clean_text!r}\n  corrupt: {corrupt_text!r}"
        )
        param_name_pos = diff_positions[0]

        # target_pos: 最后一个 token（logits 在此预测下一个 token = 参数名）
        target_pos = len(clean_ids) - 1

        # answer token ids
        answer_token_id = self.tokenizer.encode(p3)[0]
        corrupt_token_id = self.tokenizer.encode(p_alt)[0]

        return DocstringSample(
            clean_text=clean_text,
            corrupt_text=corrupt_text,
            clean_ids=torch.tensor(clean_ids, dtype=torch.long),
            corrupt_ids=torch.tensor(corrupt_ids, dtype=torch.long),
            target_pos=target_pos,
            answer_token_id=answer_token_id,
            corrupt_token_id=corrupt_token_id,
            param_name_pos=param_name_pos,
        )

    # ------------------------------------------------------------------
    # Dataset 接口
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return self.n_samples

    def __getitem__(self, idx: int) -> DocstringSample:
        return self.samples[idx]

    # ------------------------------------------------------------------
    # DataLoader 工厂
    # ------------------------------------------------------------------

    def to_dataloader(
        self,
        batch_size: int = 32,
        shuffle: bool = True,
    ) -> DataLoader[dict[str, torch.Tensor]]:
        """创建 DataLoader，将 DocstringSample 整理为 batch 字典。

        返回的每个 batch 是一个字典:
            clean_ids:          (B, seq_len)
            corrupt_ids:        (B, seq_len)
            target_pos:         (B,)
            answer_token_id:    (B,)
            corrupt_token_id:   (B,)
            param_name_pos:     (B,)
        """
        return DataLoader(
            self,
            batch_size=batch_size,
            shuffle=shuffle,
            collate_fn=_collate_docstring,
        )


def _collate_docstring(
    samples: list[DocstringSample],
) -> dict[str, torch.Tensor]:
    """将 DocstringSample 列表整理为 padded batch 字典。"""
    max_len = max(s.clean_ids.shape[0] for s in samples)
    batch_size = len(samples)

    clean_ids = torch.zeros(batch_size, max_len, dtype=torch.long)    # (B, max_len)
    corrupt_ids = torch.zeros(batch_size, max_len, dtype=torch.long)  # (B, max_len)
    target_pos = torch.zeros(batch_size, dtype=torch.long)            # (B,)
    answer_token_id = torch.zeros(batch_size, dtype=torch.long)       # (B,)
    corrupt_token_id = torch.zeros(batch_size, dtype=torch.long)      # (B,)
    param_name_pos = torch.zeros(batch_size, dtype=torch.long)        # (B,)

    for i, sample in enumerate(samples):
        seq_len = sample.clean_ids.shape[0]
        clean_ids[i, :seq_len] = sample.clean_ids
        corrupt_ids[i, :seq_len] = sample.corrupt_ids
        target_pos[i] = sample.target_pos
        answer_token_id[i] = sample.answer_token_id
        corrupt_token_id[i] = sample.corrupt_token_id
        param_name_pos[i] = sample.param_name_pos

    return {
        "clean_ids": clean_ids,                # (B, max_len)
        "corrupt_ids": corrupt_ids,            # (B, max_len)
        "target_pos": target_pos,              # (B,)
        "answer_token_id": answer_token_id,    # (B,)
        "corrupt_token_id": corrupt_token_id,  # (B,)
        "param_name_pos": param_name_pos,      # (B,)
    }
