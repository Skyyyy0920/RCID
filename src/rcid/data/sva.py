"""Subject-Verb Agreement 任务数据与对比对构建。

SVA 任务测试模型是否能跨越干扰成分正确执行主谓一致
（Finlayson et al., 2021; Linzen et al., 2016）。

对比对构造（最小修改原则）：
    clean:   "The cat near the large dogs"    → 模型在末尾应偏好单数动词（is/runs）
    corrupt: "The cats near the large dogs"   → 仅翻转主语单复数
    差值捕捉 subject-verb agreement 回路的因果效应。
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
# 名词池：(singular, plural)，均需在 GPT-2 中编码为单 token（含前导空格）
# ──────────────────────────────────────────────────────────────────────
NOUN_PAIRS: list[tuple[str, str]] = [
    (" cat", " cats"), (" dog", " dogs"), (" boy", " boys"),
    (" girl", " girls"), (" man", " men"), (" child", " children"),
    (" nurse", " nurses"), (" doctor", " doctors"),
    (" teacher", " teachers"), (" lawyer", " lawyers"),
    (" artist", " artists"), (" writer", " writers"),
    (" guard", " guards"), (" chef", " chefs"),
    (" king", " kings"), (" bird", " birds"),
    (" horse", " horses"), (" farmer", " farmers"),
    (" student", " students"), (" officer", " officers"),
    (" senator", " senators"), (" judge", " judges"),
    (" lion", " lions"), (" ship", " ships"),
]

# 介词短语模板：{ATTR} 是干扰名词（attractor）的占位符
# 使用固定冠词 "the" 保证 token 数一致
PP_TEMPLATES: list[str] = [
    " near the{ATTR}",
    " behind the{ATTR}",
    " beside the{ATTR}",
    " above the{ATTR}",
    " below the{ATTR}",
    " with the{ATTR}",
    " from the{ATTR}",
    " across the{ATTR}",
    " around the{ATTR}",
    " among the{ATTR}",
]

# 单数/复数动词对：用于评估准确率
# (singular_form, plural_form)，均为单 token
VERB_PAIRS: list[tuple[str, str]] = [
    (" is", " are"), (" was", " were"),
    (" runs", " run"), (" walks", " walk"),
    (" works", " work"), (" likes", " like"),
    (" writes", " write"), (" plays", " play"),
    (" reads", " read"), (" knows", " know"),
    (" says", " say"), (" makes", " make"),
    (" takes", " take"), (" gives", " give"),
    (" lives", " live"),
]


@dataclass
class SVASample:
    """单个 SVA 对比对的全部信息。"""

    clean_text: str
    corrupt_text: str
    clean_ids: torch.Tensor     # (seq_len,)
    corrupt_ids: torch.Tensor   # (seq_len,)
    subject_pos: int            # 主语 token 在序列中的位置
    verb_pos: int               # 末尾位置（logits 在此预测动词）
    singular_verb_ids: list[int]   # 所有单数动词 token id
    plural_verb_ids: list[int]     # 所有复数动词 token id
    clean_is_singular: bool     # clean 的主语是否为单数


class SVADataset(Dataset[SVASample]):
    """Subject-Verb Agreement 对比数据集。

    生成 "主语 + 介词短语(干扰) + ___" 模式的对比样本对。
    clean/corrupt 仅在主语的单复数上不同（最小修改）。
    """

    def __init__(
        self,
        n_samples: int = 500,
        tokenizer: GPT2Tokenizer | None = None,
        seed: int = 42,
        n_attractors: int | None = None,
    ) -> None:
        """生成 n_samples 个 SVA 对比对。

        Args:
            n_samples: 对比对数量。
            tokenizer: GPT-2 tokenizer；若为 None 则自动加载。
            seed: 随机种子，保证可复现。
            n_attractors: 干扰介词短语数量（0/1/2）。
                None 则随机选择 0-2。
        """
        self.tokenizer = tokenizer or GPT2Tokenizer.from_pretrained("gpt2")
        self.n_samples = n_samples

        self._validate_tokens()

        # 预计算动词 token id
        self.singular_verb_ids = [
            self.tokenizer.encode(sv)[0] for sv, _ in VERB_PAIRS
        ]
        self.plural_verb_ids = [
            self.tokenizer.encode(pv)[0] for _, pv in VERB_PAIRS
        ]

        rng = random.Random(seed)
        self.samples: list[SVASample] = []

        for _ in range(n_samples):
            sample = self._generate_one(rng, n_attractors)
            self.samples.append(sample)

        logger.info(
            "SVADataset: generated %d contrastive pairs", len(self.samples),
        )

    # ------------------------------------------------------------------
    # 内部方法
    # ------------------------------------------------------------------

    def _validate_tokens(self) -> None:
        """验证所有名词和动词在 GPT-2 tokenizer 中编码为单 token。"""
        for sg, pl in NOUN_PAIRS:
            for word in (sg, pl):
                ids = self.tokenizer.encode(word)
                assert len(ids) == 1, (
                    f"Noun '{word}' encodes to {len(ids)} tokens, expected 1"
                )
        for sv, pv in VERB_PAIRS:
            for word in (sv, pv):
                ids = self.tokenizer.encode(word)
                assert len(ids) == 1, (
                    f"Verb '{word}' encodes to {len(ids)} tokens, expected 1"
                )

    def _generate_one(
        self, rng: random.Random, n_attractors: int | None,
    ) -> SVASample:
        """生成单个 SVA 对比对。

        构造逻辑：
        1. 选主语名词对（singular/plural）
        2. 选 0-2 个介词短语，每个含一个干扰名词（attractor）
        3. Clean: 使用单数主语 + 复数 attractor（制造干扰）
        4. Corrupt: 使用复数主语 + 相同复数 attractor
        5. 两者仅在主语位置不同（单 token 差异）
        """
        # 选主语名词对
        subj_sg, subj_pl = rng.choice(NOUN_PAIRS)

        # 决定介词短语数量
        n_pp = n_attractors if n_attractors is not None else rng.randint(0, 2)
        n_pp = max(0, min(2, n_pp))

        # 生成介词短语（attractor 始终用复数以制造最大干扰）
        pp_text = ""
        if n_pp > 0:
            pp_templates = rng.sample(PP_TEMPLATES, n_pp)
            for pp_tmpl in pp_templates:
                attr_sg, attr_pl = rng.choice(NOUN_PAIRS)
                # Attractor 用复数（对单数主语形成干扰）
                pp_text += pp_tmpl.format(ATTR=attr_pl)

        # 构建 clean（单数主语）和 corrupt（复数主语）
        clean_text = "The" + subj_sg + pp_text
        corrupt_text = "The" + subj_pl + pp_text

        # tokenize
        clean_ids = self.tokenizer.encode(clean_text)
        corrupt_ids = self.tokenizer.encode(corrupt_text)

        assert len(clean_ids) == len(corrupt_ids), (
            f"Token length mismatch: clean={len(clean_ids)} vs "
            f"corrupt={len(corrupt_ids)}.\n"
            f"  clean:   {clean_text!r}\n  corrupt: {corrupt_text!r}"
        )

        # 定位关键位置
        # subject_pos: "The" is token 0, subject is token 1
        subject_pos = 1
        # verb_pos: 最后一个 token 位置（logits[verb_pos] 预测下一个 token = 动词）
        verb_pos = len(clean_ids) - 1

        # 验证最小修改：仅 subject_pos 不同
        diff_positions = [
            i for i in range(len(clean_ids))
            if clean_ids[i] != corrupt_ids[i]
        ]
        assert diff_positions == [subject_pos], (
            f"Minimal modification violated: diffs at {diff_positions}, "
            f"expected [{subject_pos}].\n"
            f"  clean:   {clean_text!r}\n  corrupt: {corrupt_text!r}"
        )

        return SVASample(
            clean_text=clean_text,
            corrupt_text=corrupt_text,
            clean_ids=torch.tensor(clean_ids, dtype=torch.long),
            corrupt_ids=torch.tensor(corrupt_ids, dtype=torch.long),
            subject_pos=subject_pos,
            verb_pos=verb_pos,
            singular_verb_ids=self.singular_verb_ids,
            plural_verb_ids=self.plural_verb_ids,
            clean_is_singular=True,
        )

    # ------------------------------------------------------------------
    # Dataset 接口
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return self.n_samples

    def __getitem__(self, idx: int) -> SVASample:
        return self.samples[idx]

    # ------------------------------------------------------------------
    # DataLoader 工厂
    # ------------------------------------------------------------------

    def to_dataloader(
        self,
        batch_size: int = 32,
        shuffle: bool = True,
    ) -> DataLoader[dict[str, torch.Tensor]]:
        """创建 DataLoader，将 SVASample 整理为 batch 字典。

        返回的每个 batch 是一个字典:
            clean_ids:          (B, seq_len)
            corrupt_ids:        (B, seq_len)
            subject_pos:        (B,)
            verb_pos:           (B,)
            singular_verb_ids:  (n_verb_pairs,) — 所有单数动词 id
            plural_verb_ids:    (n_verb_pairs,) — 所有复数动词 id

        Args:
            batch_size: 批大小。
            shuffle: 是否打乱顺序。
        """
        return DataLoader(
            self,
            batch_size=batch_size,
            shuffle=shuffle,
            collate_fn=_collate_sva,
        )


def _collate_sva(
    samples: list[SVASample],
) -> dict[str, torch.Tensor]:
    """将 SVASample 列表整理为 padded batch 字典。"""
    max_len = max(s.clean_ids.shape[0] for s in samples)
    batch_size = len(samples)

    clean_ids = torch.zeros(batch_size, max_len, dtype=torch.long)    # (B, max_len)
    corrupt_ids = torch.zeros(batch_size, max_len, dtype=torch.long)  # (B, max_len)
    subject_pos = torch.zeros(batch_size, dtype=torch.long)           # (B,)
    verb_pos = torch.zeros(batch_size, dtype=torch.long)              # (B,)

    for i, sample in enumerate(samples):
        seq_len = sample.clean_ids.shape[0]
        clean_ids[i, :seq_len] = sample.clean_ids
        corrupt_ids[i, :seq_len] = sample.corrupt_ids
        subject_pos[i] = sample.subject_pos
        verb_pos[i] = sample.verb_pos

    # 动词 id 对所有样本相同，只存一份
    singular_verb_ids = torch.tensor(
        samples[0].singular_verb_ids, dtype=torch.long,
    )
    plural_verb_ids = torch.tensor(
        samples[0].plural_verb_ids, dtype=torch.long,
    )

    return {
        "clean_ids": clean_ids,                    # (B, max_len)
        "corrupt_ids": corrupt_ids,                # (B, max_len)
        "subject_pos": subject_pos,                # (B,)
        "verb_pos": verb_pos,                      # (B,)
        "singular_verb_ids": singular_verb_ids,    # (n_verb_pairs,)
        "plural_verb_ids": plural_verb_ids,        # (n_verb_pairs,)
    }
