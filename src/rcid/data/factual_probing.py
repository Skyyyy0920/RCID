"""Factual Knowledge Probing dataset for contrastive analysis."""

from __future__ import annotations

import random
from typing import TYPE_CHECKING

import torch

from rcid.circuit.contrastive import ContrastiveDataset

if TYPE_CHECKING:
    from transformers import PreTrainedTokenizer

# (subject_clean, answer_clean, subject_corrupt, answer_corrupt)
FACT_PAIRS: list[tuple[str, str, str, str]] = [
    ("France", "Paris", "Germany", "Berlin"),
    ("Japan", "Tokyo", "China", "Beijing"),
    ("Italy", "Rome", "Spain", "Madrid"),
    ("Canada", "Ottawa", "Australia", "Canberra"),
    ("Egypt", "Cairo", "Turkey", "Ankara"),
    ("Brazil", "Brasilia", "Argentina", "Buenos"),
    ("India", "Delhi", "Pakistan", "Islamabad"),
    ("Russia", "Moscow", "Poland", "Warsaw"),
    ("Mexico", "Mexico", "Peru", "Lima"),
    ("Sweden", "Stockholm", "Norway", "Oslo"),
    ("Greece", "Athens", "Denmark", "Copenhagen"),
    ("Thailand", "Bangkok", "Vietnam", "Hanoi"),
    ("Portugal", "Lisbon", "Ireland", "Dublin"),
    ("Austria", "Vienna", "Belgium", "Brussels"),
    ("Finland", "Helsinki", "Hungary", "Budapest"),
    ("Cuba", "Havana", "Chile", "Santiago"),
    ("Kenya", "Nairobi", "Ghana", "Accra"),
    ("Ukraine", "Kyiv", "Romania", "Bucharest"),
    ("Colombia", "Bogota", "Venezuela", "Caracas"),
    ("Netherlands", "Amsterdam", "Switzerland", "Bern"),
]

FACTUAL_TEMPLATES: list[str] = [
    "The capital of {country} is",
    "The capital city of {country} is",
    "{country} has its capital in",
]

# OOD templates with different phrasing
FACTUAL_OOD_TEMPLATES: list[str] = [
    "If you visit {country}, you will find the capital is",
    "Among all cities in {country}, the capital is",
    "The government of {country} is headquartered in",
]


def _verify_single_token(
    tokenizer: PreTrainedTokenizer,
    text: str,
) -> int | None:
    """Return token id if text encodes to a single token, else None."""
    ids = tokenizer.encode(" " + text, add_special_tokens=False)
    if len(ids) == 1:
        return ids[0]
    return None


class FactualProbingDataset:
    """Factual knowledge probing with contrastive country pairs.

    Clean:   "The capital of France is"  → Paris
    Corrupt: "The capital of Germany is" → Berlin
    """

    def __init__(
        self,
        tokenizer: PreTrainedTokenizer,
        fact_pairs: list[tuple[str, str, str, str]] | None = None,
        templates: list[str] | None = None,
        seed: int = 42,
    ) -> None:
        self.tokenizer = tokenizer
        self.templates = templates or FACTUAL_TEMPLATES
        pairs = fact_pairs or FACT_PAIRS
        rng = random.Random(seed)

        # Filter pairs where both answers are single-token
        valid_pairs = []
        for sc, ac, sx, ax in pairs:
            cid = _verify_single_token(tokenizer, ac)
            xid = _verify_single_token(tokenizer, ax)
            if cid is not None and xid is not None:
                valid_pairs.append((sc, ac, cid, sx, ax, xid))

        assert len(valid_pairs) >= 1, "No valid fact pairs for this tokenizer"
        self.valid_pairs = valid_pairs
        rng.shuffle(self.valid_pairs)
        self.dataset = self._build(tokenizer)

    def _build(self, tokenizer: PreTrainedTokenizer) -> ContrastiveDataset:
        pad_id = tokenizer.pad_token_id or 0
        raw: list[dict] = []
        global_max = 0

        for pair in self.valid_pairs:
            sc, ac, cid, sx, ax, xid = pair
            for t_idx, template in enumerate(self.templates):
                clean_text = template.format(country=sc)
                corrupt_text = template.format(country=sx)

                c_ids = tokenizer.encode(clean_text, add_special_tokens=False)
                x_ids = tokenizer.encode(corrupt_text, add_special_tokens=False)
                pair_len = max(len(c_ids), len(x_ids))
                c_ids = c_ids + [pad_id] * (pair_len - len(c_ids))
                x_ids = x_ids + [pad_id] * (pair_len - len(x_ids))

                entity_pos = -1
                for i in range(pair_len):
                    if c_ids[i] != x_ids[i]:
                        entity_pos = i
                        break
                if entity_pos < 0:
                    continue

                raw.append({
                    "c": c_ids, "x": x_ids,
                    "correct": cid, "wrong": xid,
                    "entity_pos": entity_pos,
                    "answer_pos": pair_len - 1,
                })
                global_max = max(global_max, pair_len)

        assert len(raw) > 0, "No valid samples generated"

        clean_list = [r["c"] + [pad_id] * (global_max - len(r["c"])) for r in raw]
        corrupt_list = [r["x"] + [pad_id] * (global_max - len(r["x"])) for r in raw]

        return ContrastiveDataset(
            clean_ids=torch.tensor(clean_list, dtype=torch.long),
            corrupt_ids=torch.tensor(corrupt_list, dtype=torch.long),
            answer_pos=torch.tensor([r["answer_pos"] for r in raw], dtype=torch.long),
            correct_token_id=torch.tensor([r["correct"] for r in raw], dtype=torch.long),
            wrong_token_id=torch.tensor([r["wrong"] for r in raw], dtype=torch.long),
            key_positions={
                "entity": torch.tensor([r["entity_pos"] for r in raw], dtype=torch.long),
            },
            is_modified={"entity": True},
            model_family=tokenizer.name_or_path.lower(),
        )

    @classmethod
    def build_ood_dataset(
        cls,
        tokenizer: PreTrainedTokenizer,
        seed: int = 999,
    ) -> "FactualProbingDataset":
        """Build OOD variant with different templates."""
        return cls(
            tokenizer=tokenizer,
            templates=FACTUAL_OOD_TEMPLATES,
            seed=seed,
        )
