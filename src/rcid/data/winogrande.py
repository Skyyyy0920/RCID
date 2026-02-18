"""WinoGrande dataset for contrastive analysis of common-sense reasoning."""

from __future__ import annotations

import random
from typing import TYPE_CHECKING

import torch

from rcid.circuit.contrastive import ContrastiveDataset

if TYPE_CHECKING:
    from transformers import PreTrainedTokenizer

# Hand-crafted contrastive pairs where swapping one word changes the referent.
# Format: (clean_text, corrupt_text, clean_answer, corrupt_answer)
# The answer is the word that "it" / pronoun refers to.
WINOGRANDE_PAIRS: list[tuple[str, str, str, str]] = [
    (
        "The trophy doesn't fit in the suitcase because it is too big.",
        "The trophy doesn't fit in the suitcase because it is too small.",
        "trophy",
        "suitcase",
    ),
    (
        "The bottle fell off the table because it was unsteady.",
        "The bottle fell off the table because it was heavy.",
        "table",
        "bottle",
    ),
    (
        "The man couldn't lift his son because he was too weak.",
        "The man couldn't lift his son because he was too heavy.",
        "man",
        "son",
    ),
    (
        "The painting was moved from the wall to the shelf because it was too bare.",
        "The painting was moved from the wall to the shelf because it was too crowded.",
        "wall",
        "shelf",
    ),
    (
        "The car beat the bus in the race because it was fast.",
        "The car beat the bus in the race because it was slow.",
        "car",
        "bus",
    ),
    (
        "The teacher praised the student because she was proud.",
        "The teacher praised the student because she was talented.",
        "teacher",
        "student",
    ),
    (
        "The cat chased the mouse because it was playful.",
        "The cat chased the mouse because it was scared.",
        "cat",
        "mouse",
    ),
    (
        "The glass broke when it fell on the rock because it was fragile.",
        "The glass broke when it fell on the rock because it was sharp.",
        "glass",
        "rock",
    ),
    (
        "The lamp outshone the candle because it was bright.",
        "The lamp outshone the candle because it was dim.",
        "lamp",
        "candle",
    ),
    (
        "The rope was tied to the pole because it was loose.",
        "The rope was tied to the pole because it was sturdy.",
        "rope",
        "pole",
    ),
    (
        "The ball rolled from the hill to the valley because it was steep.",
        "The ball rolled from the hill to the valley because it was flat.",
        "hill",
        "valley",
    ),
    (
        "The jacket didn't fit the child because it was oversized.",
        "The jacket didn't fit the child because it was tiny.",
        "jacket",
        "child",
    ),
    (
        "The train passed the bicycle because it was powerful.",
        "The train passed the bicycle because it was weak.",
        "train",
        "bicycle",
    ),
    (
        "The book fell off the desk because it was tilted.",
        "The book fell off the desk because it was heavy.",
        "desk",
        "book",
    ),
    (
        "The ring was put in the box because it was valuable.",
        "The ring was put in the box because it was empty.",
        "ring",
        "box",
    ),
]


def _find_first_diff_pos(
    ids_a: list[int], ids_b: list[int]
) -> int:
    """Find the first position where two token sequences differ."""
    for i in range(min(len(ids_a), len(ids_b))):
        if ids_a[i] != ids_b[i]:
            return i
    return -1


def _find_token_pos(
    token_ids: list[int],
    target_ids: list[int],
    start: int = 0,
) -> int:
    """Find position of target token sequence in token_ids."""
    for i in range(start, len(token_ids) - len(target_ids) + 1):
        if token_ids[i : i + len(target_ids)] == target_ids:
            return i
    return -1


class WinoGrandeDataset:
    """WinoGrande contrastive dataset for common-sense reasoning.

    Clean:   "The trophy doesn't fit ... because it is too big." → trophy
    Corrupt: "The trophy doesn't fit ... because it is too small." → suitcase
    """

    def __init__(
        self,
        tokenizer: PreTrainedTokenizer,
        pairs: list[tuple[str, str, str, str]] | None = None,
        seed: int = 42,
    ) -> None:
        self.tokenizer = tokenizer
        pair_list = pairs or WINOGRANDE_PAIRS
        rng = random.Random(seed)
        rng.shuffle(pair_list)

        self.dataset = self._build(pair_list, tokenizer)

    def _build(
        self,
        pairs: list[tuple[str, str, str, str]],
        tokenizer: PreTrainedTokenizer,
    ) -> ContrastiveDataset:
        pad_id = tokenizer.pad_token_id or 0
        raw: list[dict] = []
        global_max = 0

        for clean_text, corrupt_text, clean_ans, corrupt_ans in pairs:
            c_ids = tokenizer.encode(clean_text, add_special_tokens=False)
            x_ids = tokenizer.encode(corrupt_text, add_special_tokens=False)
            pair_len = max(len(c_ids), len(x_ids))
            c_ids = c_ids + [pad_id] * (pair_len - len(c_ids))
            x_ids = x_ids + [pad_id] * (pair_len - len(x_ids))

            correct_tok = tokenizer.encode(" " + clean_ans, add_special_tokens=False)
            wrong_tok = tokenizer.encode(" " + corrupt_ans, add_special_tokens=False)
            if len(correct_tok) != 1 or len(wrong_tok) != 1:
                continue

            mod_pos = _find_first_diff_pos(c_ids, x_ids)
            if mod_pos < 0:
                continue

            it_tok = tokenizer.encode(" it", add_special_tokens=False)
            pron_pos = _find_token_pos(c_ids, it_tok)
            if pron_pos < 0:
                for pron in [" she", " he"]:
                    pron_tok = tokenizer.encode(pron, add_special_tokens=False)
                    pron_pos = _find_token_pos(c_ids, pron_tok)
                    if pron_pos >= 0:
                        break
            if pron_pos < 0:
                continue

            raw.append({
                "c": c_ids, "x": x_ids,
                "correct": correct_tok[0], "wrong": wrong_tok[0],
                "mod_pos": mod_pos, "pron_pos": pron_pos,
                "answer_pos": pair_len - 1,
            })
            global_max = max(global_max, pair_len)

        assert len(raw) > 0, "No valid WinoGrande samples"

        clean_list = [r["c"] + [pad_id] * (global_max - len(r["c"])) for r in raw]
        corrupt_list = [r["x"] + [pad_id] * (global_max - len(r["x"])) for r in raw]

        return ContrastiveDataset(
            clean_ids=torch.tensor(clean_list, dtype=torch.long),
            corrupt_ids=torch.tensor(corrupt_list, dtype=torch.long),
            answer_pos=torch.tensor([r["answer_pos"] for r in raw], dtype=torch.long),
            correct_token_id=torch.tensor([r["correct"] for r in raw], dtype=torch.long),
            wrong_token_id=torch.tensor([r["wrong"] for r in raw], dtype=torch.long),
            key_positions={
                "modified": torch.tensor([r["mod_pos"] for r in raw], dtype=torch.long),
                "pronoun": torch.tensor([r["pron_pos"] for r in raw], dtype=torch.long),
            },
            is_modified={"modified": True, "pronoun": False},
            model_family=tokenizer.name_or_path.lower(),
        )

    @classmethod
    def build_ood_dataset(
        cls,
        tokenizer: PreTrainedTokenizer,
        seed: int = 999,
    ) -> "WinoGrandeDataset":
        """Build OOD variant with longer/different pairs."""
        ood_pairs: list[tuple[str, str, str, str]] = [
            (
                "During the competition, the athlete outperformed the amateur "
                "because it was clear that he was experienced.",
                "During the competition, the athlete outperformed the amateur "
                "because it was clear that he was inexperienced.",
                "athlete",
                "amateur",
            ),
            (
                "The large truck could not fit under the bridge because it was enormous.",
                "The large truck could not fit under the bridge because it was narrow.",
                "truck",
                "bridge",
            ),
        ]
        return cls(tokenizer=tokenizer, pairs=ood_pairs, seed=seed)
