"""IOI (Indirect Object Identification) dataset for contrastive analysis."""

from __future__ import annotations

import random
from typing import TYPE_CHECKING

import torch

from rcid.circuit.contrastive import ContrastiveDataset

if TYPE_CHECKING:
    from transformers import PreTrainedTokenizer

# ~100 candidate names; filtered to single-token per tokenizer at runtime
CANDIDATE_NAMES: list[str] = [
    "Mary", "John", "Alice", "Bob", "Tom", "Sara", "James", "Emma",
    "David", "Lisa", "Mark", "Anna", "Paul", "Jane", "Mike", "Kate",
    "Chris", "Amy", "Dan", "Laura", "Jack", "Julia", "Adam", "Ruth",
    "Peter", "Helen", "Sam", "Grace", "Ian", "Diana", "Luke", "Rose",
    "Eric", "Lily", "Max", "Ella", "Ryan", "Eva", "Ben", "Iris",
    "Carl", "Mia", "Fred", "Joy", "Glen", "Fay", "Neil", "Sue",
    "Tim", "Ivy", "Ray", "May", "Will", "Claire", "Ross", "Dawn",
    "Phil", "Kim", "Dean", "Beth", "Troy", "Jade", "Cole", "Fern",
    "Sean", "Hope", "Clay", "Jean", "Hank", "Gail", "Dale", "Joan",
    "Lee", "Liam", "Owen", "Hugo", "Noel", "Bea", "Ava", "Leo",
    "Sky", "Finn", "Zoe", "Roy", "Pat", "Ken", "Don", "Hal",
    "Drew", "Kurt", "Seth", "Tina", "Vera", "Gwen", "Nora", "Meg",
    "Greg", "Chad", "Jeff", "Walt",
]

IOI_TEMPLATES: list[str] = [
    "When {IO} and {S} went to the store, {S2} gave a drink to",
    "When {IO} and {S} went to the park, {S2} gave a ball to",
    "After {IO} and {S} arrived at the party, {S2} handed a gift to",
    "Then, {IO} and {S} had a meeting. {S2} presented a report to",
    "When {IO} and {S} entered the room, {S2} passed the book to",
    "While {IO} and {S} were at school, {S2} lent a pencil to",
    "After {IO} and {S} met for lunch, {S2} offered a seat to",
    "When {IO} and {S} visited the museum, {S2} showed a painting to",
    "Then, {IO} and {S} went shopping. {S2} bought a present for",
    "When {IO} and {S} arrived at work, {S2} sent a message to",
    "While {IO} and {S} were cooking, {S2} served dinner to",
    "After {IO} and {S} finished class, {S2} returned a notebook to",
    "When {IO} and {S} sat at the table, {S2} poured some water for",
    "Then, {IO} and {S} played a game. {S2} threw the ball to",
    "When {IO} and {S} were at the cafe, {S2} ordered coffee for",
]

# OOD templates for robustness evaluation
IOI_OOD_TEMPLATES: list[str] = [
    "Yesterday, {IO} and {S} were at the office when {S2} brought a letter to",
    "Last week, {IO} and {S} went hiking. During the trip, {S2} gave water to",
    "In the evening, {IO} and {S} watched a movie. Afterwards, {S2} recommended a film to",
    "On Monday, {IO} and {S} attended a concert. {S2} saved a seat for",
    "During the holiday, {IO} and {S} stayed home. {S2} cooked a meal for",
]


def build_single_token_names(tokenizer: PreTrainedTokenizer) -> list[str]:
    """Filter candidate names to those that tokenize as a single token.

    Checks " Name" (with leading space) since names appear mid-sentence.
    """
    valid: list[str] = []
    for name in CANDIDATE_NAMES:
        ids = tokenizer.encode(" " + name, add_special_tokens=False)
        if len(ids) == 1:
            valid.append(name)
    return valid


def _find_token_position(
    token_ids: list[int],
    target_ids: list[int],
    start: int = 0,
) -> int:
    """Find the position of target token sequence in token_ids."""
    for i in range(start, len(token_ids) - len(target_ids) + 1):
        if token_ids[i : i + len(target_ids)] == target_ids:
            return i
    return -1


class IOIDataset:
    """Indirect Object Identification contrastive dataset.

    Clean:   "When Mary and John went to the store, John gave a drink to" → Mary
    Corrupt: "When Mary and John went to the store, Mary gave a drink to" → ???
    """

    def __init__(
        self,
        tokenizer: PreTrainedTokenizer,
        n_samples: int = 100,
        templates: list[str] | None = None,
        name_pool: list[str] | None = None,
        seed: int = 42,
    ) -> None:
        self.tokenizer = tokenizer
        self.templates = templates or IOI_TEMPLATES
        rng = random.Random(seed)

        if name_pool is None:
            name_pool = build_single_token_names(tokenizer)
        assert len(name_pool) >= 2, f"Need >=2 names, got {len(name_pool)}"
        self.name_pool = name_pool

        samples = self._generate_samples(n_samples, rng)
        self.dataset = self._build_contrastive_dataset(samples, tokenizer)

    def _generate_samples(
        self, n_samples: int, rng: random.Random
    ) -> list[dict]:
        samples = []
        for i in range(n_samples):
            template = self.templates[i % len(self.templates)]
            io_name, s_name = rng.sample(self.name_pool, 2)
            clean_text = template.format(IO=io_name, S=s_name, S2=s_name)
            corrupt_text = template.format(IO=io_name, S=s_name, S2=io_name)
            samples.append({
                "clean_text": clean_text,
                "corrupt_text": corrupt_text,
                "io_name": io_name,
                "s_name": s_name,
                "template_index": i % len(self.templates),
            })
        return samples

    def _build_contrastive_dataset(
        self,
        samples: list[dict],
        tokenizer: PreTrainedTokenizer,
    ) -> ContrastiveDataset:
        # Pass 1: tokenize and find positions (unpadded)
        raw: list[dict] = []
        pad_id = tokenizer.pad_token_id or 0
        global_max = 0

        for sample in samples:
            c_ids = tokenizer.encode(sample["clean_text"], add_special_tokens=False)
            x_ids = tokenizer.encode(sample["corrupt_text"], add_special_tokens=False)
            pair_len = max(len(c_ids), len(x_ids))
            c_ids = c_ids + [pad_id] * (pair_len - len(c_ids))
            x_ids = x_ids + [pad_id] * (pair_len - len(x_ids))

            io_token = tokenizer.encode(
                " " + sample["io_name"], add_special_tokens=False
            )
            s_token = tokenizer.encode(
                " " + sample["s_name"], add_special_tokens=False
            )
            assert len(io_token) == 1, f"IO not single token: {sample['io_name']}"
            assert len(s_token) == 1, f"S not single token: {sample['s_name']}"

            io_pos = _find_token_position(c_ids, io_token)
            assert io_pos >= 0, f"IO token not found: {sample['io_name']}"
            s_first = _find_token_position(c_ids, s_token)
            s2_pos = _find_token_position(c_ids, s_token, start=s_first + 1)
            assert s2_pos >= 0, "S2 token not found in clean"

            raw.append({
                "c": c_ids, "x": x_ids,
                "correct": io_token[0], "wrong": s_token[0],
                "io_pos": io_pos, "s2_pos": s2_pos,
                "end_pos": pair_len - 1,
            })
            global_max = max(global_max, pair_len)

        # Pass 2: pad to global max and collect
        clean_ids_list = [r["c"] + [pad_id] * (global_max - len(r["c"])) for r in raw]
        corrupt_ids_list = [r["x"] + [pad_id] * (global_max - len(r["x"])) for r in raw]

        return ContrastiveDataset(
            clean_ids=torch.tensor(clean_ids_list, dtype=torch.long),
            corrupt_ids=torch.tensor(corrupt_ids_list, dtype=torch.long),
            answer_pos=torch.tensor([r["end_pos"] for r in raw], dtype=torch.long),
            correct_token_id=torch.tensor([r["correct"] for r in raw], dtype=torch.long),
            wrong_token_id=torch.tensor([r["wrong"] for r in raw], dtype=torch.long),
            key_positions={
                "io": torch.tensor([r["io_pos"] for r in raw], dtype=torch.long),
                "s2": torch.tensor([r["s2_pos"] for r in raw], dtype=torch.long),
                "end": torch.tensor([r["end_pos"] for r in raw], dtype=torch.long),
            },
            is_modified={"io": False, "s2": True, "end": False},
            model_family=tokenizer.name_or_path.lower(),
        )

    @classmethod
    def build_ood_dataset(
        cls,
        tokenizer: PreTrainedTokenizer,
        name_pool: list[str],
        in_distribution_names: list[str],
        n_samples: int = 50,
        seed: int = 999,
    ) -> "IOIDataset":
        """Build OOD variant with unseen names and different templates."""
        ood_names = [n for n in name_pool if n not in in_distribution_names]
        if len(ood_names) < 2:
            ood_names = name_pool  # fallback
        return cls(
            tokenizer=tokenizer,
            n_samples=n_samples,
            templates=IOI_OOD_TEMPLATES,
            name_pool=ood_names,
            seed=seed,
        )

    def print_samples(self, n: int = 3) -> None:
        """Print n samples with token-level detail for verification."""
        for i in range(min(n, len(self.dataset))):
            sample = self.dataset[i]
            clean = self.tokenizer.decode(sample["clean_ids"].tolist())
            corrupt = self.tokenizer.decode(sample["corrupt_ids"].tolist())
            print(f"--- Sample {i} ---")
            print(f"  Clean:   {clean}")
            print(f"  Corrupt: {corrupt}")
            print(f"  Answer pos: {sample['answer_pos'].item()}")
            correct = self.tokenizer.decode([sample["correct_token_id"].item()])
            wrong = self.tokenizer.decode([sample["wrong_token_id"].item()])
            print(f"  Correct: {correct!r}, Wrong: {wrong!r}")
