"""Automatic contrastive pair generators for scalable RCID.

Three strategies for constructing (clean, corrupt) pairs from raw text:
  1. EntitySwapGenerator  — replace named entities with same-category alternatives
  2. NumberPerturbGenerator — perturb numerical values
  3. LLMGenerator — use the teacher LLM itself to produce minimal edits

All generators share a common ABC and use the teacher model to verify that
the output actually changes (i.e. the perturbation is *causally relevant*).
"""

from __future__ import annotations

import logging
import random
import re
from abc import ABC, abstractmethod
from typing import Any

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)

# ── Entity pools (regex-matchable, no external NER needed) ────────────────

PERSON_NAMES: list[str] = [
    "Alice", "Bob", "Charlie", "Diana", "Edward", "Fiona", "George",
    "Hannah", "Ivan", "Julia", "Kevin", "Laura", "Michael", "Nancy",
    "Oliver", "Patricia", "Quentin", "Rachel", "Steven", "Teresa",
    "Ulysses", "Victoria", "William", "Xavier", "Yvonne", "Zachary",
    "Arthur", "Bridget", "Connor", "Deborah", "Ethan", "Grace",
]

COUNTRIES: list[str] = [
    "France", "Germany", "Japan", "China", "India", "Brazil", "Canada",
    "Australia", "Mexico", "Italy", "Spain", "Russia", "Korea", "Egypt",
    "Turkey", "Sweden", "Norway", "Poland", "Argentina", "Thailand",
    "Vietnam", "Kenya", "Nigeria", "Colombia", "Chile", "Peru",
    "Finland", "Denmark", "Ireland", "Portugal", "Austria", "Greece",
]

CITIES: list[str] = [
    "Paris", "Berlin", "Tokyo", "Beijing", "Mumbai", "London", "Moscow",
    "Sydney", "Cairo", "Rome", "Madrid", "Toronto", "Seoul", "Bangkok",
    "Lagos", "Istanbul", "Stockholm", "Warsaw", "Vienna", "Prague",
    "Dublin", "Lisbon", "Athens", "Helsinki", "Oslo", "Copenhagen",
    "Amsterdam", "Brussels", "Zurich", "Singapore", "Dubai", "Milan",
]

COMPANIES: list[str] = [
    "Google", "Apple", "Microsoft", "Amazon", "Meta", "Tesla", "Samsung",
    "Toyota", "Sony", "Intel", "Oracle", "Adobe", "Netflix", "Spotify",
    "Uber", "Airbnb", "Twitter", "Nvidia", "Qualcomm", "Cisco",
]

# Map pool name → (word list, compiled regex pattern for whole-word match)
_ENTITY_POOLS: dict[str, list[str]] = {
    "person": PERSON_NAMES,
    "country": COUNTRIES,
    "city": CITIES,
    "company": COMPANIES,
}


def _build_entity_regex(pool: list[str]) -> re.Pattern[str]:
    """Build a compiled regex that matches any entity in *pool* as a whole word."""
    escaped = [re.escape(w) for w in sorted(pool, key=len, reverse=True)]
    return re.compile(r"\b(" + "|".join(escaped) + r")\b")


_ENTITY_PATTERNS: dict[str, re.Pattern[str]] = {
    name: _build_entity_regex(words) for name, words in _ENTITY_POOLS.items()
}

# ── Helpers ───────────────────────────────────────────────────────────────


def _teacher_output_changed(
    teacher: nn.Module,
    tokenizer: Any,
    clean_text: str,
    corrupt_text: str,
    device: torch.device | str = "cpu",
) -> bool:
    """Return True if teacher's argmax prediction at the last position differs.

    Compares top-1 token at the final position for *clean_text* vs *corrupt_text*.
    """
    teacher.eval()
    with torch.no_grad():
        clean_ids = tokenizer(
            clean_text, return_tensors="pt", truncation=True, max_length=512,
        ).input_ids.to(device)  # (1, seq_len_c)
        corrupt_ids = tokenizer(
            corrupt_text, return_tensors="pt", truncation=True, max_length=512,
        ).input_ids.to(device)  # (1, seq_len_x)

        clean_pred = teacher(clean_ids).logits[:, -1, :].argmax(dim=-1)   # (1,)
        corrupt_pred = teacher(corrupt_ids).logits[:, -1, :].argmax(dim=-1)  # (1,)

    return clean_pred.item() != corrupt_pred.item()


# ── Abstract base class ──────────────────────────────────────────────────


class ContrastivePairGenerator(ABC):
    """Base class for contrastive pair generators."""

    def __init__(
        self,
        teacher: nn.Module,
        tokenizer: Any,
        device: torch.device | str = "cpu",
    ) -> None:
        self.teacher = teacher
        self.tokenizer = tokenizer
        self.device = device

    @abstractmethod
    def generate(self, text: str) -> list[tuple[str, str]]:
        """Return a list of (clean, corrupt) text pairs derived from *text*."""
        ...

    def validate_pair(self, clean: str, corrupt: str) -> bool:
        """Check that a contrastive pair is valid.

        Criteria:
          1. clean and corrupt differ (trivially).
          2. Teacher's argmax output changes between clean and corrupt.
        """
        if clean == corrupt:
            return False
        return _teacher_output_changed(
            self.teacher, self.tokenizer, clean, corrupt, self.device,
        )

    def batch_generate(
        self,
        texts: list[str],
        max_pairs_per_text: int = 3,
    ) -> list[tuple[str, str]]:
        """Generate contrastive pairs from a list of texts.

        Returns at most *max_pairs_per_text* validated pairs per input text.
        """
        all_pairs: list[tuple[str, str]] = []
        for text in texts:
            candidates = self.generate(text)
            # Validate and cap per text
            validated = [
                (c, x) for c, x in candidates if self.validate_pair(c, x)
            ]
            all_pairs.extend(validated[:max_pairs_per_text])
        logger.info(
            "batch_generate: %d texts → %d validated pairs",
            len(texts), len(all_pairs),
        )
        return all_pairs


# ── 1. Entity swap generator ─────────────────────────────────────────────


class EntitySwapGenerator(ContrastivePairGenerator):
    """Generate contrastive pairs by swapping named entities.

    Uses regex matching against pre-defined entity pools (persons, countries,
    cities, companies).  No external NER model required.

    Example::

        clean:   "The capital of France is"
        corrupt: "The capital of Germany is"
    """

    def __init__(
        self,
        teacher: nn.Module,
        tokenizer: Any,
        device: torch.device | str = "cpu",
        max_replacements_per_entity: int = 3,
        seed: int | None = None,
    ) -> None:
        super().__init__(teacher, tokenizer, device)
        self.max_replacements = max_replacements_per_entity
        self.rng = random.Random(seed)

    def generate(self, text: str) -> list[tuple[str, str]]:
        """Find entities in *text* and produce swap variants."""
        pairs: list[tuple[str, str]] = []

        for pool_name, pattern in _ENTITY_PATTERNS.items():
            pool = _ENTITY_POOLS[pool_name]
            for match in pattern.finditer(text):
                original = match.group()
                replacements = [e for e in pool if e != original]
                if not replacements:
                    continue
                sampled = self.rng.sample(
                    replacements, min(self.max_replacements, len(replacements)),
                )
                for replacement in sampled:
                    corrupt = text[: match.start()] + replacement + text[match.end() :]
                    pairs.append((text, corrupt))

        return pairs


# ── 2. Number perturbation generator ─────────────────────────────────────

_NUMBER_RE = re.compile(r"\b(\d+)\b")


class NumberPerturbGenerator(ContrastivePairGenerator):
    """Generate contrastive pairs by perturbing numbers in the text.

    Strategies: +1, -1, +3, -3, ×2, ÷2 (integer division).
    Suitable for math-reasoning tasks (GSM8K, MATH).

    Example::

        clean:   "If John has 5 apples and gives 2, he has"
        corrupt: "If John has 7 apples and gives 2, he has"
    """

    # (label, transform_fn) — kept as class-level for easy extension
    PERTURBATIONS: list[tuple[str, Any]] = [
        ("+1", lambda n: n + 1),
        ("-1", lambda n: max(n - 1, 0)),
        ("+3", lambda n: n + 3),
        ("-3", lambda n: max(n - 3, 0)),
        ("x2", lambda n: n * 2),
        ("/2", lambda n: n // 2),
    ]

    def __init__(
        self,
        teacher: nn.Module,
        tokenizer: Any,
        device: torch.device | str = "cpu",
        max_perturbations_per_number: int = 3,
        seed: int | None = None,
    ) -> None:
        super().__init__(teacher, tokenizer, device)
        self.max_perturbations = max_perturbations_per_number
        self.rng = random.Random(seed)

    def generate(self, text: str) -> list[tuple[str, str]]:
        """Find integers in *text* and produce perturbed variants."""
        pairs: list[tuple[str, str]] = []
        matches = list(_NUMBER_RE.finditer(text))
        if not matches:
            return pairs

        for match in matches:
            original_val = int(match.group())
            perturbations = list(self.PERTURBATIONS)
            self.rng.shuffle(perturbations)

            count = 0
            for _label, fn in perturbations:
                new_val = fn(original_val)
                if new_val == original_val:
                    continue
                corrupt = (
                    text[: match.start()] + str(new_val) + text[match.end() :]
                )
                pairs.append((text, corrupt))
                count += 1
                if count >= self.max_perturbations:
                    break

        return pairs


# ── 3. LLM-based generator ──────────────────────────────────────────────

_LLM_PROMPT_TEMPLATE = (
    "Below is a sentence. Generate a minimally modified version where "
    "changing only 1-2 key words leads to a different correct answer or "
    "conclusion.\n\n"
    "Original: {text}\n\n"
    "Requirements:\n"
    "1. Change only 1-2 words.\n"
    "2. The change must cause the answer/conclusion to differ.\n"
    "3. Return ONLY the modified sentence, nothing else.\n\n"
    "Modified:"
)


class LLMGenerator(ContrastivePairGenerator):
    """Generate contrastive pairs using the teacher LLM itself.

    The teacher is prompted to produce a minimally edited variant of the
    input text.  A post-hoc check ensures the edit is small enough and
    actually changes the teacher's own output.

    Use as a fallback when EntitySwap and NumberPerturb are not applicable.
    """

    def __init__(
        self,
        teacher: nn.Module,
        tokenizer: Any,
        device: torch.device | str = "cpu",
        max_new_tokens: int = 128,
        max_word_diff: int = 3,
        num_attempts: int = 3,
        seed: int | None = None,
    ) -> None:
        super().__init__(teacher, tokenizer, device)
        self.max_new_tokens = max_new_tokens
        self.max_word_diff = max_word_diff
        self.num_attempts = num_attempts
        self.rng = random.Random(seed)

    # ---- internal helpers ------------------------------------------------

    @staticmethod
    def _is_minimal_change(
        clean: str, corrupt: str, max_word_diff: int = 3,
    ) -> bool:
        """Return True if *corrupt* differs from *clean* by at most
        *max_word_diff* words and their lengths are similar."""
        clean_words = clean.split()
        corrupt_words = corrupt.split()
        if abs(len(clean_words) - len(corrupt_words)) > 1:
            return False
        # Count differing positions (zip truncates to shorter)
        diff_count = sum(1 for a, b in zip(clean_words, corrupt_words) if a != b)
        # Account for length difference
        diff_count += abs(len(clean_words) - len(corrupt_words))
        return diff_count <= max_word_diff

    def _generate_one(self, text: str) -> str | None:
        """Prompt the teacher to produce a single minimal variant."""
        prompt = _LLM_PROMPT_TEMPLATE.format(text=text)
        input_ids = self.tokenizer(
            prompt, return_tensors="pt", truncation=True, max_length=512,
        ).input_ids.to(self.device)  # (1, prompt_len)

        self.teacher.eval()
        with torch.no_grad():
            output_ids = self.teacher.generate(
                input_ids,
                max_new_tokens=self.max_new_tokens,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
            )  # (1, prompt_len + gen_len)

        # Decode only generated portion
        gen_ids = output_ids[0, input_ids.shape[1] :]  # (gen_len,)
        generated = self.tokenizer.decode(gen_ids, skip_special_tokens=True).strip()

        # Strip any trailing explanation the model might have added
        # Take only the first line/sentence
        generated = generated.split("\n")[0].strip()
        return generated if generated else None

    # ---- public API ------------------------------------------------------

    def generate(self, text: str) -> list[tuple[str, str]]:
        """Prompt teacher up to *num_attempts* times for minimal edits."""
        pairs: list[tuple[str, str]] = []
        for _ in range(self.num_attempts):
            corrupt = self._generate_one(text)
            if corrupt is None:
                continue
            if not self._is_minimal_change(text, corrupt, self.max_word_diff):
                continue
            pairs.append((text, corrupt))
        return pairs
