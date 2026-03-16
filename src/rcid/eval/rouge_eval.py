"""ROUGE-L evaluation for RCID generation quality.

Generates responses from a model and computes ROUGE-L scores against
ground-truth references.  Used to evaluate distilled students on the
Dolly-15K test split.

Handles Qwen3 thinking-mode output (``<think>...</think>`` blocks) by
stripping them before scoring.

Usage::

    from rcid.eval.rouge_eval import evaluate_rouge, save_generations
    from rcid.data.dolly_utils import get_dolly_prompts

    prompts, refs = get_dolly_prompts(split="test")
    results = evaluate_rouge(model, tokenizer, prompts, refs)
    print(f"ROUGE-L: {results['rouge_l_f']:.4f}")
"""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import TYPE_CHECKING, Any

import torch
from tqdm import tqdm

if TYPE_CHECKING:
    from transformers import PreTrainedModel, PreTrainedTokenizer

logger = logging.getLogger(__name__)

# Regex to strip Qwen3 thinking blocks
_THINK_CLOSED = re.compile(r"<think>.*?</think>\s*", flags=re.DOTALL)
_THINK_UNCLOSED = re.compile(r"<think>.*$", flags=re.DOTALL)


def _strip_thinking(text: str) -> str:
    """Remove ``<think>...</think>`` blocks and unclosed thinking tails."""
    text = _THINK_CLOSED.sub("", text)
    text = _THINK_UNCLOSED.sub("", text)
    return text.strip()


@torch.no_grad()
def evaluate_rouge(
    model: Any,
    tokenizer: Any,
    eval_prompts: list[str],
    references: list[str],
    max_new_tokens: int = 256,
    batch_size: int = 8,
) -> dict[str, Any]:
    """Generate responses and compute ROUGE-L scores.

    Parameters
    ----------
    model : PreTrainedModel
        Model to evaluate (will be set to eval mode).
    tokenizer : PreTrainedTokenizer
        Tokenizer for the model.
    eval_prompts : list[str]
        Input prompts (e.g. formatted Dolly prompts).
    references : list[str]
        Ground truth responses.
    max_new_tokens : int
        Maximum tokens to generate per prompt.
    batch_size : int
        Generation batch size.

    Returns
    -------
    dict
        Keys: ``rouge_l_f``, ``rouge_l_p``, ``rouge_l_r`` (mean values),
        ``num_samples``, ``generations`` list, ``per_sample_rouge_l_f``.
    """
    from rouge_score import rouge_scorer

    model.eval()
    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)

    # Ensure pad token exists
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    generations: list[str] = []
    old_padding_side = tokenizer.padding_side
    tokenizer.padding_side = "left"

    for i in tqdm(range(0, len(eval_prompts), batch_size), desc="Evaluating"):
        batch_prompts = eval_prompts[i : i + batch_size]
        inputs = tokenizer(
            batch_prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512,
        ).to(model.device)

        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,  # Greedy for reproducibility
            pad_token_id=tokenizer.pad_token_id,
        )

        prompt_len = inputs["input_ids"].shape[1]
        for output in outputs:
            generated_ids = output[prompt_len:]

            # Decode raw first (keep special tokens to handle <think>)
            raw_text = tokenizer.decode(generated_ids, skip_special_tokens=False)

            # Strip Qwen3 thinking blocks, then clean special tokens
            text = _strip_thinking(raw_text)
            for tok in tokenizer.all_special_tokens:
                text = text.replace(tok, "")
            text = text.strip()

            # Fallback: if stripping produced nothing, try plain decode
            if not text:
                text = tokenizer.decode(
                    generated_ids, skip_special_tokens=True,
                ).strip()

            generations.append(text)

    tokenizer.padding_side = old_padding_side

    # Diagnostic: count empty generations
    n_empty = sum(1 for g in generations if not g)
    if n_empty > 0:
        logger.warning(
            "Empty generations: %d / %d (%.1f%%). "
            "Possible cause: model uses thinking mode or generates EOS "
            "immediately. Consider increasing max_new_tokens.",
            n_empty, len(generations), 100 * n_empty / max(len(generations), 1),
        )

    # Compute ROUGE-L scores
    scores_f: list[float] = []
    scores_p: list[float] = []
    scores_r: list[float] = []
    for gen, ref in zip(generations, references):
        score = scorer.score(ref, gen)
        scores_f.append(score["rougeL"].fmeasure)
        scores_p.append(score["rougeL"].precision)
        scores_r.append(score["rougeL"].recall)

    return {
        "rouge_l_f": sum(scores_f) / max(len(scores_f), 1),
        "rouge_l_p": sum(scores_p) / max(len(scores_p), 1),
        "rouge_l_r": sum(scores_r) / max(len(scores_r), 1),
        "num_samples": len(eval_prompts),
        "num_empty_generations": n_empty,
        "generations": generations,
        "per_sample_rouge_l_f": scores_f,
    }


def save_generations(
    prompts: list[str],
    generations: list[str],
    references: list[str],
    filepath: str,
    rouge_scores: list[float] | None = None,
) -> None:
    """Save generation results to JSON for human inspection.

    Parameters
    ----------
    prompts : list[str]
        Input prompts.
    generations : list[str]
        Model-generated responses.
    references : list[str]
        Ground truth responses.
    filepath : str
        Output JSON file path.
    rouge_scores : list[float] | None
        Per-sample ROUGE-L F1 scores (optional).
    """
    records = []
    for i, (p, g, r) in enumerate(zip(prompts, generations, references)):
        record: dict[str, Any] = {
            "prompt": p,
            "generation": g,
            "reference": r,
        }
        if rouge_scores:
            record["rouge_l_f"] = rouge_scores[i]
        records.append(record)

    Path(filepath).parent.mkdir(parents=True, exist_ok=True)
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(records, f, indent=2, ensure_ascii=False)
    logger.info("Saved %d generations to %s", len(records), filepath)
