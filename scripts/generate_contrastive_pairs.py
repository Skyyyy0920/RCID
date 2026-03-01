"""Generate contrastive pairs from Dolly-15K for RCID training.

Reads prompt-only texts from the Dolly train split (via
:func:`rcid.data.dolly_utils.get_dolly_texts_for_contrastive`) and runs
EntitySwapGenerator / NumberPerturbGenerator to produce validated
(clean, corrupt) pairs.  Results are saved as per-task JSON files in
``--output_dir`` for later consumption by GeneratedContrastiveDataset.

Usage::

    python scripts/generate_contrastive_pairs.py \
        --model_name Qwen/Qwen3-8B \
        --output_dir data/contrastive_pairs/ \
        --device cuda:0

    # Limit source texts for quick testing
    python scripts/generate_contrastive_pairs.py \
        --model_name Qwen/Qwen3-8B \
        --max_source_texts 500 \
        --device cuda:0
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from rcid.data.dolly_utils import get_dolly_texts_for_contrastive

logger = logging.getLogger(__name__)


def _save_pairs(
    pairs: list[tuple[str, str]],
    filepath: Path,
    task_type: str,
) -> None:
    """Save (clean, corrupt) pairs to a JSON file."""
    records = [
        {"clean": c, "corrupt": x, "task_type": task_type}
        for c, x in pairs
    ]
    filepath.parent.mkdir(parents=True, exist_ok=True)
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(records, f, indent=2, ensure_ascii=False)
    logger.info("Saved %d pairs to %s", len(records), filepath)


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Generate contrastive pairs from Dolly-15K",
    )
    ap.add_argument(
        "--model_name", default="Qwen/Qwen3-8B",
        help="Teacher model for output-change validation",
    )
    ap.add_argument(
        "--output_dir", default="data/contrastive_pairs/",
        help="Directory for per-task JSON output files",
    )
    ap.add_argument(
        "--max_source_texts", type=int, default=None,
        help="Limit number of Dolly prompts to process (default: all ~14k)",
    )
    ap.add_argument(
        "--max_entity_pairs", type=int, default=2500,
        help="Max validated entity-swap pairs to keep",
    )
    ap.add_argument(
        "--max_number_pairs", type=int, default=2500,
        help="Max validated number-perturb pairs to keep",
    )
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from rcid.data.contrastive_generators import (
        EntitySwapGenerator,
        NumberPerturbGenerator,
    )

    # ── Load Dolly prompts ─────────────────────────────────────────
    logger.info("Loading Dolly prompts for contrastive pair generation...")
    texts = get_dolly_texts_for_contrastive(seed=args.seed)
    if args.max_source_texts is not None:
        texts = texts[: args.max_source_texts]
    logger.info("Loaded %d Dolly prompts", len(texts))

    # ── Load teacher ───────────────────────────────────────────────
    logger.info("Loading teacher: %s", args.model_name)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    teacher = AutoModelForCausalLM.from_pretrained(
        args.model_name, torch_dtype=torch.float16,
    ).to(args.device).eval()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── Entity swap ────────────────────────────────────────────────
    logger.info("Running EntitySwapGenerator...")
    t0 = time.time()
    entity_gen = EntitySwapGenerator(
        teacher, tokenizer, device=args.device, seed=args.seed,
    )
    entity_pairs = entity_gen.batch_generate(texts, max_pairs_per_text=3)
    if len(entity_pairs) > args.max_entity_pairs:
        entity_pairs = entity_pairs[: args.max_entity_pairs]
    logger.info(
        "Entity swap: %d pairs in %.1fs",
        len(entity_pairs), time.time() - t0,
    )
    _save_pairs(entity_pairs, out_dir / "entity_swap.json", "entity_swap")

    # ── Number perturbation ────────────────────────────────────────
    logger.info("Running NumberPerturbGenerator...")
    t0 = time.time()
    number_gen = NumberPerturbGenerator(
        teacher, tokenizer, device=args.device, seed=args.seed,
    )
    number_pairs = number_gen.batch_generate(texts, max_pairs_per_text=3)
    if len(number_pairs) > args.max_number_pairs:
        number_pairs = number_pairs[: args.max_number_pairs]
    logger.info(
        "Number perturb: %d pairs in %.1fs",
        len(number_pairs), time.time() - t0,
    )
    _save_pairs(number_pairs, out_dir / "number_perturb.json", "number_perturb")

    # ── Summary ────────────────────────────────────────────────────
    total = len(entity_pairs) + len(number_pairs)
    summary: dict[str, Any] = {
        "data_source": "databricks/databricks-dolly-15k",
        "split": "train",
        "seed": args.seed,
        "n_source_texts": len(texts),
        "entity_swap_pairs": len(entity_pairs),
        "number_perturb_pairs": len(number_pairs),
        "total_pairs": total,
        "model": args.model_name,
    }
    with open(out_dir / "generation_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    logger.info("Done. Total: %d contrastive pairs saved to %s", total, out_dir)


if __name__ == "__main__":
    main()
