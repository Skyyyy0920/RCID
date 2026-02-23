"""Generate contrastive pairs from instruction data for scalable RCID.

Loads a teacher model, applies entity-swap / number-perturb / LLM generators
to instruction texts, validates pairs, and saves results as JSON.

Usage:
    python scripts/generate_contrastive_pairs.py \
        --model_name Qwen/Qwen3-8B --data_source tatsu-lab/alpaca \
        --max_samples 10000 --max_pairs 5000 --device cuda:0

    python scripts/generate_contrastive_pairs.py \
        --model_name meta-llama/Llama-3.1-8B \
        --generators entity_swap,number_perturb,llm \
        --max_pairs 3000
"""
from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path
from typing import Any

import torch
from tqdm import tqdm

logger = logging.getLogger(__name__)

# Map CLI name → generator class
GENERATOR_NAMES = ("entity_swap", "number_perturb", "llm")


def _extract_text(sample: dict[str, Any]) -> str | None:
    """Extract usable text from an instruction-data sample.

    Supports Alpaca format (instruction / input / output) and generic
    text / question / prompt fields.
    """
    # Alpaca: concatenate instruction + input (if non-empty) + output
    if "instruction" in sample:
        parts = [sample["instruction"]]
        inp = sample.get("input", "")
        if inp:
            parts.append(inp)
        out = sample.get("output", "")
        if out:
            parts.append(out)
        return " ".join(parts)

    # Generic fields
    for key in ("text", "question", "prompt", "content"):
        if key in sample and sample[key]:
            return str(sample[key])

    return None


def _build_generators(
    names: list[str],
    teacher: torch.nn.Module,
    tokenizer: Any,
    device: str,
    seed: int,
) -> dict[str, Any]:
    """Instantiate requested generators."""
    from rcid.data.contrastive_generators import (
        EntitySwapGenerator,
        LLMGenerator,
        NumberPerturbGenerator,
    )

    mapping: dict[str, type] = {
        "entity_swap": EntitySwapGenerator,
        "number_perturb": NumberPerturbGenerator,
        "llm": LLMGenerator,
    }
    generators: dict[str, Any] = {}
    for name in names:
        cls = mapping.get(name)
        if cls is None:
            logger.warning("Unknown generator: %s (skipped)", name)
            continue
        generators[name] = cls(
            teacher=teacher, tokenizer=tokenizer, device=device, seed=seed,
        )
        logger.info("Initialized generator: %s", name)
    return generators


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Generate contrastive pairs from instruction data",
    )
    ap.add_argument("--model_name", default="Qwen/Qwen3-8B",
                     help="Teacher model HF name")
    ap.add_argument("--data_source", default="tatsu-lab/alpaca",
                     help="HuggingFace dataset name")
    ap.add_argument("--data_split", default="train",
                     help="Dataset split to use")
    ap.add_argument("--max_samples", type=int, default=10000,
                     help="Max samples to process from dataset")
    ap.add_argument("--max_pairs", type=int, default=5000,
                     help="Stop after generating this many validated pairs")
    ap.add_argument("--output_path", default="data/contrastive_pairs.json",
                     help="Output JSON path")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--generators", default="entity_swap,number_perturb",
                     help="Comma-separated generator names")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    generator_names = [g.strip() for g in args.generators.split(",")]
    logger.info(
        "Config: model=%s data=%s max_samples=%d max_pairs=%d generators=%s",
        args.model_name, args.data_source, args.max_samples,
        args.max_pairs, generator_names,
    )

    # ── Load teacher ─────────────────────────────────────────────────
    logger.info("Loading teacher model: %s", args.model_name)
    from rcid.models.teacher import load_teacher
    teacher, adapter, tokenizer = load_teacher(args.model_name, device=args.device)
    logger.info("Teacher loaded on %s", args.device)

    # ── Load dataset ─────────────────────────────────────────────────
    logger.info("Loading dataset: %s (split=%s)", args.data_source, args.data_split)
    from datasets import load_dataset
    ds = load_dataset(args.data_source, split=args.data_split)
    if len(ds) > args.max_samples:
        ds = ds.select(range(args.max_samples))
    logger.info("Using %d samples from dataset", len(ds))

    # ── Build generators & validator ─────────────────────────────────
    generators = _build_generators(
        generator_names, teacher, tokenizer, args.device, args.seed,
    )
    if not generators:
        logger.error("No valid generators configured. Exiting.")
        return

    from rcid.data.contrastive_validator import ContrastivePairValidator
    validator = ContrastivePairValidator(
        teacher=teacher, adapter=adapter, tokenizer=tokenizer,
        device=args.device,
    )

    # ── Generate & validate ──────────────────────────────────────────
    results: list[dict[str, Any]] = []
    stats: dict[str, dict[str, int]] = {
        name: {"candidates": 0, "validated": 0} for name in generators
    }
    t0 = time.time()

    for sample_idx, sample in enumerate(tqdm(ds, desc="Processing samples")):
        if len(results) >= args.max_pairs:
            break

        text = _extract_text(sample)
        if not text or len(text) < 20:
            continue

        for gen_name, gen in generators.items():
            if len(results) >= args.max_pairs:
                break

            candidates = gen.generate(text)
            stats[gen_name]["candidates"] += len(candidates)

            for clean, corrupt in candidates:
                if len(results) >= args.max_pairs:
                    break

                checks = validator.validate(clean, corrupt)
                if not all(checks.values()):
                    continue

                # Find modified positions for the record
                clean_ids = tokenizer(
                    clean, return_tensors="pt", truncation=True, max_length=512,
                ).input_ids[0]
                corrupt_ids = tokenizer(
                    corrupt, return_tensors="pt", truncation=True, max_length=512,
                ).input_ids[0]

                if clean_ids.shape[0] == corrupt_ids.shape[0]:
                    modified_pos = validator.find_modified_positions(
                        clean_ids, corrupt_ids,
                    )
                else:
                    modified_pos = []

                results.append({
                    "clean": clean,
                    "corrupt": corrupt,
                    "modified_positions": modified_pos,
                    "source": gen_name,
                    "original_data_idx": sample_idx,
                })
                stats[gen_name]["validated"] += 1

    elapsed = time.time() - t0

    # ── Save ─────────────────────────────────────────────────────────
    out_path = Path(args.output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    logger.info("Saved %d pairs to %s", len(results), out_path)

    # ── Statistics ───────────────────────────────────────────────────
    logger.info("=" * 60)
    logger.info("GENERATION STATISTICS  (%.1f s)", elapsed)
    logger.info("-" * 60)
    total_candidates = 0
    total_validated = 0
    for name, s in stats.items():
        rate = s["validated"] / max(s["candidates"], 1) * 100
        logger.info(
            "  %-18s  candidates=%5d  validated=%5d  (%.1f%%)",
            name, s["candidates"], s["validated"], rate,
        )
        total_candidates += s["candidates"]
        total_validated += s["validated"]
    logger.info("-" * 60)
    overall_rate = total_validated / max(total_candidates, 1) * 100
    logger.info(
        "  %-18s  candidates=%5d  validated=%5d  (%.1f%%)",
        "TOTAL", total_candidates, total_validated, overall_rate,
    )
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
