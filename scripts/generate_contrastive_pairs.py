"""Generate contrastive pairs from instruction data for scalable RCID.

Each generator (entity_swap, number_perturb, llm) gets an independent quota
and produces a separate JSON file, enabling per-capability analysis.

Usage:
    # Per-task generation (new default)
    python scripts/generate_contrastive_pairs.py \
        --model_name Qwen/Qwen3-8B --data_source tatsu-lab/alpaca \
        --max_pairs_per_task 2500 --output_dir data/contrastive_pairs/ \
        --device cuda:0

    # Legacy single-file mode (backward compatible)
    python scripts/generate_contrastive_pairs.py \
        --model_name Qwen/Qwen3-8B \
        --output_path data/contrastive_pairs.json \
        --max_pairs 5000

Output directory structure (per-task mode)::

    data/contrastive_pairs/
        ├── entity_swap.json
        ├── number_perturb.json
        └── llm.json              # only if --generators includes llm
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import Any

import torch
from tqdm import tqdm

# Ensure src/ is on the path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

logger = logging.getLogger(__name__)

GENERATOR_NAMES = ("entity_swap", "number_perturb", "llm")


def _extract_text(sample: dict[str, Any]) -> str | None:
    """Extract usable text from an instruction-data sample."""
    if "instruction" in sample:
        parts = [sample["instruction"]]
        inp = sample.get("input", "")
        if inp:
            parts.append(inp)
        out = sample.get("output", "")
        if out:
            parts.append(out)
        return " ".join(parts)
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


def _save_task_json(
    path: Path,
    task_type: str,
    model_name: str,
    data_source: str,
    pairs: list[dict[str, Any]],
    stats: dict[str, int],
) -> None:
    """Save one per-task JSON file with metadata envelope."""
    rate = stats["validated"] / max(stats["candidates"], 1)
    output = {
        "task_type": task_type,
        "generator": task_type,
        "model_name": model_name,
        "data_source": data_source,
        "n_pairs": len(pairs),
        "stats": {
            "candidates": stats["candidates"],
            "validated": stats["validated"],
            "rate": round(rate, 4),
        },
        "pairs": pairs,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    logger.info("Saved %d pairs → %s", len(pairs), path)


# ------------------------------------------------------------------
# Per-task generation (new default)
# ------------------------------------------------------------------

def generate_per_task(
    generators: dict[str, Any],
    validator: Any,
    tokenizer: Any,
    ds: Any,
    max_pairs_per_task: int,
    max_pairs_global: int | None,
    output_dir: Path,
    model_name: str,
    data_source: str,
) -> dict[str, dict[str, int]]:
    """Generate contrastive pairs with independent per-generator quotas.

    Each generator maintains its own results list and count. Iteration
    stops when *all* generators have reached their quota or when the
    optional global cap is hit.
    """
    # Per-generator accumulators
    results: dict[str, list[dict[str, Any]]] = {n: [] for n in generators}
    stats: dict[str, dict[str, int]] = {
        n: {"candidates": 0, "validated": 0} for n in generators
    }
    finished: set[str] = set()  # generators that hit their quota

    t0 = time.time()
    global_total = 0

    for sample_idx, sample in enumerate(tqdm(ds, desc="Processing samples")):
        # Stop if all generators done
        if len(finished) == len(generators):
            break
        # Stop if global cap reached
        if max_pairs_global is not None and global_total >= max_pairs_global:
            break

        text = _extract_text(sample)
        if not text or len(text) < 20:
            continue

        for gen_name, gen in generators.items():
            if gen_name in finished:
                continue

            candidates = gen.generate(text)
            stats[gen_name]["candidates"] += len(candidates)

            for clean, corrupt in candidates:
                if len(results[gen_name]) >= max_pairs_per_task:
                    finished.add(gen_name)
                    break
                if max_pairs_global is not None and global_total >= max_pairs_global:
                    break

                checks = validator.validate(clean, corrupt)
                if not all(checks.values()):
                    continue

                # Token-level modified positions
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

                results[gen_name].append({
                    "clean": clean,
                    "corrupt": corrupt,
                    "modified_positions": modified_pos,
                    "source": gen_name,
                    "original_data_idx": sample_idx,
                })
                stats[gen_name]["validated"] += 1
                global_total += 1

            # Check again after inner loop
            if len(results[gen_name]) >= max_pairs_per_task:
                finished.add(gen_name)

    elapsed = time.time() - t0

    # Save each generator's results to its own JSON
    for gen_name, pairs in results.items():
        out_path = output_dir / f"{gen_name}.json"
        _save_task_json(
            out_path, gen_name, model_name, data_source, pairs, stats[gen_name],
        )

    _print_stats(stats, elapsed, global_total)
    return stats


# ------------------------------------------------------------------
# Legacy single-file generation (backward compatible)
# ------------------------------------------------------------------

def generate_single_file(
    generators: dict[str, Any],
    validator: Any,
    tokenizer: Any,
    ds: Any,
    max_pairs: int,
    output_path: Path,
) -> dict[str, dict[str, int]]:
    """Original logic: all generators share one global max_pairs cap."""
    results: list[dict[str, Any]] = []
    stats: dict[str, dict[str, int]] = {
        n: {"candidates": 0, "validated": 0} for n in generators
    }
    t0 = time.time()

    for sample_idx, sample in enumerate(tqdm(ds, desc="Processing samples")):
        if len(results) >= max_pairs:
            break
        text = _extract_text(sample)
        if not text or len(text) < 20:
            continue

        for gen_name, gen in generators.items():
            if len(results) >= max_pairs:
                break
            candidates = gen.generate(text)
            stats[gen_name]["candidates"] += len(candidates)

            for clean, corrupt in candidates:
                if len(results) >= max_pairs:
                    break
                checks = validator.validate(clean, corrupt)
                if not all(checks.values()):
                    continue

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

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    logger.info("Saved %d pairs to %s", len(results), output_path)

    _print_stats(stats, elapsed, len(results))
    return stats


# ------------------------------------------------------------------
# Shared
# ------------------------------------------------------------------

def _print_stats(
    stats: dict[str, dict[str, int]], elapsed: float, total: int,
) -> None:
    """Log generation statistics."""
    logger.info("=" * 62)
    logger.info("GENERATION STATISTICS  (%.1f s)", elapsed)
    logger.info("-" * 62)
    total_cand = 0
    total_val = 0
    for name, s in stats.items():
        rate = s["validated"] / max(s["candidates"], 1) * 100
        logger.info(
            "  %-20s  candidates=%5d  validated=%5d  (%.1f%%)",
            name, s["candidates"], s["validated"], rate,
        )
        total_cand += s["candidates"]
        total_val += s["validated"]
    logger.info("-" * 62)
    overall = total_val / max(total_cand, 1) * 100
    logger.info(
        "  %-20s  candidates=%5d  validated=%5d  (%.1f%%)",
        "TOTAL", total_cand, total_val, overall,
    )
    logger.info("=" * 62)


# ------------------------------------------------------------------
# CLI
# ------------------------------------------------------------------

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
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--generators", default="entity_swap,number_perturb",
                     help="Comma-separated generator names")
    ap.add_argument("--seed", type=int, default=42)

    # ── Per-task mode (new default) ──────────────────────────────────
    ap.add_argument("--max_pairs_per_task", type=int, default=2500,
                     help="Independent quota per generator (per-task mode)")
    ap.add_argument("--output_dir", default=None,
                     help="Output directory for per-task JSONs "
                          "(e.g. data/contrastive_pairs/)")

    # ── Legacy single-file mode ─────────────────────────────────────
    ap.add_argument("--max_pairs", type=int, default=None,
                     help="Global pair cap. In per-task mode this is an "
                          "optional overall ceiling; in legacy mode it is "
                          "the sole limit (default: no global cap)")
    ap.add_argument("--output_path", default=None,
                     help="Legacy: single output JSON file. If set, "
                          "uses old shared-quota logic")

    args = ap.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Decide mode: legacy (--output_path) vs per-task (--output_dir)
    use_legacy = args.output_path is not None
    if not use_legacy and args.output_dir is None:
        args.output_dir = "data/contrastive_pairs/"

    generator_names = [g.strip() for g in args.generators.split(",")]

    if use_legacy:
        logger.info(
            "Mode: LEGACY single-file  max_pairs=%s  output=%s",
            args.max_pairs, args.output_path,
        )
    else:
        logger.info(
            "Mode: PER-TASK  max_per_task=%d  global_cap=%s  output_dir=%s",
            args.max_pairs_per_task, args.max_pairs, args.output_dir,
        )
    logger.info(
        "Config: model=%s data=%s max_samples=%d generators=%s",
        args.model_name, args.data_source, args.max_samples, generator_names,
    )

    # ── Load teacher (once, shared by all generators) ────────────────
    logger.info("Loading teacher model: %s", args.model_name)
    from rcid.models.teacher import load_teacher
    teacher, adapter, tokenizer = load_teacher(
        args.model_name, device=args.device,
    )
    logger.info("Teacher loaded on %s", args.device)

    # ── Load dataset ─────────────────────────────────────────────────
    logger.info(
        "Loading dataset: %s (split=%s)", args.data_source, args.data_split,
    )
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

    # ── Generate ─────────────────────────────────────────────────────
    if use_legacy:
        max_p = args.max_pairs if args.max_pairs is not None else 5000
        generate_single_file(
            generators, validator, tokenizer, ds,
            max_pairs=max_p,
            output_path=Path(args.output_path),
        )
    else:
        generate_per_task(
            generators, validator, tokenizer, ds,
            max_pairs_per_task=args.max_pairs_per_task,
            max_pairs_global=args.max_pairs,
            output_dir=Path(args.output_dir),
            model_name=args.model_name,
            data_source=args.data_source,
        )


if __name__ == "__main__":
    main()
