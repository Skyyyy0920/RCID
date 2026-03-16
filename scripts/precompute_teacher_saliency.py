"""Precompute teacher input saliency for SaGD (Saliency-Guided KD).

Iterates over an InstructionDataset, computes the teacher model's input
saliency (gradient norm of response log-likelihood w.r.t. embeddings),
and saves the result as a ``.pt`` file for use during training.

Usage::

    python scripts/precompute_teacher_saliency.py \
        --model_family qwen3 \
        --data_source databricks/databricks-dolly-15k \
        --output_path data/teacher_saliency_qwen3.pt \
        --device cuda:0

Output format::

    {
        "saliency": [Tensor(L_0,), Tensor(L_1,), ...],  # per-sample
        "metadata": {
            "model": "Qwen/Qwen3-8B",
            "dataset": "databricks/databricks-dolly-15k",
            "n_samples": 12859,
            "saliency_temperature": 2.0,
        }
    }
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from rcid.models.teacher import load_teacher
from rcid.data.instruction_dataset import InstructionDataset
from rcid.distillation.saliency import SaliencyComputer

logger = logging.getLogger(__name__)

MODEL_CONFIGS = {
    "qwen3": "Qwen/Qwen3-8B",
    "llama3": "meta-llama/Llama-3.1-8B",
}


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Precompute teacher saliency for SaGD",
    )
    ap.add_argument("--model_family", choices=["qwen3", "llama3"], default="qwen3")
    ap.add_argument("--data_source", default="databricks/databricks-dolly-15k")
    ap.add_argument("--max_samples", type=int, default=None)
    ap.add_argument("--max_seq_len", type=int, default=512)
    ap.add_argument("--batch_size", type=int, default=4,
                    help="Batch size for saliency computation")
    ap.add_argument("--saliency_temperature", type=float, default=2.0)
    ap.add_argument("--output_path", type=str, required=True,
                    help="Where to save the .pt cache")
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    model_name = MODEL_CONFIGS[args.model_family]
    logger.info("Loading teacher: %s", model_name)
    teacher, _, tokenizer = load_teacher(model_name, device=args.device)

    logger.info("Loading dataset: %s (max=%s)", args.data_source, args.max_samples)
    dataset = InstructionDataset(
        dataset_name=args.data_source, tokenizer=tokenizer,
        max_seq_len=args.max_seq_len, max_samples=args.max_samples,
        split="train",
    )

    loader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=False,
        collate_fn=dataset.collate_fn,
    )

    computer = SaliencyComputer(temperature=args.saliency_temperature)

    # Precompute saliency for every sample (store by index)
    saliency_dict: dict[int, torch.Tensor] = {}
    n_samples = len(dataset)
    logger.info("Computing saliency for %d samples...", n_samples)

    t0 = time.time()
    for batch in tqdm(loader, desc="Saliency"):
        input_ids = batch["input_ids"].to(args.device)        # (B, L)
        attention_mask = batch["attention_mask"].to(args.device)
        labels_mask = batch["labels_mask"].to(args.device)
        indices = batch["index"]                                # (B,)

        saliency = computer.compute(
            teacher, input_ids, attention_mask, labels_mask,
        )  # (B, L)

        # Store per-sample, trimmed to actual length
        for i, idx in enumerate(indices):
            actual_len = int(attention_mask[i].sum().item())
            saliency_dict[idx.item()] = saliency[i, :actual_len].cpu()

    elapsed = time.time() - t0
    logger.info("Saliency computed in %.1f s (%.2f s/sample)",
                elapsed, elapsed / max(n_samples, 1))

    # Convert to ordered list
    saliency_list = [saliency_dict[i] for i in range(n_samples)]

    # Save
    out_path = Path(args.output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    cache = {
        "saliency": saliency_list,
        "metadata": {
            "model": model_name,
            "dataset": args.data_source,
            "n_samples": n_samples,
            "saliency_temperature": args.saliency_temperature,
            "max_seq_len": args.max_seq_len,
        },
    }
    torch.save(cache, out_path)
    logger.info("Saved teacher saliency cache: %s (%d samples)", out_path, n_samples)


if __name__ == "__main__":
    main()
