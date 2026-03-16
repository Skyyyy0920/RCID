"""Diagnose saliency alignment between a trained student and cached teacher.

Computes JSD statistics between a trained student's input saliency and the
precomputed teacher saliency cache.  Useful for understanding how well SaGD
has aligned the student's attention patterns.

Usage::

    python scripts/diagnose_saliency.py \
        --student_checkpoint outputs/large_scale/qwen3/standard_kd_sagd/seed_42/student_final.pt \
        --teacher_saliency_path data/teacher_saliency_qwen3.pt \
        --model_family qwen3 \
        --output_path outputs/diagnostics/sagd_jsd_report.json \
        --device cuda:0

Output JSON::

    {
        "n_samples": 12859,
        "jsd_mean": 0.152,
        "jsd_std": 0.083,
        "jsd_median": 0.138,
        "jsd_p90": 0.271,
        "top20_indices": [3421, 892, ...],
        "top20_jsd": [0.491, 0.478, ...],
    }
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
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from rcid.models.student import load_student
from rcid.data.instruction_dataset import InstructionDataset
from rcid.distillation.saliency import SaliencyComputer

logger = logging.getLogger(__name__)

MODEL_CONFIGS: dict[str, dict[str, str]] = {
    "qwen3": {"student": "Qwen/Qwen3-0.6B"},
    "llama3": {"student": "meta-llama/Llama-3.2-1B"},
}

DEFAULT_DATASET = "databricks/databricks-dolly-15k"


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Diagnose saliency alignment (student vs teacher)",
    )
    ap.add_argument("--student_checkpoint", type=str, required=True,
                    help="Path to trained student .pt checkpoint")
    ap.add_argument("--teacher_saliency_path", type=str, required=True,
                    help="Path to precomputed teacher saliency .pt cache")
    ap.add_argument("--model_family", choices=["qwen3", "llama3"], default="qwen3")
    ap.add_argument("--data_source", default=DEFAULT_DATASET)
    ap.add_argument("--max_samples", type=int, default=None)
    ap.add_argument("--max_seq_len", type=int, default=512)
    ap.add_argument("--batch_size", type=int, default=4)
    ap.add_argument("--saliency_temperature", type=float, default=2.0)
    ap.add_argument("--output_path", type=str, default=None,
                    help="Where to save JSON report (optional)")
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    # ── Load student ────────────────────────────────────────────────
    mcfg = MODEL_CONFIGS[args.model_family]
    logger.info("Loading student: %s", mcfg["student"])
    student, s_adp, tokenizer = load_student(mcfg["student"], device=args.device)
    student.load_state_dict(
        torch.load(args.student_checkpoint, map_location=args.device)
    )
    student.eval()

    # ── Load teacher saliency cache ─────────────────────────────────
    logger.info("Loading teacher saliency: %s", args.teacher_saliency_path)
    cache = torch.load(
        args.teacher_saliency_path, map_location="cpu", weights_only=False,
    )
    teacher_saliencies: list[torch.Tensor] = cache["saliency"]
    logger.info(
        "Teacher saliency: %d samples (from %s)",
        len(teacher_saliencies), cache.get("metadata", {}).get("model", "?"),
    )

    # ── Load dataset ────────────────────────────────────────────────
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

    # ── Compute student saliency + JSD ──────────────────────────────
    all_jsd: list[float] = []
    sample_indices: list[int] = []

    logger.info("Computing student saliency for %d samples...", len(dataset))
    t0 = time.time()

    for batch in tqdm(loader, desc="Saliency diagnosis"):
        ids = batch["input_ids"].to(args.device)           # (B, L)
        attn = batch["attention_mask"].to(args.device)
        labels_mask = batch["labels_mask"].to(args.device)
        indices = batch["index"]                             # (B,)

        # Student saliency
        s_sal = computer.compute(student, ids, attn.long(), labels_mask)  # (B, L)

        B, L = ids.shape
        t_sal = torch.zeros(B, L, device=args.device)
        for j, idx_t in enumerate(indices):
            idx_val = idx_t.item()
            if idx_val < len(teacher_saliencies):
                s = teacher_saliencies[idx_val]
                L_s = min(len(s), L)
                t_sal[j, :L_s] = s[:L_s].to(args.device)

        # Convert to distributions
        t_dist = computer.to_distribution(t_sal, labels_mask, attention_mask=attn)
        s_dist = computer.to_distribution(s_sal, labels_mask, attention_mask=attn)

        # JSD
        jsd = computer.divergence(t_dist, s_dist, labels_mask)  # (B,)

        for j in range(B):
            all_jsd.append(jsd[j].item())
            sample_indices.append(indices[j].item())

    elapsed = time.time() - t0
    logger.info("Diagnosis complete in %.1f s", elapsed)

    # ── Statistics ──────────────────────────────────────────────────
    jsd_tensor = torch.tensor(all_jsd)
    n = len(jsd_tensor)

    report: dict[str, Any] = {
        "n_samples": n,
        "jsd_mean": round(jsd_tensor.mean().item(), 6),
        "jsd_std": round(jsd_tensor.std().item(), 6),
        "jsd_median": round(jsd_tensor.median().item(), 6),
        "jsd_p90": round(jsd_tensor.quantile(0.9).item(), 6),
        "jsd_p95": round(jsd_tensor.quantile(0.95).item(), 6),
        "jsd_max": round(jsd_tensor.max().item(), 6),
        "jsd_min": round(jsd_tensor.min().item(), 6),
    }

    # Top-20 hardest samples
    top_k = min(20, n)
    top_vals, top_idx = jsd_tensor.topk(top_k)
    report["top20_indices"] = [sample_indices[i] for i in top_idx.tolist()]
    report["top20_jsd"] = [round(v, 6) for v in top_vals.tolist()]

    logger.info("JSD stats: mean=%.4f  std=%.4f  median=%.4f  p90=%.4f",
                report["jsd_mean"], report["jsd_std"],
                report["jsd_median"], report["jsd_p90"])

    # ── Save report ─────────────────────────────────────────────────
    if args.output_path:
        out = Path(args.output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)
        logger.info("Report saved: %s", out)
    else:
        print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
