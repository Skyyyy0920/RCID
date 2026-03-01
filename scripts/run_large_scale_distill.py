"""Large-scale distillation with adaptive KL methods on Dolly-15K.

Trains a student on instruction data with sequence-level KL, using one of
several adaptive divergence methods.  Evaluation uses ROUGE-L on the
Dolly test split.

Methods
-------
  standard_kd       — forward KL baseline
  reverse_kl        — reverse KL baseline
  standard_kd_akl   — AKL (Wu et al., COLING 2025)
  standard_kd_klr   — KL-Ratio adaptive (ours)

Usage::

    python scripts/run_large_scale_distill.py --model_family qwen3 \
        --method standard_kd --device cuda:0

    python scripts/run_large_scale_distill.py --model_family qwen3 \
        --method standard_kd_klr --klr_granularity batch --klr_beta 0.99

Output directory structure::

    outputs/large_scale/{model_family}/{method}/seed_{seed}/
        ├── student_final.pt
        ├── training_log.json
        ├── training_stats.jsonl
        ├── eval_results.json        (ROUGE-L)
        └── test_generations.json    (generated texts for inspection)
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

# Ensure src/ is on the path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from rcid import set_all_seeds
from rcid.models.teacher import load_teacher
from rcid.models.student import load_student

logger = logging.getLogger(__name__)

MODEL_CONFIGS: dict[str, dict[str, str]] = {
    "qwen3": {"teacher": "Qwen/Qwen3-8B", "student": "Qwen/Qwen3-0.6B"},
    "llama3": {
        "teacher": "meta-llama/Llama-3.1-8B",
        "student": "meta-llama/Llama-3.2-1B",
    },
}

METHODS = [
    "standard_kd",
    "reverse_kl",
    "standard_kd_akl",
    "standard_kd_klr",
]

DEFAULT_DATASET = "databricks/databricks-dolly-15k"


# ------------------------------------------------------------------
# ROUGE-L evaluation
# ------------------------------------------------------------------

def _evaluate_rouge(
    student: torch.nn.Module,
    tokenizer: Any,
    save_dir: Path,
    device: str,
) -> dict[str, Any]:
    """Evaluate on Dolly test split with ROUGE-L."""
    from rcid.data.dolly_utils import get_dolly_prompts
    from rcid.eval.rouge_eval import evaluate_rouge, save_generations

    eval_prompts, eval_refs = get_dolly_prompts(split="test")
    logger.info("Evaluating ROUGE-L on %d Dolly test samples...", len(eval_prompts))

    results = evaluate_rouge(
        student, tokenizer, eval_prompts, eval_refs,
        max_new_tokens=256, batch_size=8,
    )
    logger.info("Test ROUGE-L: %.4f", results["rouge_l_f"])

    # Save eval results
    with open(save_dir / "eval_results.json", "w", encoding="utf-8") as f:
        json.dump(
            {k: v for k, v in results.items() if k != "generations"},
            f, indent=2,
        )

    # Save generations for human inspection
    save_generations(
        eval_prompts, results["generations"], eval_refs,
        filepath=str(save_dir / "test_generations.json"),
        rouge_scores=results.get("per_sample_rouge_l_f"),
    )

    return results


# ------------------------------------------------------------------
# Main run function
# ------------------------------------------------------------------

def run_single(
    method: str,
    model_family: str,
    seed: int,
    device: str,
    data_source: str,
    max_train_samples: int | None,
    output_dir: str,
    *,
    epochs: int = 3,
    batch_size: int = 8,
    gradient_accumulation: int = 4,
    lr: float = 2e-5,
    max_seq_len: int = 512,
    fp16: bool = True,
    temperature: float = 2.0,
    # Adaptive method options
    klr_granularity: str = "token",
    klr_beta: float = 0.99,
    klr_fixed_alpha: float | None = None,
    akl_mu: float = 0.5,
    # Eval
    skip_eval: bool = False,
) -> dict[str, Any]:
    """Full pipeline for one run: load → train → eval → save."""
    set_all_seeds(seed)
    mcfg = MODEL_CONFIGS[model_family]

    # ── Output directory ────────────────────────────────────────────
    run_dir = Path(output_dir) / model_family / method / f"seed_{seed}"
    run_dir.mkdir(parents=True, exist_ok=True)

    # ── Save config.json ────────────────────────────────────────────
    run_config: dict[str, Any] = {
        "method": method,
        "model_family": model_family,
        "teacher": mcfg["teacher"],
        "student": mcfg["student"],
        "seed": seed,
        "data_source": data_source,
        "max_train_samples": max_train_samples,
        "epochs": epochs,
        "batch_size": batch_size,
        "gradient_accumulation": gradient_accumulation,
        "effective_batch_size": batch_size * gradient_accumulation,
        "lr": lr,
        "temperature": temperature,
        "max_seq_len": max_seq_len,
        "fp16": fp16,
        "device": device,
        "klr_granularity": klr_granularity,
        "klr_beta": klr_beta,
        "klr_fixed_alpha": klr_fixed_alpha,
        "akl_mu": akl_mu,
    }
    with open(run_dir / "config.json", "w", encoding="utf-8") as f:
        json.dump(run_config, f, indent=2)

    # ── Load models ─────────────────────────────────────────────────
    logger.info("Loading teacher: %s", mcfg["teacher"])
    teacher, t_adp, tokenizer = load_teacher(mcfg["teacher"], device=device)
    logger.info("Loading student: %s", mcfg["student"])
    student, s_adp, _ = load_student(mcfg["student"], device=device)

    # ── Instruction dataset ─────────────────────────────────────────
    from rcid.data.instruction_dataset import InstructionDataset

    logger.info(
        "Loading instruction dataset: %s (max=%s)", data_source, max_train_samples,
    )
    main_ds = InstructionDataset(
        dataset_name=data_source,
        tokenizer=tokenizer,
        max_seq_len=max_seq_len,
        max_samples=max_train_samples,
        split="train",
    )
    logger.info("Instruction dataset ready: %d samples", len(main_ds))

    # ── Build trainer config dict ───────────────────────────────────
    trainer_cfg: dict[str, Any] = {
        "method": method,
        "epochs": epochs,
        "batch_size": batch_size,
        "gradient_accumulation": gradient_accumulation,
        "lr": lr,
        "weight_decay": 0.01,
        "max_grad_norm": 1.0,
        "warmup_ratio": 0.03,
        "fp16": fp16,
        "temperature": temperature,
        "save_every_n_epochs": 1,
        "use_wandb": False,
        "log_every": 50,
        "jsonl_every": 100,
        # Adaptive method params
        "klr_granularity": klr_granularity,
        "klr_beta": klr_beta,
        "klr_fixed_alpha": klr_fixed_alpha,
        "akl_mu": akl_mu,
    }

    # ── Create ScalableDistillationTrainer ──────────────────────────
    from rcid.distillation.scalable_trainer import ScalableDistillationTrainer

    trainer = ScalableDistillationTrainer(
        teacher=teacher,
        student=student,
        teacher_adapter=t_adp,
        student_adapter=s_adp,
        tokenizer=tokenizer,
        main_dataset=main_ds,
        config=trainer_cfg,
    )

    # ── Train ───────────────────────────────────────────────────────
    logger.info(
        "Training: method=%s  epochs=%d  batch=%dx%d  lr=%.2e",
        method, epochs, batch_size, gradient_accumulation, lr,
    )
    t0 = time.time()
    history = trainer.train(save_dir=str(run_dir))
    train_secs = time.time() - t0
    logger.info("Training complete in %.1f s", train_secs)

    # ── Evaluate (KL on main data) ──────────────────────────────────
    eval_kl = trainer.evaluate()

    # ── Save student_final.pt ───────────────────────────────────────
    final_ckpt = run_dir / "student_final.pt"
    torch.save(student.state_dict(), final_ckpt)
    logger.info("Final checkpoint: %s", final_ckpt)

    # ── ROUGE-L evaluation on Dolly test split ──────────────────────
    rouge_results: dict[str, Any] = {}
    if not skip_eval:
        try:
            rouge_results = _evaluate_rouge(student, tokenizer, run_dir, device)
        except Exception as e:
            logger.error("ROUGE-L evaluation failed: %s", e)
            rouge_results = {"error": str(e)}

    # ── Save training_log.json ──────────────────────────────────────
    training_log: dict[str, Any] = {
        "method": method,
        "model_family": model_family,
        "seed": seed,
        "train_time_sec": round(train_secs, 1),
        "n_main_samples": len(main_ds),
        "history": history,
        "eval_kl_loss": eval_kl.get("kl_loss"),
        "final_loss": history["loss"][-1] if history.get("loss") else None,
        "rouge_l_f": rouge_results.get("rouge_l_f"),
        "checkpoint_path": str(final_ckpt),
    }
    with open(run_dir / "training_log.json", "w", encoding="utf-8") as f:
        json.dump(training_log, f, indent=2)
    logger.info("Training log: %s", run_dir / "training_log.json")

    return training_log


# ------------------------------------------------------------------
# CLI
# ------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(
        description="Large-scale distillation with adaptive KL methods",
    )
    ap.add_argument(
        "--model_family", choices=["qwen3", "llama3"], default="qwen3",
    )
    ap.add_argument(
        "--method", choices=METHODS, default="standard_kd",
        help="Distillation method (default: standard_kd)",
    )
    ap.add_argument(
        "--data_source", default=DEFAULT_DATASET,
        help="HuggingFace dataset for instruction KL (default: Dolly-15K)",
    )
    ap.add_argument("--max_train_samples", type=int, default=None)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--gradient_accumulation", type=int, default=4)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--max_seq_len", type=int, default=512)
    ap.add_argument("--temperature", type=float, default=2.0)
    ap.add_argument("--fp16", action="store_true", default=True)
    ap.add_argument("--no_fp16", dest="fp16", action="store_false")
    # Adaptive method options
    ap.add_argument("--klr_granularity", default="token", choices=["token", "batch"])
    ap.add_argument("--klr_beta", type=float, default=0.99)
    ap.add_argument("--klr_fixed_alpha", type=float, default=None)
    ap.add_argument("--akl_mu", type=float, default=0.5)
    # General
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--output_dir", default="outputs/large_scale")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--skip_eval", action="store_true", help="Skip ROUGE-L eval")
    args = ap.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    logger.info(
        "=== Large-scale distill: method=%s  family=%s  seed=%d  data=%s ===",
        args.method, args.model_family, args.seed, args.data_source,
    )

    result = run_single(
        method=args.method,
        model_family=args.model_family,
        seed=args.seed,
        device=args.device,
        data_source=args.data_source,
        max_train_samples=args.max_train_samples,
        output_dir=args.output_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        gradient_accumulation=args.gradient_accumulation,
        lr=args.lr,
        max_seq_len=args.max_seq_len,
        fp16=args.fp16,
        temperature=args.temperature,
        klr_granularity=args.klr_granularity,
        klr_beta=args.klr_beta,
        klr_fixed_alpha=args.klr_fixed_alpha,
        akl_mu=args.akl_mu,
        skip_eval=args.skip_eval,
    )

    logger.info(
        "Done. train_time=%.1fs  final_loss=%s  eval_kl=%s  rouge_l=%.4f",
        result["train_time_sec"],
        result["final_loss"],
        result["eval_kl_loss"],
        result.get("rouge_l_f") or 0.0,
    )


if __name__ == "__main__":
    main()
