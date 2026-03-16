"""Paper experiments: unified entry point for all KL-Ratio experiments.

Runs training on Dolly-15K + ROUGE-L evaluation on Dolly test split.

Usage::

    # Single experiment
    python scripts/run_paper_experiments.py --experiment forward_kl --device cuda:0

    # Eval-only (re-evaluate existing checkpoint)
    python scripts/run_paper_experiments.py --experiment klr_batch_ema \
        --eval_only --device cuda:0

    # Custom output directory
    python scripts/run_paper_experiments.py --experiment akl \
        --output_dir outputs/paper --device cuda:0

Available experiments:
  Main table:  forward_kl, reverse_kl, jeffreys, akl, klr_token, klr_batch_ema
  Beta ablation: klr_no_ema, klr_beta_0.9, klr_beta_0.95, klr_beta_0.999
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

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from rcid import set_all_seeds
from rcid.models.teacher import load_teacher
from rcid.models.student import load_student

logger = logging.getLogger(__name__)

# ======================================================================
# Experiment configurations
# ======================================================================

EXPERIMENT_CONFIGS: dict[str, dict[str, Any]] = {
    # ── Main Table ────────────────────────────────────────────────────
    "forward_kl": {
        "method": "standard_kd",
        "description": "Standard forward KL baseline",
    },
    "reverse_kl": {
        "method": "reverse_kl",
        "description": "Reverse KL baseline (via KLRatioLoss fixed_alpha=0)",
    },
    "jeffreys": {
        "method": "standard_kd_klr",
        "klr_granularity": "batch",
        "klr_fixed_alpha": 0.5,
        "description": "Jeffreys divergence (fixed alpha=0.5)",
    },
    "akl": {
        "method": "standard_kd_akl",
        "akl_mu": 0.5,
        "description": "AKL (Wu et al., COLING 2025)",
    },
    "klr_token": {
        "method": "standard_kd_klr",
        "klr_granularity": "token",
        "description": "KL-Ratio token-level (ours)",
    },
    "klr_batch_ema": {
        "method": "standard_kd_klr",
        "klr_granularity": "batch",
        "klr_beta": 0.99,
        "description": "KL-Ratio batch + EMA beta=0.99 (ours)",
    },
    # ── Beta Ablation ─────────────────────────────────────────────────
    "klr_no_ema": {
        "method": "standard_kd_klr",
        "klr_granularity": "batch",
        "klr_beta": 0.0,
        "description": "KL-Ratio batch, no EMA (beta=0)",
    },
    "klr_beta_0.9": {
        "method": "standard_kd_klr",
        "klr_granularity": "batch",
        "klr_beta": 0.9,
        "description": "KL-Ratio batch, beta=0.9",
    },
    "klr_beta_0.95": {
        "method": "standard_kd_klr",
        "klr_granularity": "batch",
        "klr_beta": 0.95,
        "description": "KL-Ratio batch, beta=0.95",
    },
    "klr_beta_0.999": {
        "method": "standard_kd_klr",
        "klr_granularity": "batch",
        "klr_beta": 0.999,
        "description": "KL-Ratio batch, beta=0.999",
    },
    # ── SaGD (Saliency-Guided KD) ────────────────────────────────────
    "sagd": {
        "method": "standard_kd_sagd",
        "teacher_saliency_path": "data/teacher_saliency_qwen3.pt",
        "sagd_every_n_steps": 1,
        "sagd_tau_w": 1.0,
        "saliency_temperature": 2.0,
        "description": "SaGD: saliency-guided sample reweighting (tau_w=1.0)",
    },
    # Ablation: tau_w=100.0 makes softmax(jsd/tau_w) nearly uniform across
    # all samples, so per-sample reweighting has almost no effect.
    # This isolates the contribution of the saliency alignment loss
    # (L_sal = cosine distance between teacher/student saliency vectors).
    "sagd_loss_only": {
        "method": "standard_kd_sagd",
        "teacher_saliency_path": "data/teacher_saliency_qwen3.pt",
        "sagd_every_n_steps": 1,
        "sagd_tau_w": 100.0,
        "saliency_temperature": 2.0,
        "description": "SaGD ablation: near-uniform weights (tau_w=100)",
    },
    "sagd_reweight_only": {
        "method": "standard_kd_sagd",
        "teacher_saliency_path": "data/teacher_saliency_qwen3.pt",
        "sagd_every_n_steps": 1,
        "sagd_tau_w": 0.1,
        "lambda_sal": 0.0,
        "saliency_temperature": 2.0,
        "description": "SaGD ablation: reweighting only, no L_sal (tau_w=0.1, lambda_sal=0)",
    },
}

# Shared training hyper-parameters (all experiments must match)
TRAINING_DEFAULTS: dict[str, Any] = {
    "teacher_model": "Qwen/Qwen3-8B",
    "student_model": "Qwen/Qwen3-0.6B",
    "dataset": "databricks/databricks-dolly-15k",
    "max_train_samples": None,  # Use full Dolly train split (~14k)
    "epochs": 3,
    "batch_size": 8,
    "gradient_accumulation": 4,
    "lr": 2e-5,
    "warmup_ratio": 0.03,
    "max_seq_len": 512,
    "temperature": 2.0,
    "fp16": True,
    "seed": 42,
}

EVAL_BATCH_SIZE = 8


# ------------------------------------------------------------------
# Training
# ------------------------------------------------------------------

def run_training(
    exp_name: str,
    exp_cfg: dict[str, Any],
    *,
    device: str,
    output_dir: str,
) -> dict[str, Any]:
    """Train one experiment and save checkpoint + stats."""
    cfg = {**TRAINING_DEFAULTS}
    run_dir = Path(output_dir) / exp_name
    run_dir.mkdir(parents=True, exist_ok=True)

    seed = cfg["seed"]
    set_all_seeds(seed)

    # Merge experiment-specific overrides into trainer config
    trainer_cfg: dict[str, Any] = {
        "method": exp_cfg["method"],
        "epochs": cfg["epochs"],
        "batch_size": cfg["batch_size"],
        "gradient_accumulation": cfg["gradient_accumulation"],
        "lr": cfg["lr"],
        "weight_decay": 0.01,
        "max_grad_norm": 1.0,
        "warmup_ratio": cfg["warmup_ratio"],
        "fp16": cfg["fp16"],
        "temperature": cfg["temperature"],
        "save_every_n_epochs": 1,
        "use_wandb": False,
        "log_every": 50,
        "jsonl_every": 100,
    }
    # Copy experiment-specific keys (klr_granularity, klr_beta, etc.)
    for k, v in exp_cfg.items():
        if k not in ("method", "description"):
            trainer_cfg[k] = v

    # Save full config
    full_config = {
        "experiment": exp_name,
        "description": exp_cfg.get("description", ""),
        "training": cfg,
        "trainer": trainer_cfg,
    }
    with open(run_dir / "config.json", "w", encoding="utf-8") as f:
        json.dump(full_config, f, indent=2)

    # ── Load models ───────────────────────────────────────────────
    logger.info("Loading teacher: %s", cfg["teacher_model"])
    teacher, t_adp, tokenizer = load_teacher(
        cfg["teacher_model"], device=device,
    )
    logger.info("Loading student: %s", cfg["student_model"])
    student, s_adp, _ = load_student(cfg["student_model"], device=device)

    # ── Dataset ───────────────────────────────────────────────────
    from rcid.data.instruction_dataset import InstructionDataset

    logger.info("Loading data: %s (max=%s)", cfg["dataset"], cfg["max_train_samples"])
    main_ds = InstructionDataset(
        dataset_name=cfg["dataset"],
        tokenizer=tokenizer,
        max_seq_len=cfg["max_seq_len"],
        max_samples=cfg["max_train_samples"],
    )
    logger.info("Dataset: %d samples", len(main_ds))

    # ── Train ─────────────────────────────────────────────────────
    from rcid.distillation.scalable_trainer import ScalableDistillationTrainer

    trainer = ScalableDistillationTrainer(
        teacher=teacher, student=student,
        teacher_adapter=t_adp, student_adapter=s_adp,
        tokenizer=tokenizer,
        main_dataset=main_ds,
        config=trainer_cfg,
    )

    logger.info(
        "Training: exp=%s  method=%s  epochs=%d  batch=%dx%d  lr=%.2e",
        exp_name, trainer_cfg["method"], cfg["epochs"],
        cfg["batch_size"], cfg["gradient_accumulation"], cfg["lr"],
    )
    t0 = time.time()
    history = trainer.train(save_dir=str(run_dir))
    train_secs = time.time() - t0
    logger.info("Training complete in %.1f s", train_secs)

    # ── Eval KL ───────────────────────────────────────────────────
    eval_metrics = trainer.evaluate()

    # ── Save checkpoint ───────────────────────────────────────────
    final_ckpt = run_dir / "student_final.pt"
    torch.save(student.state_dict(), final_ckpt)
    logger.info("Final checkpoint: %s", final_ckpt)

    # ── Save training_stats.json ──────────────────────────────────
    training_stats: dict[str, Any] = {
        "experiment": exp_name,
        "method": trainer_cfg["method"],
        "seed": seed,
        "train_time_sec": round(train_secs, 1),
        "n_samples": len(main_ds),
        "history": history,
        "eval_kl_loss": eval_metrics.get("kl_loss"),
        "final_loss": history["loss"][-1] if history.get("loss") else None,
        "checkpoint_path": str(final_ckpt),
    }
    with open(run_dir / "training_stats.json", "w", encoding="utf-8") as f:
        json.dump(training_stats, f, indent=2)

    return training_stats


# ------------------------------------------------------------------
# Evaluation (ROUGE-L on Dolly test split)
# ------------------------------------------------------------------

def run_eval(
    exp_name: str,
    student_model: str,
    checkpoint_path: str,
    output_dir: str,
    device: str,
    batch_size: int = 8,
) -> dict[str, Any]:
    """Evaluate checkpoint with ROUGE-L on Dolly test split."""
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from rcid.data.dolly_utils import get_dolly_prompts
    from rcid.eval.rouge_eval import evaluate_rouge, save_generations

    run_dir = Path(output_dir) / exp_name

    logger.info("Loading student for eval: %s", student_model)
    tokenizer = AutoTokenizer.from_pretrained(student_model)
    model = AutoModelForCausalLM.from_pretrained(
        student_model, torch_dtype=torch.float16,
    ).to(device)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()

    eval_prompts, eval_refs = get_dolly_prompts(split="test")
    logger.info("Evaluating ROUGE-L on %d Dolly test samples...", len(eval_prompts))

    results = evaluate_rouge(
        model, tokenizer, eval_prompts, eval_refs,
        max_new_tokens=256, batch_size=batch_size,
    )
    logger.info("Test ROUGE-L: %.4f", results["rouge_l_f"])

    # Save eval results (without full generations list)
    out_json = run_dir / "eval_results.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(
            {k: v for k, v in results.items() if k != "generations"},
            f, indent=2,
        )

    # Save generations for inspection
    save_generations(
        eval_prompts, results["generations"], eval_refs,
        filepath=str(run_dir / "test_generations.json"),
        rouge_scores=results.get("per_sample_rouge_l_f"),
    )

    return results


# ------------------------------------------------------------------
# CLI
# ------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(description="Paper experiments entry point")
    ap.add_argument(
        "--experiment", required=True,
        choices=list(EXPERIMENT_CONFIGS.keys()),
        help="Experiment name",
    )
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--output_dir", default="outputs/paper")
    ap.add_argument(
        "--eval_only", action="store_true",
        help="Skip training, only run ROUGE-L eval on existing checkpoint",
    )
    ap.add_argument(
        "--skip_eval", action="store_true",
        help="Skip ROUGE-L eval after training",
    )
    ap.add_argument("--eval_batch_size", type=int, default=EVAL_BATCH_SIZE)
    args = ap.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    exp_name = args.experiment
    exp_cfg = EXPERIMENT_CONFIGS[exp_name]
    run_dir = Path(args.output_dir) / exp_name

    logger.info("=" * 60)
    logger.info("  Experiment: %s", exp_name)
    logger.info("  Description: %s", exp_cfg.get("description", ""))
    logger.info("=" * 60)

    # ── Training ──────────────────────────────────────────────────
    if not args.eval_only:
        result = run_training(
            exp_name, exp_cfg,
            device=args.device,
            output_dir=args.output_dir,
        )
        logger.info(
            "Training done: time=%.1fs  final_loss=%s",
            result["train_time_sec"], result["final_loss"],
        )

    # ── Evaluation (ROUGE-L on Dolly test split) ────────────────
    if not args.skip_eval:
        ckpt = run_dir / "student_final.pt"
        if not ckpt.exists():
            logger.error("Checkpoint not found: %s", ckpt)
            sys.exit(1)

        eval_res = run_eval(
            exp_name=exp_name,
            student_model=TRAINING_DEFAULTS["student_model"],
            checkpoint_path=str(ckpt),
            output_dir=args.output_dir,
            device=args.device,
            batch_size=args.eval_batch_size,
        )
        logger.info(
            "ROUGE-L: F=%.4f  P=%.4f  R=%.4f",
            eval_res.get("rouge_l_f", 0),
            eval_res.get("rouge_l_p", 0),
            eval_res.get("rouge_l_r", 0),
        )

    logger.info("Done: %s", exp_name)


if __name__ == "__main__":
    main()
