"""Phase 2: AKL baseline + KL-Ratio adaptive distillation experiments.

Compares five methods on Alpaca-52K → Qwen3-0.6B distillation:

  1. forward_kl        — standard forward KL (baseline)
  2. jeffreys          — fixed Jeffreys divergence (0.5 FKL + 0.5 RKL)
  3. akl_mu0.5         — AKL (Wu et al., COLING 2025)
  4. klr_token          — KL-Ratio per-token adaptive (ours)
  5. klr_batch_ema      — KL-Ratio batch-level + EMA (ours)

Usage::

    # Run all experiments
    python scripts/run_phase2_experiments.py --device cuda:0

    # Run specific experiments
    python scripts/run_phase2_experiments.py --experiments akl_mu0.5,klr_token

    # Skip evaluation (training only)
    python scripts/run_phase2_experiments.py --experiments klr_token --skip_eval

Output::

    outputs/phase2/{experiment_name}/
        ├── student_final.pt
        ├── training_stats.json
        ├── eval_results.json
        └── config.json
"""
from __future__ import annotations

import argparse
import json
import logging
import subprocess
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

EVAL_BENCHMARKS = ["gsm8k", "mmlu", "arc_challenge"]

# ------------------------------------------------------------------
# Experiment definitions
# ------------------------------------------------------------------

EXPERIMENTS: dict[str, dict[str, Any]] = {
    "forward_kl": {
        "method": "standard_kd",
        "overrides": {},
    },
    "jeffreys": {
        "method": "standard_kd_padd",
        "overrides": {
            "padd_tau": 1000.0,       # tau → ∞ ⇒ alpha → 0.5 ⇒ Jeffreys
            "padd_alpha_min": 0.5,
            "padd_alpha_max": 0.5,
        },
    },
    "akl_mu0.5": {
        "method": "standard_kd_akl",
        "overrides": {"akl_mu": 0.5},
    },
    "klr_token": {
        "method": "standard_kd_klr",
        "overrides": {"klr_granularity": "token"},
    },
    "klr_batch_ema": {
        "method": "standard_kd_klr",
        "overrides": {"klr_granularity": "batch", "klr_beta": 0.99},
    },
}


# ------------------------------------------------------------------
# Training
# ------------------------------------------------------------------

def run_single(
    exp_name: str,
    exp_cfg: dict[str, Any],
    *,
    model_family: str,
    seed: int,
    device: str,
    data_source: str,
    max_train_samples: int | None,
    output_dir: str,
    epochs: int = 3,
    batch_size: int = 8,
    gradient_accumulation: int = 4,
    lr: float = 2e-5,
    max_seq_len: int = 512,
    temperature: float = 2.0,
    fp16: bool = True,
) -> dict[str, Any]:
    """Train one experiment configuration."""
    set_all_seeds(seed)
    mcfg = MODEL_CONFIGS[model_family]

    method: str = exp_cfg["method"]
    overrides: dict[str, Any] = exp_cfg.get("overrides", {})

    run_dir = Path(output_dir) / exp_name
    run_dir.mkdir(parents=True, exist_ok=True)

    # ── Save config.json ────────────────────────────────────────────
    run_config: dict[str, Any] = {
        "experiment": exp_name,
        "method": method,
        "overrides": overrides,
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

    logger.info("Loading data: %s (max=%s)", data_source, max_train_samples)
    main_ds = InstructionDataset(
        dataset_name=data_source,
        tokenizer=tokenizer,
        max_seq_len=max_seq_len,
        max_samples=max_train_samples,
    )
    logger.info("Dataset: %d samples", len(main_ds))

    # ── Build trainer config ────────────────────────────────────────
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
        "lambda_rcid": 0.0,
        "save_every_n_epochs": 1,
        "use_wandb": False,
        "log_every": 50,
    }
    trainer_cfg.update(overrides)

    # ── Train ───────────────────────────────────────────────────────
    from rcid.distillation.scalable_trainer import ScalableDistillationTrainer

    trainer = ScalableDistillationTrainer(
        teacher=teacher, student=student,
        teacher_adapter=t_adp, student_adapter=s_adp,
        tokenizer=tokenizer,
        main_dataset=main_ds,
        contrastive_dataset=None,
        config=trainer_cfg,
    )

    logger.info(
        "Training: exp=%s  method=%s  epochs=%d  batch=%dx%d  lr=%.2e",
        exp_name, method, epochs, batch_size, gradient_accumulation, lr,
    )
    t0 = time.time()
    history = trainer.train(save_dir=str(run_dir))
    train_secs = time.time() - t0
    logger.info("Training complete in %.1f s", train_secs)

    eval_metrics = trainer.evaluate()

    # ── Save checkpoint ─────────────────────────────────────────────
    final_ckpt = run_dir / "student_final.pt"
    torch.save(student.state_dict(), final_ckpt)
    logger.info("Final checkpoint: %s", final_ckpt)

    # ── Save training_stats.json ────────────────────────────────────
    training_stats: dict[str, Any] = {
        "experiment": exp_name,
        "method": method,
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
# Benchmark evaluation
# ------------------------------------------------------------------

def run_eval(
    model_name: str,
    model_path: str,
    output_dir: str,
    benchmarks: list[str],
    device: str,
    batch_size: int = 4,
) -> dict[str, Any]:
    """Run lm-eval and save results to output_dir/eval_results.json."""
    script = Path(__file__).resolve().parent / "eval_benchmarks.py"
    out_json = str(Path(output_dir) / "eval_results.json")
    cmd = [
        sys.executable, str(script),
        "--model_name", model_name,
        "--model_path", model_path,
        "--benchmarks", ",".join(benchmarks),
        "--device", device,
        "--batch_size", str(batch_size),
        "--output_path", out_json,
    ]
    logger.info("Running lm-eval: %s", " ".join(cmd))
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=7200)
    if proc.returncode != 0:
        logger.error("lm-eval failed:\n%s", proc.stderr[-500:])
        return {"error": proc.stderr[-300:]}

    if Path(out_json).exists():
        with open(out_json, "r", encoding="utf-8") as f:
            return json.load(f)
    return {"error": "eval_results.json not found"}


# ------------------------------------------------------------------
# CLI
# ------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(description="Phase 2 experiments")
    ap.add_argument(
        "--experiments", default="all",
        help="Comma-separated experiment names, or 'all' (default: all)",
    )
    ap.add_argument("--model_family", default="qwen3", choices=["qwen3", "llama3"])
    ap.add_argument("--data_source", default="tatsu-lab/alpaca")
    ap.add_argument("--max_train_samples", type=int, default=52000)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--gradient_accumulation", type=int, default=4)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--max_seq_len", type=int, default=512)
    ap.add_argument("--temperature", type=float, default=2.0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--output_dir", default="outputs/phase2")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--fp16", action="store_true", default=True)
    ap.add_argument("--no_fp16", dest="fp16", action="store_false")
    ap.add_argument("--skip_eval", action="store_true")
    ap.add_argument("--eval_batch_size", type=int, default=4)
    args = ap.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Resolve experiment list
    if args.experiments == "all":
        exp_names = list(EXPERIMENTS.keys())
    else:
        exp_names = [e.strip() for e in args.experiments.split(",")]
        for e in exp_names:
            if e not in EXPERIMENTS:
                logger.error("Unknown experiment: %s. Available: %s", e,
                             list(EXPERIMENTS.keys()))
                sys.exit(1)

    logger.info("=== Phase 2: %d experiments ===", len(exp_names))
    mcfg = MODEL_CONFIGS[args.model_family]

    for exp_name in exp_names:
        exp_cfg = EXPERIMENTS[exp_name]
        logger.info("")
        logger.info("=" * 60)
        logger.info("  Experiment: %s", exp_name)
        logger.info("=" * 60)

        result = run_single(
            exp_name=exp_name,
            exp_cfg=exp_cfg,
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
            temperature=args.temperature,
            fp16=args.fp16,
        )
        logger.info(
            "  Training done: time=%.1fs  final_loss=%s",
            result["train_time_sec"], result["final_loss"],
        )

        if not args.skip_eval:
            run_dir = Path(args.output_dir) / exp_name
            eval_res = run_eval(
                model_name=mcfg["student"],
                model_path=result["checkpoint_path"],
                output_dir=str(run_dir),
                benchmarks=EVAL_BENCHMARKS,
                device=args.device,
                batch_size=args.eval_batch_size,
            )
            logger.info("  Eval: %s", json.dumps(
                {k: v for k, v in eval_res.get("results", {}).items()},
                indent=2,
            ))

    logger.info("")
    logger.info("=== Phase 2 complete. Results in: %s ===", args.output_dir)


if __name__ == "__main__":
    main()
