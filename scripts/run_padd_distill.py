"""Phase 1 PADD (Position-Adaptive Divergence Distillation) experiment.

Trains a student with standard KD or PADD on instruction data, then
evaluates on core benchmarks via lm-eval.

Experiment matrix
-----------------
  standard_kd              — baseline (pure forward KL)
  standard_kd_padd         — PADD with tau grid search
  pure_reverse_kl          — ablation: alpha fixed to 0 (all reverse KL)
  fixed_jsd                — ablation: alpha fixed to 0.5 (JSD)

Usage::

    # Single run
    python scripts/run_padd_distill.py --method standard_kd_padd --tau 1.0

    # Ablation: pure reverse KL
    python scripts/run_padd_distill.py --method pure_reverse_kl

    # Ablation: fixed JSD
    python scripts/run_padd_distill.py --method fixed_jsd

Output::

    outputs/padd/{model_family}/{method}/tau_{tau}/seed_{seed}/
        ├── student_final.pt
        ├── training_log.json
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

METHODS = ["standard_kd", "standard_kd_padd", "pure_reverse_kl", "fixed_jsd"]

EVAL_BENCHMARKS = ["mmlu", "gsm8k", "arc_challenge"]


# ------------------------------------------------------------------
# Training
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
    tau: float = 1.0,
    epochs: int = 3,
    batch_size: int = 8,
    gradient_accumulation: int = 4,
    lr: float = 2e-5,
    max_seq_len: int = 512,
    temperature: float = 2.0,
    fp16: bool = True,
) -> dict[str, Any]:
    """Full pipeline: load -> train -> save -> (optionally eval)."""
    set_all_seeds(seed)
    mcfg = MODEL_CONFIGS[model_family]

    # ── Output directory ────────────────────────────────────────────
    if method == "standard_kd":
        run_dir = Path(output_dir) / model_family / method / f"seed_{seed}"
    else:
        run_dir = (
            Path(output_dir) / model_family / method
            / f"tau_{tau}" / f"seed_{seed}"
        )
    run_dir.mkdir(parents=True, exist_ok=True)

    # ── Resolve PADD config ─────────────────────────────────────────
    # pure_reverse_kl: alpha clamped to [0, 0] → always reverse KL
    # fixed_jsd:       alpha clamped to [0.5, 0.5] → always JSD
    # standard_kd_padd: normal adaptive alpha
    padd_alpha_min = 0.1
    padd_alpha_max = 0.9
    trainer_method = method

    if method == "pure_reverse_kl":
        padd_alpha_min = 0.0
        padd_alpha_max = 0.0
        trainer_method = "standard_kd_padd"  # reuse PADD path
    elif method == "fixed_jsd":
        padd_alpha_min = 0.5
        padd_alpha_max = 0.5
        trainer_method = "standard_kd_padd"

    # ── Save config.json ────────────────────────────────────────────
    run_config: dict[str, Any] = {
        "method": method,
        "trainer_method": trainer_method,
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
        "padd_tau": tau,
        "padd_alpha_min": padd_alpha_min,
        "padd_alpha_max": padd_alpha_max,
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

    logger.info("Loading instruction data: %s (max=%s)", data_source, max_train_samples)
    main_ds = InstructionDataset(
        dataset_name=data_source,
        tokenizer=tokenizer,
        max_seq_len=max_seq_len,
        max_samples=max_train_samples,
    )
    logger.info("Instruction dataset: %d samples", len(main_ds))

    # ── Build trainer config ────────────────────────────────────────
    trainer_cfg: dict[str, Any] = {
        "method": trainer_method,
        "epochs": epochs,
        "batch_size": batch_size,
        "gradient_accumulation": gradient_accumulation,
        "lr": lr,
        "weight_decay": 0.01,
        "max_grad_norm": 1.0,
        "warmup_ratio": 0.03,
        "fp16": fp16,
        "temperature": temperature,
        "lambda_rcid": 0.0,  # no RCID in PADD experiments
        "save_every_n_epochs": 1,
        "use_wandb": False,
        "log_every": 50,
        # PADD-specific
        "padd_tau": tau,
        "padd_alpha_min": padd_alpha_min,
        "padd_alpha_max": padd_alpha_max,
    }

    # ── Create trainer and train ────────────────────────────────────
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
        "Training: method=%s  tau=%.2f  epochs=%d  batch=%dx%d  lr=%.2e",
        method, tau, epochs, batch_size, gradient_accumulation, lr,
    )
    t0 = time.time()
    history = trainer.train(save_dir=str(run_dir))
    train_secs = time.time() - t0
    logger.info("Training complete in %.1f s", train_secs)

    # ── Evaluate (KL on main data) ──────────────────────────────────
    eval_metrics = trainer.evaluate()

    # ── Save student_final.pt ───────────────────────────────────────
    final_ckpt = run_dir / "student_final.pt"
    torch.save(student.state_dict(), final_ckpt)
    logger.info("Final checkpoint: %s", final_ckpt)

    # ── Save training_log.json ──────────────────────────────────────
    training_log: dict[str, Any] = {
        "method": method,
        "model_family": model_family,
        "seed": seed,
        "tau": tau,
        "train_time_sec": round(train_secs, 1),
        "n_samples": len(main_ds),
        "history": history,
        "eval_kl_loss": eval_metrics.get("kl_loss"),
        "final_loss": history["loss"][-1] if history.get("loss") else None,
        "checkpoint_path": str(final_ckpt),
    }
    with open(run_dir / "training_log.json", "w", encoding="utf-8") as f:
        json.dump(training_log, f, indent=2)

    return training_log


# ------------------------------------------------------------------
# Benchmark evaluation
# ------------------------------------------------------------------

def run_eval(
    model_name: str,
    model_path: str,
    benchmarks: list[str],
    device: str,
    batch_size: int = 16,
) -> dict[str, Any]:
    """Run lm-eval on a saved checkpoint via the eval_benchmarks script."""
    script = Path(__file__).resolve().parent / "eval_benchmarks.py"
    bench_str = ",".join(benchmarks)
    cmd = [
        sys.executable, str(script),
        "--model_name", model_name,
        "--model_path", model_path,
        "--benchmarks", bench_str,
        "--device", device,
        "--batch_size", str(batch_size),
    ]
    logger.info("Running lm-eval: %s", " ".join(cmd))
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=7200)
    if proc.returncode != 0:
        logger.error("lm-eval failed:\n%s", proc.stderr[-500:])
        return {"error": proc.stderr[-300:]}

    # Parse the output JSON saved by eval_benchmarks.py
    results_path = Path(model_path).parent / "benchmark_results.json"
    if results_path.exists():
        with open(results_path, "r", encoding="utf-8") as f:
            return json.load(f)
    return {"error": "benchmark_results.json not found"}


# ------------------------------------------------------------------
# CLI
# ------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(description="PADD Phase 1 experiment")
    ap.add_argument("--method", choices=METHODS, default="standard_kd_padd")
    ap.add_argument("--tau", type=float, default=1.0, help="PADD tau")
    ap.add_argument("--model_family", choices=["qwen3", "llama3"], default="qwen3")
    ap.add_argument("--data_source", default="tatsu-lab/alpaca")
    ap.add_argument("--max_train_samples", type=int, default=52000)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--gradient_accumulation", type=int, default=4)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--max_seq_len", type=int, default=512)
    ap.add_argument("--temperature", type=float, default=2.0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--output_dir", default="outputs/padd")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--fp16", action="store_true", default=True)
    ap.add_argument("--no_fp16", dest="fp16", action="store_false")
    ap.add_argument(
        "--skip_eval", action="store_true",
        help="Skip lm-eval after training",
    )
    ap.add_argument("--eval_batch_size", type=int, default=16)
    args = ap.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    logger.info(
        "=== PADD Phase 1: method=%s  tau=%.2f  seed=%d ===",
        args.method, args.tau, args.seed,
    )

    result = run_single(
        method=args.method,
        model_family=args.model_family,
        seed=args.seed,
        device=args.device,
        data_source=args.data_source,
        max_train_samples=args.max_train_samples,
        output_dir=args.output_dir,
        tau=args.tau,
        epochs=args.epochs,
        batch_size=args.batch_size,
        gradient_accumulation=args.gradient_accumulation,
        lr=args.lr,
        max_seq_len=args.max_seq_len,
        temperature=args.temperature,
        fp16=args.fp16,
    )
    logger.info(
        "Training done. time=%.1fs  final_loss=%s",
        result["train_time_sec"], result["final_loss"],
    )

    # ── Benchmark evaluation ────────────────────────────────────────
    if not args.skip_eval:
        mcfg = MODEL_CONFIGS[args.model_family]
        eval_result = run_eval(
            model_name=mcfg["student"],
            model_path=result["checkpoint_path"],
            benchmarks=EVAL_BENCHMARKS,
            device=args.device,
            batch_size=args.eval_batch_size,
        )
        logger.info("Eval results: %s", json.dumps(eval_result, indent=2))


if __name__ == "__main__":
    main()
