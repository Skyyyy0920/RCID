"""Experiment 1: Do existing distillation methods preserve teacher mechanisms?

Tests baseline methods (standard_kd, fitnets, informed_fitnets) and measures
whether distilled students maintain mechanistic consistency with the teacher.

Usage:
    python scripts/run_exp1.py --model_family qwen3 --task ioi --device cuda:0
    python scripts/run_exp1.py --model_family llama3 --task factual --seed 42 123 456
"""
from __future__ import annotations

import argparse, json, logging, time
from pathlib import Path

import torch

from common import (
    MODEL_CONFIGS, DEFAULT_TRAIN_CONFIG, build_dataset, prepare_alignment,
    save_student_checkpoint,
)
from rcid import set_all_seeds
from rcid.distillation.trainer import UnifiedTrainer
from rcid.eval.causal_consistency import CausalConsistencyEvaluator
from rcid.eval.task_accuracy import evaluate_task_accuracy
from rcid.models.student import load_student
from rcid.models.teacher import load_teacher

logger = logging.getLogger(__name__)
BASELINE_METHODS = ["standard_kd", "fitnets", "informed_fitnets"]


def run_single(
    method: str, task: str, seed: int, model_family: str,
    device: str, output_dir: Path, config: dict,
) -> dict:
    """Run a single (method, task, seed) experiment and return results dict."""
    set_all_seeds(seed)
    cfg = MODEL_CONFIGS[model_family]
    result_path = output_dir / f"{method}_{task}_seed{seed}.json"
    if result_path.exists():
        logger.info(f"Exists: {result_path}, skipping")
        with open(result_path) as f:
            return json.load(f)

    teacher, t_adp, tok = load_teacher(cfg["teacher"], device=device)
    student, s_adp, _ = load_student(cfg["student"], device=device)
    _, dataset = build_dataset(task, tok, seed=seed)
    logger.info(f"Dataset: {len(dataset)} samples, seq_len={dataset.seq_len}")

    teacher_acc = evaluate_task_accuracy(teacher, t_adp, dataset)
    logger.info(f"Teacher accuracy: {teacher_acc['accuracy']:.4f}")

    # Alignment: needed for fitnets/informed_fitnets training, and for eval
    if method in ("fitnets", "informed_fitnets"):
        cps, lmap, Ws = prepare_alignment(
            teacher, student, t_adp, s_adp, dataset, device,
            top_k=config.get("top_k", 20),
            diversity_ratio=config.get("diversity_ratio", 0.5),
            all_layer_W=(method == "fitnets"),  # FitNets needs W for every layer
        )
    else:
        cps, lmap, Ws = [], {}, {}

    # Train
    logger.info(f"Training: method={method}, seed={seed}")
    t0 = time.time()
    trainer = UnifiedTrainer(
        method=method, teacher=teacher, student=student,
        teacher_adapter=t_adp, student_adapter=s_adp, dataset=dataset,
        config=config, checkpoints=cps, layer_mapping=lmap, W_matrices=Ws,
    )
    history = trainer.train(epochs=config["epochs"], batch_size=config["batch_size"])
    train_time = time.time() - t0
    logger.info(f"Training done in {train_time:.1f}s")

    student.eval()
    student_acc = evaluate_task_accuracy(student, s_adp, dataset)
    logger.info(f"Student accuracy: {student_acc['accuracy']:.4f}")

    # For standard_kd, compute alignment artifacts now (needed for CC eval)
    if not cps:
        cps, lmap, Ws = prepare_alignment(
            teacher, student, t_adp, s_adp, dataset, device,
            top_k=config.get("top_k", 20),
        )

    logger.info("Evaluating causal consistency...")
    cc = CausalConsistencyEvaluator().evaluate(
        teacher, student, t_adp, s_adp, dataset, cps, lmap,
    )
    logger.info(f"Mean CC: {cc['mean_correlation']:.4f}")

    # Save student checkpoint
    ckpt_path = save_student_checkpoint(student, output_dir, method, task, seed)

    per_cp = {f"({k[0]},{k[1]})": v for k, v in cc.get("per_checkpoint", {}).items()}
    result = {
        "model_family": model_family, "method": method, "task": task, "seed": seed,
        "teacher_model": cfg["teacher"], "student_model": cfg["student"],
        "teacher_accuracy": teacher_acc, "student_accuracy": student_acc,
        "accuracy": student_acc["accuracy"],
        "causal_consistency": cc["mean_correlation"],
        "causal_consistency_detail": {"mean_correlation": cc["mean_correlation"],
                                      "per_checkpoint": per_cp},
        "checkpoint_path": ckpt_path,
        "training": {"final_loss": history["loss"][-1] if history["loss"] else None,
                     "loss_history": history["loss"],
                     "train_time_seconds": train_time},
        "config": config, "n_checkpoints": len(cps),
    }
    result_path.parent.mkdir(parents=True, exist_ok=True)
    with open(result_path, "w") as f:
        json.dump(result, f, indent=2)
    logger.info(f"Saved: {result_path}")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Exp1: Do existing distillation methods preserve teacher mechanisms?")
    parser.add_argument("--model_family", choices=["qwen3", "llama3"], default="qwen3")
    parser.add_argument("--task", choices=["ioi", "factual", "winogrande"], default="ioi")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, nargs="+", default=[42, 123, 456])
    parser.add_argument("--method", nargs="+", default=BASELINE_METHODS,
                        choices=BASELINE_METHODS)
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--config", type=str, default=None, help="YAML config path")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    config = dict(DEFAULT_TRAIN_CONFIG)
    if args.config:
        from omegaconf import OmegaConf
        file_cfg = OmegaConf.to_container(OmegaConf.load(args.config), resolve=True)
        if isinstance(file_cfg, dict):
            config.update(file_cfg.get("training", file_cfg))
    if args.epochs is not None:
        config["epochs"] = args.epochs
    if args.batch_size is not None:
        config["batch_size"] = args.batch_size

    root = Path(__file__).resolve().parent.parent
    out_dir = Path(args.output_dir) if args.output_dir else (
        root / "outputs" / "results" / "exp1" / args.model_family)
    out_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Exp1: family={args.model_family}, task={args.task}")
    logger.info(f"Methods: {args.method}, Seeds: {args.seed}, Output: {out_dir}")

    all_results = []
    for method in args.method:
        for seed in args.seed:
            logger.info(f"\n{'='*60}\n  {method} | {args.task} | seed={seed}\n{'='*60}")
            all_results.append(run_single(
                method, args.task, seed, args.model_family,
                args.device, out_dir, config))
            torch.cuda.empty_cache()

    # Summary table
    print(f"\n{'='*70}")
    print(f"  EXPERIMENT 1 SUMMARY: {args.model_family} / {args.task}")
    print(f"{'='*70}")
    print(f"{'Method':<20} {'Seed':<8} {'Accuracy':<12} {'CC (mean)':<12} {'Loss':<10}")
    print("-" * 70)
    for r in all_results:
        acc = r.get("accuracy", 0)
        cc_val = r.get("causal_consistency", 0)
        loss = r.get("training", {}).get("final_loss")
        loss_s = f"{loss:.4f}" if loss is not None else "N/A"
        print(f"{r['method']:<20} {r['seed']:<8} {acc:<12.4f} {cc_val:<12.4f} {loss_s:<10}")
    print(f"{'='*70}")

    summary_path = out_dir / f"exp1_summary_{args.task}.json"
    with open(summary_path, "w") as f:
        json.dump(all_results, f, indent=2)
    logger.info(f"Summary saved to {summary_path}")


if __name__ == "__main__":
    main()
