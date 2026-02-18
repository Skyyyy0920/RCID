"""Experiment 1: Do existing distillation methods preserve teacher mechanisms?

Tests baseline methods (standard_kd, fitnets, informed_fitnets) and measures
whether distilled students maintain mechanistic consistency with the teacher.

Usage:
    python scripts/run_exp1.py --model_family qwen3 --task ioi --device cuda:0
    python scripts/run_exp1.py --model_family llama3 --task factual --seed 42 123 456
"""
from __future__ import annotations

import argparse, json, logging, sys, time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from rcid import set_all_seeds
from rcid.alignment.cka import cka_matrix
from rcid.alignment.layer_matching import match_layers
from rcid.alignment.procrustes import compute_procrustes_matrices
from rcid.circuit.checkpoint_selection import select_checkpoints
from rcid.circuit.patching import extract_contrastive_differences
from rcid.data.factual_probing import FactualProbingDataset
from rcid.data.ioi import IOIDataset
from rcid.data.winogrande import WinoGrandeDataset
from rcid.distillation.trainer import UnifiedTrainer
from rcid.eval.causal_consistency import CausalConsistencyEvaluator
from rcid.eval.task_accuracy import evaluate_task_accuracy
from rcid.models.student import load_student
from rcid.models.teacher import load_teacher

logger = logging.getLogger(__name__)

MODEL_CONFIGS = {
    "qwen3": {"teacher": "Qwen/Qwen3-8B", "student": "Qwen/Qwen3-0.6B"},
    "llama3": {"teacher": "meta-llama/Llama-3.1-8B", "student": "meta-llama/Llama-3.2-1B"},
}
BASELINE_METHODS = ["standard_kd", "fitnets", "informed_fitnets"]
DEFAULT_CONFIG = {
    "epochs": 20, "batch_size": 16, "lr": 5e-5, "temperature": 2.0,
    "weight_decay": 0.01, "lambda_kl": 1.0, "lambda_rcid": 1.0,
    "grad_clip": 1.0, "fp16": True,
}


def build_dataset(task: str, tokenizer, seed: int):
    """Build contrastive dataset; returns (wrapper, ContrastiveDataset)."""
    builders = {
        "ioi": lambda: IOIDataset(tokenizer=tokenizer, n_samples=500, seed=seed),
        "factual": lambda: FactualProbingDataset(tokenizer=tokenizer, seed=seed),
        "winogrande": lambda: WinoGrandeDataset(tokenizer=tokenizer, seed=seed),
    }
    if task not in builders:
        raise ValueError(f"Unknown task: {task}")
    wrapper = builders[task]()
    return wrapper, wrapper.dataset


def prepare_alignment(teacher, student, t_adapter, s_adapter, dataset, device):
    """Extract diffs, select checkpoints, compute CKA mapping + Procrustes W."""
    t_layers = list(range(t_adapter.get_num_layers(teacher)))
    s_layers = list(range(s_adapter.get_num_layers(student)))
    clean, corrupt = dataset.clean_ids.to(device), dataset.corrupt_ids.to(device)

    logger.info("Extracting teacher contrastive differences...")
    t_diffs = extract_contrastive_differences(teacher, t_adapter, clean, corrupt, t_layers)
    logger.info("Selecting causal checkpoints...")
    checkpoints = select_checkpoints(t_diffs, dataset, top_k=20, diversity_ratio=0.5)
    logger.info(f"Selected {len(checkpoints)} checkpoints")

    logger.info("Extracting student contrastive differences...")
    s_diffs = extract_contrastive_differences(student, s_adapter, clean, corrupt, s_layers)

    flatten = lambda reps: {l: v.reshape(-1, v.shape[-1]) for l, v in reps.items()}
    logger.info("Computing CKA matrix and layer mapping...")
    cka_scores = cka_matrix(flatten(t_diffs), flatten(s_diffs))
    layer_mapping = match_layers(cka_scores=cka_scores, strategy="greedy")

    cp_t = list({cp[0] for cp in checkpoints})
    cp_s = list({layer_mapping[tl] for tl in cp_t if tl in layer_mapping})
    t_flat = {l: t_diffs[l].reshape(-1, t_diffs[l].shape[-1]) for l in cp_t if l in t_diffs}
    s_flat = {l: s_diffs[l].reshape(-1, s_diffs[l].shape[-1]) for l in cp_s if l in s_diffs}
    logger.info("Computing Procrustes alignment matrices...")
    W_matrices = compute_procrustes_matrices(t_flat, s_flat, layer_mapping)
    return checkpoints, layer_mapping, W_matrices


def run_single(method, task, seed, model_family, device, output_dir, config):
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
    _, dataset = build_dataset(task, tok, seed)
    logger.info(f"Dataset: {len(dataset)} samples, seq_len={dataset.seq_len}")

    teacher_acc = evaluate_task_accuracy(teacher, t_adp, dataset)
    logger.info(f"Teacher accuracy: {teacher_acc['accuracy']:.4f}")

    # Alignment: needed for fitnets/informed_fitnets training, and for eval
    if method in ("fitnets", "informed_fitnets"):
        cps, lmap, Ws = prepare_alignment(teacher, student, t_adp, s_adp, dataset, device)
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
        cps, lmap, Ws = prepare_alignment(teacher, student, t_adp, s_adp, dataset, device)

    logger.info("Evaluating causal consistency...")
    cc = CausalConsistencyEvaluator().evaluate(
        teacher, student, t_adp, s_adp, dataset, cps, lmap,
    )
    logger.info(f"Mean CC: {cc['mean_correlation']:.4f}")

    per_cp = {f"({k[0]},{k[1]})": v for k, v in cc.get("per_checkpoint", {}).items()}
    result = {
        "model_family": model_family, "method": method, "task": task, "seed": seed,
        "teacher_model": cfg["teacher"], "student_model": cfg["student"],
        "teacher_accuracy": teacher_acc, "student_accuracy": student_acc,
        "causal_consistency": {"mean_correlation": cc["mean_correlation"],
                               "per_checkpoint": per_cp},
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

    config = dict(DEFAULT_CONFIG)
    if args.config:
        from omegaconf import OmegaConf
        file_cfg = OmegaConf.to_container(OmegaConf.load(args.config), resolve=True)
        if isinstance(file_cfg, dict):
            config.update(file_cfg)
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

    # Summary table
    print(f"\n{'='*70}")
    print(f"  EXPERIMENT 1 SUMMARY: {args.model_family} / {args.task}")
    print(f"{'='*70}")
    print(f"{'Method':<20} {'Seed':<8} {'Accuracy':<12} {'CC (mean)':<12} {'Loss':<10}")
    print("-" * 70)
    for r in all_results:
        acc = r["student_accuracy"]["accuracy"]
        cc = r["causal_consistency"]["mean_correlation"]
        loss = r["training"]["final_loss"]
        loss_s = f"{loss:.4f}" if loss is not None else "N/A"
        print(f"{r['method']:<20} {r['seed']:<8} {acc:<12.4f} {cc:<12.4f} {loss_s:<10}")
    print(f"{'='*70}")

    summary_path = out_dir / f"exp1_summary_{args.task}.json"
    with open(summary_path, "w") as f:
        json.dump(all_results, f, indent=2)
    logger.info(f"Summary saved to {summary_path}")


if __name__ == "__main__":
    main()
