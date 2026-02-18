"""Experiment 5: Cross-architecture generalization (LLaMA 3).

StandardKD vs RCID on LLaMA-3.1-8B -> LLaMA-3.2-1B.
2 methods x 2 tasks x 3 seeds = 12 runs.

Usage:
    python scripts/run_exp5_cross_arch.py --task ioi factual --device cuda:0
    python scripts/run_exp5_cross_arch.py --task ioi --seed 42 123 --skip_existing
"""
from __future__ import annotations

import argparse, json, logging, sys, time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from rcid import set_all_seeds
from rcid.models.teacher import load_teacher
from rcid.models.student import load_student
from rcid.data.ioi import IOIDataset
from rcid.data.factual_probing import FactualProbingDataset
from rcid.circuit.patching import extract_contrastive_differences
from rcid.circuit.checkpoint_selection import select_checkpoints
from rcid.alignment.cka import cka_matrix
from rcid.alignment.layer_matching import match_layers
from rcid.alignment.procrustes import compute_procrustes_matrices
from rcid.distillation.trainer import UnifiedTrainer
from rcid.eval.causal_consistency import CausalConsistencyEvaluator
from rcid.eval.task_accuracy import evaluate_task_accuracy

logger = logging.getLogger(__name__)

TEACHER_NAME = "meta-llama/Llama-3.1-8B"
STUDENT_NAME = "meta-llama/Llama-3.2-1B"
METHODS: list[str] = ["standard_kd", "rcid"]
TASKS: list[str] = ["ioi", "factual"]
DEFAULT_SEEDS: list[int] = [42, 123, 456]
TRAIN_CONFIG: dict = {
    "epochs": 20, "batch_size": 16, "lr": 5e-5, "weight_decay": 0.01,
    "lambda_kl": 1.0, "lambda_rcid": 1.0, "temperature": 2.0,
    "grad_clip": 1.0, "fp16": True,
}


def build_dataset(task: str, tokenizer, seed: int):
    """Return the inner ContrastiveDataset for *task*."""
    if task == "ioi":
        return IOIDataset(tokenizer=tokenizer, n_samples=500, seed=seed).dataset
    if task == "factual":
        return FactualProbingDataset(tokenizer=tokenizer, seed=seed).dataset
    raise ValueError(f"Unknown task: {task}")


def _flatten(per_checkpoint: dict) -> dict:
    return {f"L{k[0]}_P{k[1]}": v for k, v in per_checkpoint.items()}


def out_path(base: str, method: str, task: str, seed: int) -> Path:
    return Path(base) / f"{method}_{task}_seed{seed}.json"


def run_single(method: str, task: str, seed: int,
               device: str, out_dir: str, config: dict) -> dict:
    """Train + evaluate one (method, task, seed) combination on LLaMA 3."""
    set_all_seeds(seed)
    logger.info("Loading teacher=%s, student=%s", TEACHER_NAME, STUDENT_NAME)
    teacher, t_adp, tokenizer = load_teacher(TEACHER_NAME, device=device)
    student, s_adp, _ = load_student(STUDENT_NAME, device=device)

    dataset = build_dataset(task, tokenizer, seed)
    logger.info("Dataset %s: %d samples, seq_len=%d", task, len(dataset), dataset.seq_len)

    # Teacher contrastive diffs + checkpoint selection
    clean_dev = dataset.clean_ids.to(device)
    corrupt_dev = dataset.corrupt_ids.to(device)
    n_tl = t_adp.get_num_layers(teacher)
    t_diffs = extract_contrastive_differences(
        teacher, t_adp, clean_dev, corrupt_dev, layers=list(range(n_tl)),
    )
    checkpoints = select_checkpoints(t_diffs, dataset, top_k=20)
    logger.info("Selected %d checkpoints", len(checkpoints))

    # CKA -> layer mapping -> Procrustes
    n_sl = s_adp.get_num_layers(student)
    s_diffs = extract_contrastive_differences(
        student, s_adp, clean_dev, corrupt_dev, layers=list(range(n_sl)),
    )
    t_reps = {l: t_diffs[l].mean(dim=1).cpu() for l in range(n_tl)}  # (N, d_T)
    s_reps = {l: s_diffs[l].mean(dim=1).cpu() for l in range(n_sl)}  # (N, d_S)
    layer_mapping = match_layers(cka_scores=cka_matrix(t_reps, s_reps), strategy="greedy")

    cp_tl = sorted({cp[0] for cp in checkpoints})
    cp_sl = sorted({layer_mapping[tl] for tl in cp_tl})
    W_matrices = {k: v.to(device) for k, v in compute_procrustes_matrices(
        {l: t_diffs[l].mean(dim=1).cpu() for l in cp_tl},
        {l: s_diffs[l].mean(dim=1).cpu() for l in cp_sl},
        layer_mapping,
    ).items()}
    logger.info("Layer mapping: %s", layer_mapping)

    # Train
    logger.info("Training method=%s, epochs=%d ...", method, config["epochs"])
    t0 = time.time()
    trainer = UnifiedTrainer(
        method=method, teacher=teacher, student=student,
        teacher_adapter=t_adp, student_adapter=s_adp, dataset=dataset,
        config=config, checkpoints=checkpoints,
        layer_mapping=layer_mapping, W_matrices=W_matrices,
    )
    history = trainer.train(epochs=config["epochs"], batch_size=config["batch_size"])
    train_time = time.time() - t0

    # Evaluate
    student.eval()
    acc = evaluate_task_accuracy(student, s_adp, dataset)
    cc = CausalConsistencyEvaluator().evaluate(
        teacher, student, t_adp, s_adp, dataset, checkpoints, layer_mapping,
    )
    logger.info("acc=%.4f  CC=%.4f", acc["accuracy"], cc["mean_correlation"])

    result = {
        "method": method, "task": task, "seed": seed,
        "model_family": "llama3", "teacher": TEACHER_NAME, "student": STUDENT_NAME,
        "n_samples": len(dataset), "n_checkpoints": len(checkpoints),
        "accuracy": acc["accuracy"], "logit_diff_mean": acc["logit_diff_mean"],
        "logit_diff_std": acc["logit_diff_std"],
        "causal_consistency_mean": cc["mean_correlation"],
        "causal_consistency_per_cp": _flatten(cc["per_checkpoint"]),
        "train_time_sec": round(train_time, 1),
        "final_loss": history["loss"][-1] if history["loss"] else None,
        "config": config,
    }
    fp = out_path(out_dir, method, task, seed)
    fp.parent.mkdir(parents=True, exist_ok=True)
    with open(fp, "w") as f:
        json.dump(result, f, indent=2)
    logger.info("Saved %s", fp)
    return result


def log_summary(results: list[dict]) -> None:
    """Print cross-architecture comparison table."""
    logger.info("=" * 70)
    logger.info("CROSS-ARCHITECTURE SUMMARY (LLaMA 3)")
    logger.info("-" * 70)
    for r in sorted(results, key=lambda x: (x["task"], x["method"], x["seed"])):
        logger.info("%-14s %-8s seed=%d  acc=%.4f  CC=%.4f",
                     r["method"], r["task"], r["seed"],
                     r["accuracy"], r["causal_consistency_mean"])
    logger.info("=" * 70)


def main() -> None:
    parser = argparse.ArgumentParser(description="Exp5: LLaMA 3 cross-arch")
    parser.add_argument("--task", nargs="+", choices=TASKS, default=TASKS)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, nargs="+", default=DEFAULT_SEEDS)
    parser.add_argument("--output_dir", default="outputs/results/exp5_cross_arch")
    parser.add_argument("--skip_existing", action="store_true",
                        help="Skip runs whose output JSON already exists")
    parser.add_argument("--config", type=str, default=None,
                        help="Path to YAML config override (reserved)")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    logger.info("Exp5: LLaMA 3 | tasks=%s | seeds=%s", args.task, args.seed)

    config = dict(TRAIN_CONFIG)
    all_results: list[dict] = []

    for task in args.task:
        for seed in args.seed:
            for method in METHODS:
                fp = out_path(args.output_dir, method, task, seed)
                if args.skip_existing and fp.exists():
                    logger.info("SKIP (exists): %s", fp)
                    try:
                        with open(fp) as f:
                            all_results.append(json.load(f))
                    except Exception:
                        pass
                    continue
                logger.info("=== RUN: method=%s task=%s seed=%d ===", method, task, seed)
                all_results.append(run_single(method, task, seed,
                                              args.device, args.output_dir, config))
                torch.cuda.empty_cache()

    if all_results:
        log_summary(all_results)
    logger.info("Experiment 5 complete.")


if __name__ == "__main__":
    main()
