"""Experiment 2: RCID vs baselines — does RCID improve mechanism transfer?

Usage:
    python scripts/run_exp2.py --model_family qwen3 --task ioi --device cuda:0
    python scripts/run_exp2.py --model_family llama3 --task factual --seed 42 123
    python scripts/run_exp2.py --model_family qwen3 --task winogrande --skip_existing
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
from rcid.data.winogrande import WinoGrandeDataset
from rcid.circuit.patching import extract_contrastive_differences
from rcid.circuit.checkpoint_selection import select_checkpoints
from rcid.alignment.cka import cka_matrix
from rcid.alignment.layer_matching import match_layers
from rcid.alignment.procrustes import compute_procrustes_matrices
from rcid.distillation.trainer import UnifiedTrainer
from rcid.eval.causal_consistency import CausalConsistencyEvaluator
from rcid.eval.task_accuracy import evaluate_task_accuracy
from rcid.eval.perplexity import evaluate_perplexity

logger = logging.getLogger(__name__)

MODEL_CONFIGS = {
    "qwen3": {"teacher": "Qwen/Qwen3-8B", "student": "Qwen/Qwen3-0.6B"},
    "llama3": {"teacher": "meta-llama/Llama-3.1-8B", "student": "meta-llama/Llama-3.2-1B"},
}
METHODS = ["standard_kd", "fitnets", "informed_fitnets", "rcid"]
DEFAULT_SEEDS = [42, 123, 456]
TRAIN_CONFIG = dict(
    epochs=20, batch_size=16, lr=5e-5, weight_decay=0.01,
    lambda_kl=1.0, lambda_rcid=1.0, temperature=2.0, grad_clip=1.0, fp16=True,
)
# Placeholder texts for perplexity; real evaluation would use WikiText-2.
PPL_TEXTS = [
    "The quick brown fox jumps over the lazy dog.",
    "Natural language processing has made remarkable progress in recent years.",
    "Transformers use self-attention mechanisms to model long-range dependencies.",
    "Knowledge distillation transfers information from a large model to a small one.",
    "The residual stream carries information across transformer layers.",
    "Causal interventions reveal the internal mechanisms of neural networks.",
    "Contrastive learning captures the difference between similar inputs.",
    "Mechanistic interpretability aims to understand how models compute.",
    "The alignment of representations across models requires careful calibration.",
    "Evaluation of distilled models should go beyond task accuracy alone.",
]


def build_dataset(task: str, tokenizer, seed: int):
    """Build contrastive dataset for *task* and return inner ContrastiveDataset."""
    if task == "ioi":
        return IOIDataset(tokenizer=tokenizer, n_samples=200, seed=seed).dataset
    if task == "factual":
        return FactualProbingDataset(tokenizer=tokenizer, seed=seed).dataset
    if task == "winogrande":
        return WinoGrandeDataset(tokenizer=tokenizer, seed=seed).dataset
    raise ValueError(f"Unknown task: {task}")


def _out_path(base: str, family: str, method: str, task: str, seed: int) -> Path:
    return Path(base) / family / f"{method}_{task}_seed{seed}.json"


def _flatten(per_cp: dict) -> dict:
    return {f"L{k[0]}_P{k[1]}": v for k, v in per_cp.items()}


def run_single(method: str, task: str, seed: int, model_family: str,
               device: str, out_dir: str, config: dict) -> dict:
    """Full pipeline: load -> align -> train -> evaluate for one run."""
    set_all_seeds(seed)
    cfg = MODEL_CONFIGS[model_family]

    # 1. Load models
    logger.info("Loading teacher=%s  student=%s", cfg["teacher"], cfg["student"])
    teacher, t_adp, tokenizer = load_teacher(cfg["teacher"], device=device)
    student, s_adp, _ = load_student(cfg["student"], device=device)

    # 2. Dataset
    dataset = build_dataset(task, tokenizer, seed)
    logger.info("Dataset: %s  n=%d  seq_len=%d", task, len(dataset), dataset.seq_len)
    dev_clean = dataset.clean_ids.to(device)   # (N, seq)
    dev_corrupt = dataset.corrupt_ids.to(device)

    # 3. Teacher contrastive diffs + checkpoint selection
    n_tl = t_adp.get_num_layers(teacher)
    t_diffs = extract_contrastive_differences(
        teacher, t_adp, dev_clean, dev_corrupt, layers=list(range(n_tl)),
    )
    checkpoints = select_checkpoints(t_diffs, dataset, top_k=20)
    logger.info("Checkpoints: %d selected", len(checkpoints))

    # 4. CKA -> layer mapping -> Procrustes
    n_sl = s_adp.get_num_layers(student)
    s_diffs = extract_contrastive_differences(
        student, s_adp, dev_clean, dev_corrupt, layers=list(range(n_sl)),
    )
    t_reps = {l: t_diffs[l].mean(dim=1).cpu() for l in range(n_tl)}  # (N, d_T)
    s_reps = {l: s_diffs[l].mean(dim=1).cpu() for l in range(n_sl)}  # (N, d_S)
    cka = cka_matrix(t_reps, s_reps)                                  # (n_tl, n_sl)
    layer_map = match_layers(cka_scores=cka, strategy="greedy")

    cp_tl = sorted({c[0] for c in checkpoints})
    cp_sl = sorted({layer_map[t] for t in cp_tl})
    t_flat = {l: t_diffs[l].mean(dim=1).cpu() for l in cp_tl}
    s_flat = {l: s_diffs[l].mean(dim=1).cpu() for l in cp_sl}
    W_mats = compute_procrustes_matrices(t_flat, s_flat, layer_map)
    W_mats = {k: v.to(device) for k, v in W_mats.items()}

    # 5. Train
    logger.info("Training method=%s epochs=%d ...", method, config["epochs"])
    t0 = time.time()
    trainer = UnifiedTrainer(
        method=method, teacher=teacher, student=student,
        teacher_adapter=t_adp, student_adapter=s_adp,
        dataset=dataset, config=config,
        checkpoints=checkpoints, layer_mapping=layer_map, W_matrices=W_mats,
    )
    history = trainer.train(epochs=config["epochs"], batch_size=config["batch_size"])
    train_secs = time.time() - t0

    # 6. Evaluate
    student.eval()
    acc = evaluate_task_accuracy(student, s_adp, dataset)
    cc = CausalConsistencyEvaluator().evaluate(
        teacher, student, t_adp, s_adp, dataset, checkpoints, layer_map,
    )
    ppl = evaluate_perplexity(student, tokenizer, PPL_TEXTS)

    # 7. Save
    result = {
        "method": method, "task": task, "seed": seed,
        "model_family": model_family,
        "teacher": cfg["teacher"], "student": cfg["student"],
        "n_samples": len(dataset), "n_checkpoints": len(checkpoints),
        "accuracy": acc["accuracy"],
        "logit_diff_mean": acc["logit_diff_mean"],
        "logit_diff_std": acc["logit_diff_std"],
        "causal_consistency_mean": cc["mean_correlation"],
        "causal_consistency_per_cp": _flatten(cc["per_checkpoint"]),
        "perplexity": ppl,
        "train_time_sec": round(train_secs, 1),
        "final_loss": history["loss"][-1] if history["loss"] else None,
        "config": config,
    }
    out_file = _out_path(out_dir, model_family, method, task, seed)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with open(out_file, "w") as f:
        json.dump(result, f, indent=2)
    logger.info("Saved %s", out_file)
    return result


def log_factor_decomposition(results: dict[str, dict]) -> None:
    """Log ablation: position selection vs contrastive-diff matching."""
    def _cc(m: str) -> float | None:
        r = results.get(m)
        return r["causal_consistency_mean"] if r else None

    logger.info("=" * 60)
    logger.info("FACTOR DECOMPOSITION")
    logger.info("=" * 60)
    for m in METHODS:
        r = results.get(m)
        if r:
            logger.info("  %-20s  acc=%.3f  CC=%.4f  ppl=%.2f",
                        m, r["accuracy"], r["causal_consistency_mean"], r["perplexity"])
    logger.info("-" * 60)
    fn, inf, rc, sk = _cc("fitnets"), _cc("informed_fitnets"), _cc("rcid"), _cc("standard_kd")
    if fn is not None and inf is not None:
        logger.info("  FitNets -> InformedFitNets (position):      CC delta = %+.4f", inf - fn)
    if inf is not None and rc is not None:
        logger.info("  InformedFitNets -> RCID (contrastive diff): CC delta = %+.4f", rc - inf)
    if sk is not None and rc is not None:
        logger.info("  StandardKD -> RCID (total):                 CC delta = %+.4f", rc - sk)
    logger.info("=" * 60)


def main() -> None:
    ap = argparse.ArgumentParser(description="Exp2: RCID vs baselines")
    ap.add_argument("--model_family", choices=["qwen3", "llama3"], default="qwen3")
    ap.add_argument("--task", choices=["ioi", "factual", "winogrande"], default="ioi")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--seed", type=int, nargs="+", default=DEFAULT_SEEDS)
    ap.add_argument("--methods", nargs="+", default=METHODS, choices=METHODS)
    ap.add_argument("--output_dir", default="outputs/results/exp2")
    ap.add_argument("--skip_existing", action="store_true",
                    help="Skip runs whose output JSON already exists")
    ap.add_argument("--config", type=str, default=None,
                    help="Path to YAML config (reserved for future use)")
    args = ap.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    logger.info("Exp2: family=%s task=%s seeds=%s", args.model_family, args.task, args.seed)
    config = dict(TRAIN_CONFIG)

    for seed in args.seed:
        seed_results: dict[str, dict] = {}
        for method in args.methods:
            out_file = _out_path(args.output_dir, args.model_family, method, args.task, seed)
            if args.skip_existing and out_file.exists():
                logger.info("SKIP (exists): %s", out_file)
                try:
                    with open(out_file) as f:
                        seed_results[method] = json.load(f)
                except Exception:
                    pass
                continue

            logger.info("=== RUN: method=%s task=%s seed=%d family=%s ===",
                        method, args.task, seed, args.model_family)
            seed_results[method] = run_single(
                method, args.task, seed, args.model_family,
                args.device, args.output_dir, config,
            )
            torch.cuda.empty_cache()

        if len(seed_results) > 1:
            log_factor_decomposition(seed_results)

    logger.info("Experiment 2 complete.")


if __name__ == "__main__":
    main()
