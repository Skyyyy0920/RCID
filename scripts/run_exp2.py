"""Experiment 2: RCID vs baselines — does RCID improve mechanism transfer?

Usage:
    python scripts/run_exp2.py --model_family qwen3 --task ioi --device cuda:0
    python scripts/run_exp2.py --model_family llama3 --task factual --seed 42 123
    python scripts/run_exp2.py --model_family qwen3 --task winogrande --skip_existing
"""
from __future__ import annotations

import argparse, json, logging, time
from pathlib import Path

import torch

from common import (
    MODEL_CONFIGS, DEFAULT_TRAIN_CONFIG, build_dataset, prepare_alignment,
    save_student_checkpoint, load_perplexity_texts,
)
from rcid import set_all_seeds
from rcid.distillation.trainer import UnifiedTrainer
from rcid.eval.causal_consistency import CausalConsistencyEvaluator
from rcid.eval.task_accuracy import evaluate_task_accuracy
from rcid.eval.perplexity import evaluate_perplexity
from rcid.models.student import load_student
from rcid.models.teacher import load_teacher

logger = logging.getLogger(__name__)
METHODS = ["standard_kd", "fitnets", "informed_fitnets", "rcid"]


def _out_path(base: str, family: str, method: str, task: str, seed: int) -> Path:
    return Path(base) / family / f"{method}_{task}_seed{seed}.json"


def _flatten(per_cp: dict) -> dict:
    return {f"L{k[0]}_P{k[1]}": v for k, v in per_cp.items()}


def run_single(
    method: str, task: str, seed: int, model_family: str,
    device: str, out_dir: str, config: dict,
) -> dict:
    """Full pipeline: load -> align -> train -> evaluate for one run."""
    set_all_seeds(seed)
    cfg = MODEL_CONFIGS[model_family]

    teacher, t_adp, tokenizer = load_teacher(cfg["teacher"], device=device)
    student, s_adp, _ = load_student(cfg["student"], device=device)

    _, dataset = build_dataset(task, tokenizer, seed=seed)
    logger.info("Dataset: %s  n=%d  seq_len=%d", task, len(dataset), dataset.seq_len)

    # Unified alignment (uses flatten for CKA, not mean-pool)
    cps, layer_map, W_mats = prepare_alignment(
        teacher, student, t_adp, s_adp, dataset, device,
        top_k=config.get("top_k", 20),
        diversity_ratio=config.get("diversity_ratio", 0.5),
        all_layer_W=(method == "fitnets"),
    )
    W_mats = {k: v.to(device) for k, v in W_mats.items()}

    # Train
    logger.info("Training method=%s epochs=%d ...", method, config["epochs"])
    t0 = time.time()
    trainer = UnifiedTrainer(
        method=method, teacher=teacher, student=student,
        teacher_adapter=t_adp, student_adapter=s_adp,
        dataset=dataset, config=config,
        checkpoints=cps, layer_mapping=layer_map, W_matrices=W_mats,
    )
    history = trainer.train(epochs=config["epochs"], batch_size=config["batch_size"])
    train_secs = time.time() - t0

    # Evaluate
    student.eval()
    acc = evaluate_task_accuracy(student, s_adp, dataset)
    cc = CausalConsistencyEvaluator().evaluate(
        teacher, student, t_adp, s_adp, dataset, cps, layer_map,
    )
    ppl_texts = load_perplexity_texts()
    ppl = evaluate_perplexity(student, tokenizer, ppl_texts)

    # Save checkpoint
    out_path_dir = Path(out_dir) / model_family
    ckpt_path = save_student_checkpoint(student, out_path_dir, method, task, seed)

    result = {
        "method": method, "task": task, "seed": seed,
        "model_family": model_family,
        "teacher": cfg["teacher"], "student": cfg["student"],
        "n_samples": len(dataset), "n_checkpoints": len(cps),
        "accuracy": acc["accuracy"],
        "logit_diff_mean": acc["logit_diff_mean"],
        "logit_diff_std": acc["logit_diff_std"],
        "causal_consistency": cc["mean_correlation"],
        "causal_consistency_per_cp": _flatten(cc["per_checkpoint"]),
        "perplexity": ppl,
        "checkpoint_path": ckpt_path,
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
        return r["causal_consistency"] if r else None

    logger.info("=" * 60)
    logger.info("FACTOR DECOMPOSITION")
    for m in METHODS:
        r = results.get(m)
        if r:
            logger.info("  %-20s  acc=%.3f  CC=%.4f  ppl=%.2f",
                        m, r["accuracy"], r["causal_consistency"], r["perplexity"])
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
    ap.add_argument("--seed", type=int, nargs="+", default=[42, 123, 456])
    ap.add_argument("--methods", nargs="+", default=METHODS, choices=METHODS)
    ap.add_argument("--output_dir", default="outputs/results/exp2")
    ap.add_argument("--skip_existing", action="store_true")
    ap.add_argument("--config", type=str, default=None, help="Path to YAML config")
    args = ap.parse_args()

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
