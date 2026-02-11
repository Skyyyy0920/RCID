"""Experiment 2: Distillation Method Comparison (Section 4.2).

Trains 4 methods (StandardKD, FitNets, PrakashCKA, RCID) x 3 seeds.
Evaluates IOI accuracy, causal consistency, and WikiText perplexity.

Usage: python scripts/run_exp2.py [--config configs/exp2_distillation_comparison.yaml]
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import sys
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from omegaconf import OmegaConf
from torch import nn

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from rcid.alignment.layer_matching import find_layer_mapping_from_models
from rcid.alignment.procrustes import procrustes_align
from rcid.circuit.checkpoint_selection import (
    checkpoints_to_tuples, collect_key_positions_ioi, select_checkpoints,
)
from rcid.circuit.patching import extract_causal_imprints
from rcid.data.ioi import IOIDataset
from rcid.distillation.baselines import FitNetsLoss, PrakashCKALoss, StandardKDLoss
from rcid.distillation.rcid_loss import RCIDLoss
from rcid.distillation.trainer import DistillationTrainer, TrainConfig
from rcid.eval import causal_consistency, perplexity, task_accuracy
from rcid.models.student import StudentModel
from rcid.models.teacher import TeacherModel

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Exp2: Distillation Comparison")
    p.add_argument("--config", default="configs/exp2_distillation_comparison.yaml")
    return p.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def save_json(data: dict, path: Path) -> None:
    with open(path, "w") as f:
        json.dump(data, f, indent=2, default=str)
    logger.info("Saved: %s", path)


def build_losses(
    method_cfg: Any, cfg: Any, teacher: TeacherModel, student: StudentModel,
    clean_ids: torch.Tensor, corrupt_ids: torch.Tensor,
    checkpoints: list[tuple[int, int]],
    teacher_imprints: dict[tuple[int, int], torch.Tensor],
    shared_mapping: dict[int, int],
) -> tuple[StandardKDLoss | None, nn.Module | None,
           dict[tuple[int, int], torch.Tensor] | None]:
    """Build KD + auxiliary loss for a method. Returns (kd, aux, imprints)."""
    kd = StandardKDLoss(temperature=cfg.training.temperature)
    aux_type = method_cfg.aux_loss
    d_T, d_S = cfg.teacher.d_model, cfg.student.d_model

    if aux_type is None or str(aux_type) == "null":
        return kd, None, None
    if aux_type == "fitnets":
        pairs = FitNetsLoss.make_uniform_pairs(cfg.teacher.n_layers, cfg.student.n_layers)
        return kd, FitNetsLoss(d_T, d_S, pairs), None
    if aux_type == "prakash_cka":
        pairs = FitNetsLoss.make_uniform_pairs(cfg.teacher.n_layers, cfg.student.n_layers)
        return kd, PrakashCKALoss(d_T, d_S, pairs), None
    if aux_type == "rcid":
        # Procrustes: extract student imprints at mapped positions
        student.model.eval()
        s_cps = [(shared_mapping[l], t) for l, t in checkpoints]
        s_imprints = extract_causal_imprints(
            student.model, clean_ids, corrupt_ids, s_cps,
        )
        student.model.train()
        src = torch.cat([s_imprints[(shared_mapping[l], t)]
                         for l, t in checkpoints], dim=0)  # (N*C, d_S)
        tgt = torch.cat([teacher_imprints[(l, t)]
                         for l, t in checkpoints], dim=0)  # (N*C, d_T)
        W = procrustes_align(src, tgt)  # (d_T, d_S)
        aux = RCIDLoss(W=W, checkpoints=checkpoints, layer_mapping=shared_mapping)
        return kd, aux, teacher_imprints
    raise ValueError(f"Unknown aux_loss: {aux_type}")


def run_evaluation(
    student_model: nn.Module, teacher_model: nn.Module,
    eval_loader: Any, cfg: Any,
    checkpoints: list[tuple[int, int]], layer_mapping: dict[int, int],
) -> dict[str, float]:
    """Run IOI accuracy, causal consistency, perplexity."""
    student_model.eval()
    results: dict[str, float] = {}
    acc = task_accuracy.evaluate(student_model, eval_loader, task="ioi")
    results["ioi_accuracy"] = acc["accuracy"]
    cc = causal_consistency.evaluate(
        student_model, eval_loader, teacher_model=teacher_model,
        checkpoints=checkpoints, layer_mapping=layer_mapping,
    )
    results["causal_consistency"] = cc["consistency"]
    ppl = perplexity.evaluate(
        student_model, max_samples=cfg.eval.perplexity_max_samples,
    )
    results["perplexity"] = ppl["perplexity"]
    return results


def plot_training_curves(results: dict, output_dir: Path) -> None:
    """Training loss curves: 4 methods overlaid, one subplot per seed."""
    methods = list(results.keys())
    seeds = sorted(next(iter(results.values())).keys())
    fig, axes = plt.subplots(1, len(seeds), figsize=(5 * len(seeds), 4), squeeze=False)
    colors = {"StandardKD": "C0", "FitNets": "C1", "PrakashCKA": "C2", "RCID": "C3"}
    for j, seed in enumerate(seeds):
        ax = axes[0, j]
        for m in methods:
            losses = results[m][seed].get("train_losses", [])
            if losses:
                ax.plot(losses, label=m, color=colors.get(m, None), alpha=0.8)
        ax.set_xlabel("Epoch"); ax.set_ylabel("Loss"); ax.set_title(f"Seed {seed}")
        ax.legend(fontsize=7)
    fig.suptitle("Training Curves — Distillation Method Comparison")
    fig.tight_layout()
    fig.savefig(output_dir / "training_curves.png", dpi=200); plt.close(fig)


def plot_bar_charts(results: dict, output_dir: Path) -> None:
    """Bar chart: 3 metrics x 4 methods with error bars."""
    methods = list(results.keys())
    metrics = ["ioi_accuracy", "causal_consistency", "perplexity"]
    titles = ["IOI Accuracy (%)", "Causal Consistency", "Perplexity"]
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    x = np.arange(len(methods))
    for i, (metric, title) in enumerate(zip(metrics, titles)):
        means, stds = [], []
        for m in methods:
            vals = [results[m][s][metric] for s in results[m]]
            means.append(np.mean(vals)); stds.append(np.std(vals))
        if metric == "ioi_accuracy":
            means = [v * 100 for v in means]; stds = [v * 100 for v in stds]
        bars = axes[i].bar(x, means, yerr=stds, capsize=4, alpha=0.8,
                           color=["C0", "C1", "C2", "C3"][:len(methods)])
        axes[i].set_xticks(x); axes[i].set_xticklabels(methods, fontsize=8, rotation=15)
        axes[i].set_title(title); axes[i].set_ylabel(title)
        for bar, m in zip(bars, means):
            axes[i].text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                         f"{m:.1f}", ha="center", va="bottom", fontsize=7)
    fig.suptitle("Metric Comparison (mean +/- std across 3 seeds)")
    fig.tight_layout()
    fig.savefig(output_dir / "metric_comparison.png", dpi=200); plt.close(fig)


def build_comparison_table(results: dict, output_dir: Path) -> None:
    """Save formatted comparison table as text."""
    methods = list(results.keys())
    metrics = ["ioi_accuracy", "causal_consistency", "perplexity"]
    lines = [f"{'Method':<15} {'IOI Acc':>10} {'Consistency':>14} {'Perplexity':>12}",
             "-" * 55]
    for m in methods:
        row = f"{m:<15}"
        for metric in metrics:
            vals = [results[m][s][metric] for s in results[m]]
            mu, sd = np.mean(vals), np.std(vals)
            if metric == "ioi_accuracy":
                row += f" {mu*100:>6.1f}+/-{sd*100:4.1f}"
            else:
                row += f" {mu:>8.2f}+/-{sd:5.2f}"
        lines.append(row)
    table = "\n".join(lines)
    (output_dir / "comparison_table.txt").write_text(table)
    logger.info("Comparison table:\n%s", table)


def main() -> None:
    args = parse_args()
    cfg = OmegaConf.load(args.config)
    output_dir = Path(cfg.experiment.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Phase A: shared precomputation
    teacher = TeacherModel(cfg.teacher.name)
    ioi_train = IOIDataset(n_samples=cfg.data.n_train_pairs, seed=42)
    ioi_val = IOIDataset(n_samples=cfg.data.n_val_pairs, seed=43)
    train_loader = ioi_train.to_dataloader(batch_size=cfg.data.batch_size)
    val_loader = ioi_val.to_dataloader(batch_size=cfg.data.batch_size, shuffle=False)

    # Full batch for extraction
    full_loader = ioi_train.to_dataloader(batch_size=cfg.data.n_train_pairs, shuffle=False)
    full_batch = next(iter(full_loader))
    clean_ids = full_batch["clean_ids"].to(teacher.device)
    corrupt_ids = full_batch["corrupt_ids"].to(teacher.device)

    # Checkpoint selection
    key_pos = collect_key_positions_ioi(full_batch)
    cp_results = select_checkpoints(
        teacher.model, clean_ids, corrupt_ids,
        key_positions=key_pos, top_k=cfg.circuit.top_k_checkpoints,
    )
    checkpoints = checkpoints_to_tuples(cp_results)
    teacher_imprints = extract_causal_imprints(
        teacher.model, clean_ids, corrupt_ids, checkpoints,
    )

    # Shared CKA layer mapping (with temp student)
    set_seed(42)
    temp_student = StudentModel.from_scratch(
        n_layer=cfg.student.n_layers, n_embd=cfg.student.d_model,
        n_head=cfg.student.n_head,
    )
    t_layers = sorted(set(l for l, t in checkpoints))
    t_positions = sorted(set(t for l, t in checkpoints))
    shared_mapping, _ = find_layer_mapping_from_models(
        teacher.model, temp_student.model, clean_ids, corrupt_ids,
        teacher_layers=t_layers, token_positions=t_positions,
    )
    del temp_student
    logger.info("Shared layer mapping: %s", shared_mapping)

    # Phase B: training loop
    all_results: dict[str, dict[int, dict]] = {}
    for method_cfg in cfg.methods:
        name = method_cfg.name
        all_results[name] = {}
        for seed in cfg.experiment.seeds:
            logger.info("=" * 60)
            logger.info("Training: %s, seed=%d", name, seed)
            set_seed(seed)
            student = StudentModel.from_scratch(
                n_layer=cfg.student.n_layers, n_embd=cfg.student.d_model,
                n_head=cfg.student.n_head,
            )
            kd_fn, aux_fn, run_imprints = build_losses(
                method_cfg, cfg, teacher, student, clean_ids, corrupt_ids,
                checkpoints, teacher_imprints, shared_mapping,
            )
            train_cfg = TrainConfig(
                epochs=cfg.training.epochs, lr=cfg.training.lr,
                max_grad_norm=cfg.training.max_grad_norm,
                lambda_rcid=cfg.training.lambda_rcid, lambda_kl=cfg.training.lambda_kl,
                seed=seed, save_dir=f"{cfg.training.save_dir}/{name}/seed_{seed}",
            )
            trainer = DistillationTrainer(
                teacher_model=teacher.model, student_model=student.model,
                config=train_cfg, kd_loss_fn=kd_fn, aux_loss_fn=aux_fn,
                teacher_imprints=run_imprints,
            )
            state = trainer.train(train_loader, val_loader)
            eval_loader = ioi_val.to_dataloader(batch_size=cfg.data.batch_size, shuffle=False)
            metrics = run_evaluation(
                student.model, teacher.model, eval_loader, cfg,
                checkpoints, shared_mapping,
            )
            metrics["train_losses"] = state.train_losses
            metrics["val_losses"] = state.val_losses
            all_results[name][seed] = metrics
            logger.info("Results: acc=%.3f, cc=%.3f, ppl=%.1f",
                        metrics["ioi_accuracy"], metrics["causal_consistency"],
                        metrics["perplexity"])

    # Phase C: save and visualize
    save_json(all_results, output_dir / "exp2_results.json")
    build_comparison_table(all_results, output_dir)
    plot_training_curves(all_results, output_dir)
    plot_bar_charts(all_results, output_dir)
    logger.info("Experiment 2 complete. Results: %s", output_dir)


if __name__ == "__main__":
    main()
