"""Experiment 3: Multi-task Causal Imprint Analysis (Section 4.3).

Extracts imprints for IOI and Greater-Than independently,
analyzes cross-task overlap, then compares distillation strategies.

Usage: python scripts/run_exp3.py [--config configs/exp3_multitask.yaml]
"""

from __future__ import annotations

import argparse, json, logging, random, sys
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
    checkpoints_to_tuples, collect_key_positions_greater_than,
    collect_key_positions_ioi, select_checkpoints,
)
from rcid.circuit.patching import extract_causal_imprints
from rcid.data.greater_than import GreaterThanDataset
from rcid.data.ioi import IOIDataset
from rcid.distillation.baselines import StandardKDLoss
from rcid.distillation.rcid_loss import RCIDLoss
from rcid.distillation.trainer import DistillationTrainer, TrainConfig
from rcid.eval import causal_consistency, perplexity, task_accuracy
from rcid.models.student import StudentModel
from rcid.models.teacher import TeacherModel

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Exp3: Multi-task Analysis")
    p.add_argument("--config", default="configs/exp3_multitask.yaml")
    return p.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def save_json(data: dict, path: Path) -> None:
    with open(path, "w") as f:
        json.dump(data, f, indent=2, default=str)
    logger.info("Saved: %s", path)


def extract_task_data(teacher: TeacherModel, task: str, cfg: Any):
    """Extract checkpoints + imprints for one task. Returns (cps, imprints, clean, corrupt)."""
    n = getattr(cfg.data, task).n_train_pairs
    ds = (IOIDataset(n_samples=n, seed=42) if task == "ioi"
          else GreaterThanDataset(n_samples=n, seed=42))
    batch = next(iter(ds.to_dataloader(batch_size=n, shuffle=False)))
    key_pos = (collect_key_positions_ioi(batch) if task == "ioi"
               else collect_key_positions_greater_than(batch))
    clean = batch["clean_ids"].to(teacher.device)
    corrupt = batch["corrupt_ids"].to(teacher.device)
    cp_res = select_checkpoints(
        teacher.model, clean, corrupt,
        key_positions=key_pos, top_k=cfg.circuit.top_k_checkpoints)
    cps = checkpoints_to_tuples(cp_res)
    imprints = extract_causal_imprints(teacher.model, clean, corrupt, cps)
    logger.info("Task=%s: %d checkpoints", task, len(cps))
    return cps, imprints, clean, corrupt


def analyze_overlap(
    ioi_cps: list[tuple[int, int]], gt_cps: list[tuple[int, int]],
    ioi_imp: dict, gt_imp: dict, output_dir: Path,
) -> dict[str, Any]:
    """Analyze cross-task imprint overlap: cosine matrix + principal angles."""
    ioi_L = sorted(set(l for l, _ in ioi_cps))
    gt_L = sorted(set(l for l, _ in gt_cps))
    shared = sorted(set(ioi_L) & set(gt_L))
    eps = 1e-8
    # Mean directions per checkpoint
    def _normed_means(cps, imp):
        out = []
        for cp in cps:
            m = imp[cp].mean(dim=0)
            out.append(m / (m.norm() + eps))
        return out
    ioi_m, gt_m = _normed_means(ioi_cps, ioi_imp), _normed_means(gt_cps, gt_imp)
    cosines = np.array([[(a @ b).item() for b in gt_m] for a in ioi_m])
    # Principal angles
    ioi_all = torch.cat([ioi_imp[c] for c in ioi_cps], dim=0).T.float()
    gt_all = torch.cat([gt_imp[c] for c in gt_cps], dim=0).T.float()
    k = min(5, ioi_all.shape[1], gt_all.shape[1])
    U_i = torch.linalg.svd(ioi_all)[0][:, :k]
    U_g = torch.linalg.svd(gt_all)[0][:, :k]
    pa = torch.arccos(torch.linalg.svdvals(U_i.T @ U_g)[:k].clamp(-1, 1)).tolist()
    # Plot
    _plot_cosine_overlap(ioi_cps, gt_cps, cosines, output_dir)
    report = {
        "ioi_layers": ioi_L, "gt_layers": gt_L, "shared_layers": shared,
        "cosine_mean_abs": float(np.mean(np.abs(cosines))),
        "principal_angles_deg": [float(np.degrees(a)) for a in pa],
        "mean_pa_deg": float(np.mean([np.degrees(a) for a in pa])),
    }
    logger.info("Overlap: %d shared layers, |cos|=%.3f, PA=%.1f°",
                len(shared), report["cosine_mean_abs"], report["mean_pa_deg"])
    return report


def _plot_cosine_overlap(ioi_cps, gt_cps, cosines, out_dir):
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(cosines, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
    ax.set_xticks(range(len(gt_cps))); ax.set_yticks(range(len(ioi_cps)))
    ax.set_xticklabels([f"L{l},P{t}" for l, t in gt_cps], fontsize=8, rotation=30)
    ax.set_yticklabels([f"L{l},P{t}" for l, t in ioi_cps], fontsize=8)
    ax.set_xlabel("GT Checkpoints"); ax.set_ylabel("IOI Checkpoints")
    ax.set_title("Cross-Task Cosine Similarity")
    for i in range(len(ioi_cps)):
        for j in range(len(gt_cps)):
            ax.text(j, i, f"{cosines[i,j]:.2f}", ha="center", va="center",
                    fontsize=7, color="white" if abs(cosines[i,j]) > 0.5 else "black")
    fig.colorbar(im, ax=ax, shrink=0.8); fig.tight_layout()
    fig.savefig(out_dir / "cosine_overlap.png", dpi=200); plt.close(fig)


def build_strategy(
    strat_cfg: Any, cfg: Any, teacher: TeacherModel, student: StudentModel,
    ioi_d: dict, gt_d: dict, mapping: dict[int, int],
) -> tuple[nn.Module, nn.Module | None, dict | None, list, dict]:
    """Build KD + aux loss for a strategy. Returns (kd, aux, imprints, cps, map)."""
    kd = StandardKDLoss(temperature=cfg.training.temperature)
    src = strat_cfg.task_source
    if src is None or str(src) == "null":
        return kd, None, None, ioi_d["cps"], mapping
    if src == "ioi":
        cps, t_imp, cl, co = ioi_d["cps"], ioi_d["imp"], ioi_d["cl"], ioi_d["co"]
    elif src == "greater_than":
        cps, t_imp, cl, co = gt_d["cps"], gt_d["imp"], gt_d["cl"], gt_d["co"]
    else:  # "both"
        cps = list(dict.fromkeys(ioi_d["cps"] + gt_d["cps"]))
        t_imp = {**ioi_d["imp"], **gt_d["imp"]}
        cl, co = ioi_d["cl"], ioi_d["co"]
    # Procrustes
    student.model.eval()
    s_cps = [(mapping[l], t) for l, t in cps]
    s_imp = extract_causal_imprints(student.model, cl, co, s_cps)
    student.model.train()
    source = torch.cat([s_imp[(mapping[l], t)] for l, t in cps], dim=0)
    target = torch.cat([t_imp[(l, t)] for l, t in cps], dim=0)
    W = procrustes_align(source, target)
    aux = RCIDLoss(W=W, checkpoints=cps, layer_mapping=mapping)
    return kd, aux, t_imp, cps, mapping


def evaluate_both(
    student: nn.Module, teacher: nn.Module, cfg: Any,
    ioi_vl: Any, gt_vl: Any, cps: list, mapping: dict,
) -> dict[str, float]:
    """Evaluate IOI acc, GT acc, causal consistency, perplexity."""
    student.eval()
    r: dict[str, float] = {}
    r["ioi_accuracy"] = task_accuracy.evaluate(student, ioi_vl, task="ioi")["accuracy"]
    r["gt_accuracy"] = task_accuracy.evaluate(
        student, gt_vl, task="greater_than")["accuracy"]
    r["causal_consistency"] = causal_consistency.evaluate(
        student, ioi_vl, teacher_model=teacher,
        checkpoints=cps, layer_mapping=mapping)["consistency"]
    r["perplexity"] = perplexity.evaluate(
        student, max_samples=cfg.eval.perplexity_max_samples)["perplexity"]
    return r


def _agg(results, strats, met):
    """Aggregate metric across seeds → (means, stds)."""
    means, stds = [], []
    for s in strats:
        vals = [results[s][sd][met] for sd in results[s]]
        means.append(np.mean(vals)); stds.append(np.std(vals))
    return means, stds


def plot_comparison(results: dict, output_dir: Path) -> None:
    """Bar chart: 4 metrics × 4 strategies with error bars."""
    strats = list(results.keys())
    mets = ["ioi_accuracy", "gt_accuracy", "causal_consistency", "perplexity"]
    titles = ["IOI Acc (%)", "GT Acc (%)", "Consistency", "Perplexity"]
    fig, axes = plt.subplots(1, 4, figsize=(18, 4)); x = np.arange(len(strats))
    for i, (met, title) in enumerate(zip(mets, titles)):
        means, stds = _agg(results, strats, met)
        if "accuracy" in met:
            means = [v*100 for v in means]; stds = [v*100 for v in stds]
        bars = axes[i].bar(x, means, yerr=stds, capsize=4, alpha=0.8,
                           color=["C0","C1","C2","C3"][:len(strats)])
        axes[i].set_xticks(x); axes[i].set_xticklabels(strats, fontsize=7, rotation=20)
        axes[i].set_title(title); axes[i].set_ylabel(title)
        for bar, m in zip(bars, means):
            axes[i].text(bar.get_x()+bar.get_width()/2, bar.get_height(),
                         f"{m:.1f}", ha="center", va="bottom", fontsize=7)
    fig.suptitle("Multi-task Strategy Comparison"); fig.tight_layout()
    fig.savefig(output_dir / "multitask_comparison.png", dpi=200); plt.close(fig)


def save_table(results: dict, output_dir: Path) -> None:
    """Save formatted comparison table as text."""
    strats = list(results.keys())
    mets = ["ioi_accuracy", "gt_accuracy", "causal_consistency", "perplexity"]
    lines = [f"{'Strategy':<20} {'IOI':>8} {'GT':>8} {'CC':>10} {'PPL':>10}", "-"*60]
    for s in strats:
        row = f"{s:<20}"
        for m in mets:
            mu, sd = _agg(results, [s], m)
            mu, sd = mu[0], sd[0]
            fmt = f" {mu*100:>5.1f}±{sd*100:3.1f}" if "accuracy" in m else f" {mu:>6.2f}±{sd:4.2f}"
            row += fmt
        lines.append(row)
    table = "\n".join(lines)
    (output_dir / "comparison_table.txt").write_text(table)
    logger.info("Comparison table:\n%s", table)


def main() -> None:
    args = parse_args()
    cfg = OmegaConf.load(args.config)
    out = Path(cfg.experiment.output_dir); out.mkdir(parents=True, exist_ok=True)
    # Phase A: extract checkpoints for both tasks
    teacher = TeacherModel(cfg.teacher.name)
    ioi_cps, ioi_imp, ioi_cl, ioi_co = extract_task_data(teacher, "ioi", cfg)
    gt_cps, gt_imp, gt_cl, gt_co = extract_task_data(teacher, "greater_than", cfg)
    ioi_d = {"cps": ioi_cps, "imp": ioi_imp, "cl": ioi_cl, "co": ioi_co}
    gt_d = {"cps": gt_cps, "imp": gt_imp, "cl": gt_cl, "co": gt_co}
    # Phase B: overlap analysis
    overlap = analyze_overlap(ioi_cps, gt_cps, ioi_imp, gt_imp, out)
    # Shared CKA layer mapping
    set_seed(42)
    tmp = StudentModel.from_scratch(
        n_layer=cfg.student.n_layers, n_embd=cfg.student.d_model, n_head=cfg.student.n_head)
    all_cps = ioi_cps + gt_cps
    shared_map, _ = find_layer_mapping_from_models(
        teacher.model, tmp.model, ioi_cl, ioi_co,
        teacher_layers=sorted(set(l for l,_ in all_cps)),
        token_positions=sorted(set(t for _,t in all_cps)))
    del tmp; logger.info("Shared layer mapping: %s", shared_map)
    # Datasets
    train_loader = IOIDataset(n_samples=cfg.data.ioi.n_train_pairs, seed=42
                              ).to_dataloader(batch_size=cfg.data.ioi.batch_size)
    ioi_vl = IOIDataset(n_samples=cfg.data.ioi.n_val_pairs, seed=43
                        ).to_dataloader(batch_size=cfg.data.ioi.batch_size, shuffle=False)
    gt_vl = GreaterThanDataset(n_samples=cfg.data.greater_than.n_val_pairs, seed=43
                               ).to_dataloader(batch_size=cfg.data.greater_than.batch_size,
                                               shuffle=False)
    # Phase C: 4 strategies × 3 seeds
    all_results: dict[str, dict[int, dict]] = {}
    for strat_cfg in cfg.strategies:
        name = strat_cfg.name; all_results[name] = {}
        for seed in cfg.experiment.seeds:
            logger.info("=" * 60 + "\nTraining: %s, seed=%d", name, seed)
            set_seed(seed)
            student = StudentModel.from_scratch(
                n_layer=cfg.student.n_layers, n_embd=cfg.student.d_model,
                n_head=cfg.student.n_head)
            kd, aux, imp, cps, mp = build_strategy(
                strat_cfg, cfg, teacher, student, ioi_d, gt_d, shared_map)
            tcfg = TrainConfig(
                epochs=cfg.training.epochs, lr=cfg.training.lr,
                max_grad_norm=cfg.training.get("max_grad_norm", 1.0),
                lambda_rcid=cfg.training.lambda_rcid, lambda_kl=cfg.training.lambda_kl,
                seed=seed, save_dir=f"{cfg.training.save_dir}/{name}/seed_{seed}")
            trainer = DistillationTrainer(
                teacher_model=teacher.model, student_model=student.model,
                config=tcfg, kd_loss_fn=kd, aux_loss_fn=aux, teacher_imprints=imp)
            state = trainer.train(train_loader, ioi_vl)
            metrics = evaluate_both(
                student.model, teacher.model, cfg, ioi_vl, gt_vl, cps, mp)
            metrics["train_losses"] = state.train_losses
            metrics["val_losses"] = state.val_losses
            all_results[name][seed] = metrics
            logger.info("Results: ioi=%.3f, gt=%.3f, cc=%.3f, ppl=%.1f",
                        metrics["ioi_accuracy"], metrics["gt_accuracy"],
                        metrics["causal_consistency"], metrics["perplexity"])
    # Phase D: save and visualize
    save_json({"overlap_analysis": overlap, "strategy_results": all_results},
              out / "exp3_results.json")
    save_table(all_results, out); plot_comparison(all_results, out)
    logger.info("Experiment 3 complete. Results: %s", out)


if __name__ == "__main__":
    main()
