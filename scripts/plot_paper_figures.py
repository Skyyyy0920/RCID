"""Generate paper figures from experiment outputs.

Reads ``training_stats.jsonl`` and ``eval_results.json`` from each
experiment directory to produce four figures:

  Figure 1: alpha_ema trajectory during training
  Figure 2: FKL and RKL curves during training
  Figure 3: Training loss comparison across methods
  Figure 4: Beta ablation bar chart (GSM8K accuracy)

Usage::

    python scripts/plot_paper_figures.py --output_dir outputs/paper
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    HAS_MPL = True
except ImportError:
    HAS_MPL = False


# ------------------------------------------------------------------
# Data loading helpers
# ------------------------------------------------------------------

def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    """Read a JSONL file into a list of dicts."""
    records: list[dict[str, Any]] = []
    if not path.exists():
        return records
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def _load_eval(path: Path) -> dict[str, Any]:
    """Load eval_results.json."""
    if not path.exists():
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _get_score(eval_data: dict[str, Any], task: str) -> float:
    """Extract a benchmark score (0-1 range)."""
    results = eval_data.get("results", {})
    task_data = results.get(task, {})
    return task_data.get("score", 0.0)


# ------------------------------------------------------------------
# Figure 1: Alpha trajectory
# ------------------------------------------------------------------

def plot_alpha_trajectory(output_dir: Path) -> None:
    """Alpha_ema vs training step for batch-EMA experiments."""
    fig, ax = plt.subplots(figsize=(8, 4))

    experiments = [
        ("klr_batch_ema", r"$\beta=0.99$ (default)", "#2196F3"),
        ("klr_beta_0.9", r"$\beta=0.9$", "#FF9800"),
        ("klr_beta_0.95", r"$\beta=0.95$", "#4CAF50"),
        ("klr_beta_0.999", r"$\beta=0.999$", "#9C27B0"),
        ("klr_no_ema", r"$\beta=0$ (no EMA)", "#F44336"),
    ]

    for exp_name, label, color in experiments:
        records = _load_jsonl(output_dir / exp_name / "training_stats.jsonl")
        if not records:
            continue
        steps = [r["step"] for r in records if "alpha_mean" in r]
        alphas = [r["alpha_mean"] for r in records if "alpha_mean" in r]
        if steps:
            ax.plot(steps, alphas, label=label, color=color, linewidth=1.5)

    ax.axhline(y=0.5, color="gray", linestyle="--", alpha=0.5, label="Jeffreys")
    ax.set_xlabel("Training Step")
    ax.set_ylabel(r"$\alpha$ (FKL weight)")
    ax.set_title(r"Adaptive $\alpha$ Trajectory During Training")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    fig_dir = output_dir / "figures"
    fig_dir.mkdir(exist_ok=True)
    fig.savefig(fig_dir / "alpha_trajectory.pdf", dpi=300)
    fig.savefig(fig_dir / "alpha_trajectory.png", dpi=150)
    plt.close(fig)
    logger.info("Saved: alpha_trajectory.pdf/png")


# ------------------------------------------------------------------
# Figure 2: FKL and RKL curves
# ------------------------------------------------------------------

def plot_fkl_rkl_curves(output_dir: Path) -> None:
    """FKL_mean and RKL_mean vs training step for klr_batch_ema."""
    fig, ax = plt.subplots(figsize=(8, 4))

    records = _load_jsonl(output_dir / "klr_batch_ema" / "training_stats.jsonl")
    if not records:
        logger.warning("No JSONL data for klr_batch_ema, skipping Figure 2")
        return

    steps = [r["step"] for r in records if "fkl_mean" in r]
    fkls = [r["fkl_mean"] for r in records if "fkl_mean" in r]
    rkls = [r["rkl_mean"] for r in records if "rkl_mean" in r]

    if steps:
        ax.plot(steps, fkls, label="FKL (forward KL)", color="#2196F3", linewidth=1.5)
        ax.plot(steps, rkls, label="RKL (reverse KL)", color="#F44336", linewidth=1.5)

    ax.set_xlabel("Training Step")
    ax.set_ylabel("KL Divergence")
    ax.set_title("Forward KL vs Reverse KL During Training (KLR-batch-EMA)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    fig_dir = output_dir / "figures"
    fig_dir.mkdir(exist_ok=True)
    fig.savefig(fig_dir / "fkl_rkl_curves.pdf", dpi=300)
    fig.savefig(fig_dir / "fkl_rkl_curves.png", dpi=150)
    plt.close(fig)
    logger.info("Saved: fkl_rkl_curves.pdf/png")


# ------------------------------------------------------------------
# Figure 3: Training loss comparison
# ------------------------------------------------------------------

def plot_loss_comparison(output_dir: Path) -> None:
    """Training loss curves across main methods."""
    fig, ax = plt.subplots(figsize=(8, 4))

    methods = [
        ("forward_kl", "Forward KL", "#607D8B"),
        ("jeffreys", "Jeffreys", "#FF9800"),
        ("akl", "AKL", "#9C27B0"),
        ("klr_batch_ema", "KLR-batch-EMA", "#2196F3"),
    ]

    for exp_name, label, color in methods:
        records = _load_jsonl(output_dir / exp_name / "training_stats.jsonl")
        if not records:
            continue
        steps = [r["step"] for r in records]
        losses = [r["loss"] for r in records]
        if steps:
            ax.plot(steps, losses, label=label, color=color, linewidth=1.5, alpha=0.8)

    ax.set_xlabel("Training Step")
    ax.set_ylabel("Training Loss")
    ax.set_title("Training Loss Comparison")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    fig_dir = output_dir / "figures"
    fig_dir.mkdir(exist_ok=True)
    fig.savefig(fig_dir / "loss_comparison.pdf", dpi=300)
    fig.savefig(fig_dir / "loss_comparison.png", dpi=150)
    plt.close(fig)
    logger.info("Saved: loss_comparison.pdf/png")


# ------------------------------------------------------------------
# Figure 4: Beta ablation bar chart
# ------------------------------------------------------------------

def plot_beta_ablation(output_dir: Path) -> None:
    """GSM8K accuracy for different beta values."""
    fig, ax = plt.subplots(figsize=(7, 4))

    configs = [
        ("klr_no_ema", r"$\beta=0$"),
        ("klr_beta_0.9", r"$\beta=0.9$"),
        ("klr_beta_0.95", r"$\beta=0.95$"),
        ("klr_batch_ema", r"$\beta=0.99$"),
        ("klr_beta_0.999", r"$\beta=0.999$"),
    ]

    labels: list[str] = []
    scores: list[float] = []
    colors: list[str] = []

    for exp_name, label in configs:
        eval_data = _load_eval(output_dir / exp_name / "eval_results.json")
        gsm8k = _get_score(eval_data, "gsm8k") * 100  # percent
        labels.append(label)
        scores.append(gsm8k)
        colors.append("#2196F3" if exp_name == "klr_batch_ema" else "#90CAF9")

    bars = ax.bar(range(len(labels)), scores, color=colors, edgecolor="white")
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels)
    ax.set_ylabel("GSM8K Accuracy (%)")
    ax.set_title(r"Effect of EMA Coefficient $\beta$ on GSM8K")
    ax.grid(True, alpha=0.3, axis="y")

    # Value labels on bars
    for bar, score in zip(bars, scores):
        if score > 0:
            ax.text(
                bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
                f"{score:.1f}", ha="center", va="bottom", fontsize=9,
            )

    fig.tight_layout()

    fig_dir = output_dir / "figures"
    fig_dir.mkdir(exist_ok=True)
    fig.savefig(fig_dir / "beta_ablation.pdf", dpi=300)
    fig.savefig(fig_dir / "beta_ablation.png", dpi=150)
    plt.close(fig)
    logger.info("Saved: beta_ablation.pdf/png")


# ------------------------------------------------------------------
# CLI
# ------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(description="Generate paper figures")
    ap.add_argument("--output_dir", default="outputs/paper")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")

    if not HAS_MPL:
        logger.error("matplotlib not installed, cannot generate figures")
        return

    output_dir = Path(args.output_dir)
    logger.info("Generating figures from: %s", output_dir)

    plot_alpha_trajectory(output_dir)
    plot_fkl_rkl_curves(output_dir)
    plot_loss_comparison(output_dir)
    plot_beta_ablation(output_dir)

    logger.info("All figures saved to: %s/figures/", output_dir)


if __name__ == "__main__":
    main()
