"""Experiment 1: Causal Imprint Extraction & Analysis (Section 4.1).

Extracts causal imprints from GPT-2 on IOI, selects top-k checkpoints,
generates heatmap/PCA/cosine visualizations, compares with known circuit.

Usage: python scripts/run_exp1.py [--config configs/exp1_imprint_extraction.yaml]
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from omegaconf import OmegaConf

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from rcid.circuit.checkpoint_selection import (
    CheckpointResult, checkpoints_to_tuples, collect_key_positions_ioi,
    select_checkpoints,
)
from rcid.circuit.patching import compute_imprint_norms, extract_causal_imprints
from rcid.data.ioi import IOIDataset
from rcid.models.teacher import TeacherModel

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Exp1: Causal Imprint Extraction")
    p.add_argument("--config", default="configs/exp1_imprint_extraction.yaml")
    return p.parse_args()


def save_json(data: dict, path: Path) -> None:
    with open(path, "w") as f:
        json.dump(data, f, indent=2, default=str)
    logger.info("Saved: %s", path)


def compare_with_known_circuit(
    cp_results: list[CheckpointResult], cfg: Any,
) -> dict[str, Any]:
    """Compare discovered checkpoints against known IOI circuit."""
    nm = set(cfg.circuit.known_name_mover_layers)
    si = set(cfg.circuit.known_s_inhibition_layers)
    layers = [r.layer for r in cp_results]
    nm_hit = sum(1 for l in layers if l in nm)
    si_hit = sum(1 for l in layers if l in si)
    report = {
        "n_in_name_movers_L9_11": nm_hit,
        "n_in_s_inhibition_L7_8": si_hit,
        "n_in_known_circuit": nm_hit + si_hit,
        "n_total": len(cp_results),
        "discovered_layers": layers,
        "discovered_positions": [r.token_pos for r in cp_results],
    }
    logger.info("Known circuit overlap: %d/%d in NM, %d/%d in SI",
                nm_hit, len(cp_results), si_hit, len(cp_results))
    return report


def plot_norm_heatmap(
    all_norms: list[tuple[int, int, float]],
    key_positions: list[int],
    cfg: Any,
    output_dir: Path,
) -> None:
    """Plot (layer x key_position) heatmap of imprint norms."""
    n_layers = cfg.teacher.n_layers
    n_pos = len(key_positions)
    matrix = np.zeros((n_layers, n_pos))  # (n_layers, n_pos)
    pos_to_idx = {p: i for i, p in enumerate(key_positions)}
    for layer, pos, norm in all_norms:
        if pos in pos_to_idx:
            matrix[layer, pos_to_idx[pos]] = norm

    fig, ax = plt.subplots(figsize=(max(8, n_pos * 0.8), 6))
    im = ax.imshow(matrix, aspect="auto", cmap="YlOrRd", origin="lower")
    ax.set_xlabel("Key Token Position")
    ax.set_ylabel("Layer")
    ax.set_title("Causal Imprint Norms — IOI Task (GPT-2)")
    ax.set_xticks(range(n_pos))
    ax.set_xticklabels([str(p) for p in key_positions], fontsize=8)
    ax.set_yticks(range(n_layers))
    ax.set_yticklabels([f"L{i}" for i in range(n_layers)], fontsize=9)
    fig.colorbar(im, ax=ax, shrink=0.8, label="Mean L2 Norm")
    # Mark top-5
    flat = [(matrix[l, j], l, j) for l in range(n_layers) for j in range(n_pos)]
    flat.sort(reverse=True)
    for rank, (val, l, j) in enumerate(flat[:5]):
        ax.plot(j, l, "k*", markersize=10)
        ax.annotate(f"#{rank+1}", (j, l), xytext=(3, 3),
                    textcoords="offset points", fontsize=7, fontweight="bold")
    fig.tight_layout()
    fig.savefig(output_dir / "imprint_heatmap.png", dpi=cfg.visualization.figure_dpi)
    plt.close(fig)
    logger.info("Saved: imprint_heatmap.png")


def plot_pca(
    top_imprints: dict[tuple[int, int], torch.Tensor],
    checkpoints: list[tuple[int, int]],
    output_dir: Path,
) -> None:
    """PCA scatter of top-k checkpoint imprints (PC1 vs PC2, PC1 vs PC3)."""
    vectors = []
    for l, t in checkpoints:
        vectors.append(top_imprints[(l, t)].cpu().float())  # (N, d_model)
    all_vecs = torch.cat(vectors, dim=0)  # (k*N, d_model)
    # Torch-based PCA via SVD on centered data
    mean = all_vecs.mean(dim=0, keepdim=True)
    centered = all_vecs - mean
    U, S, Vh = torch.linalg.svd(centered, full_matrices=False)
    proj = (centered @ Vh[:3].T).numpy()  # (k*N, 3)
    total_var = (S ** 2).sum()
    ev = [(S[i] ** 2 / total_var).item() for i in range(3)]
    colors = plt.cm.tab10(np.linspace(0, 1, len(checkpoints)))
    n_per = vectors[0].shape[0]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for i, (l, t) in enumerate(checkpoints):
        s, e = i * n_per, (i + 1) * n_per
        axes[0].scatter(proj[s:e, 0], proj[s:e, 1], c=[colors[i]],
                        label=f"L{l},P{t}", alpha=0.5, s=15)
        axes[1].scatter(proj[s:e, 0], proj[s:e, 2], c=[colors[i]], alpha=0.5, s=15)
    axes[0].set_xlabel(f"PC1 ({ev[0]:.1%})"); axes[0].set_ylabel(f"PC2 ({ev[1]:.1%})")
    axes[1].set_xlabel(f"PC1 ({ev[0]:.1%})"); axes[1].set_ylabel(f"PC3 ({ev[2]:.1%})")
    axes[0].legend(fontsize=8); axes[0].set_title("PC1 vs PC2")
    axes[1].set_title("PC1 vs PC3")
    fig.suptitle(f"PCA of Top-{len(checkpoints)} Causal Imprints ({sum(ev):.1%} var)")
    fig.tight_layout()
    fig.savefig(output_dir / "imprint_pca.png", dpi=200)
    plt.close(fig)
    logger.info("Saved: imprint_pca.png")


def plot_cosine_matrix(
    top_imprints: dict[tuple[int, int], torch.Tensor],
    checkpoints: list[tuple[int, int]],
    output_dir: Path,
) -> None:
    """Cosine similarity matrix between checkpoint mean directions."""
    eps = 1e-8
    means = []
    for l, t in checkpoints:
        m = top_imprints[(l, t)].mean(dim=0)  # (d_model,)
        means.append(m / (m.norm() + eps))
    n = len(means)
    sim = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            sim[i, j] = (means[i] @ means[j]).item()
    labels = [f"L{l},P{t}" for l, t in checkpoints]
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(sim, cmap="RdBu_r", vmin=-1, vmax=1, aspect="equal")
    ax.set_xticks(range(n)); ax.set_xticklabels(labels, fontsize=8, rotation=30)
    ax.set_yticks(range(n)); ax.set_yticklabels(labels, fontsize=8)
    for i_ in range(n):
        for j_ in range(n):
            ax.text(j_, i_, f"{sim[i_, j_]:.2f}", ha="center", va="center",
                    fontsize=7, color="white" if abs(sim[i_, j_]) > 0.5 else "black")
    fig.colorbar(im, ax=ax, shrink=0.8)
    ax.set_title("Cosine Similarity of Mean Imprint Directions")
    fig.tight_layout()
    fig.savefig(output_dir / "imprint_cosine.png", dpi=200)
    plt.close(fig)
    logger.info("Saved: imprint_cosine.png")


def main() -> None:
    args = parse_args()
    cfg = OmegaConf.load(args.config)
    output_dir = Path(cfg.experiment.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(cfg.experiment.seed)

    # 1. Load teacher
    teacher = TeacherModel(cfg.teacher.name)

    # 2. Build IOI dataset
    dataset = IOIDataset(n_samples=cfg.data.n_contrastive_pairs, seed=cfg.experiment.seed)
    loader = dataset.to_dataloader(batch_size=cfg.data.batch_size, shuffle=False)
    batch = next(iter(loader))
    clean_ids = batch["clean_ids"].to(teacher.device)    # (N, seq_len)
    corrupt_ids = batch["corrupt_ids"].to(teacher.device)  # (N, seq_len)

    # 3. Key positions
    key_positions = collect_key_positions_ioi(batch)
    logger.info("Key token positions: %s", key_positions)

    # 4. Select top-k checkpoints
    cp_results = select_checkpoints(
        teacher.model, clean_ids, corrupt_ids,
        key_positions=key_positions, top_k=cfg.circuit.top_k_checkpoints,
    )
    checkpoints = checkpoints_to_tuples(cp_results)

    # 5. Full norm scan for heatmap
    all_norms = compute_imprint_norms(
        teacher.model, clean_ids, corrupt_ids,
        layers=list(range(cfg.teacher.n_layers)), token_positions=key_positions,
    )

    # 6. Extract top-k imprints for PCA + cosine
    top_imprints = extract_causal_imprints(
        teacher.model, clean_ids, corrupt_ids, checkpoints,
    )

    # 7. Compare with known circuit
    overlap = compare_with_known_circuit(cp_results, cfg)

    # 8. Visualizations
    plot_norm_heatmap(all_norms, key_positions, cfg, output_dir)
    plot_pca(top_imprints, checkpoints, output_dir)
    plot_cosine_matrix(top_imprints, checkpoints, output_dir)

    # 9. Save summary
    summary = {
        "top_checkpoints": [
            {"rank": i + 1, "layer": r.layer, "token_pos": r.token_pos,
             "mean_norm": round(r.mean_norm, 4), "std_norm": round(r.std_norm, 4)}
            for i, r in enumerate(cp_results)
        ],
        "known_circuit_overlap": overlap,
        "n_contrastive_pairs": cfg.data.n_contrastive_pairs,
        "key_positions": key_positions,
    }
    save_json(summary, output_dir / "exp1_summary.json")
    logger.info("Experiment 1 complete. Results: %s", output_dir)


if __name__ == "__main__":
    main()
