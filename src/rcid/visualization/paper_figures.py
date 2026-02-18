"""Paper figure generation: tables, scatter plots, bar charts, heatmaps."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

# DPI=300, colorblind-friendly palette
DPI = 300
PALETTE = sns.color_palette("colorblind")
METHOD_COLORS = {
    "standard_kd": PALETTE[0],
    "fitnets": PALETTE[1],
    "informed_fitnets": PALETTE[2],
    "rcid": PALETTE[3],
}
METHOD_LABELS = {
    "standard_kd": "StandardKD",
    "fitnets": "FitNets",
    "informed_fitnets": "InformedFitNets",
    "rcid": "RCID",
}


def _load_results(results_dir: Path) -> list[dict[str, Any]]:
    """Load all JSON result files from a directory."""
    results = []
    for p in sorted(results_dir.glob("**/*.json")):
        with open(p) as f:
            results.append(json.load(f))
    return results


def plot_exp1_table(
    results_dir: Path,
    output_path: Path,
    model_family: str = "qwen3",
) -> None:
    """Table 1: Baseline methods — task accuracy vs causal consistency."""
    results = _load_results(results_dir / model_family)
    if not results:
        return

    fig, ax = plt.subplots(figsize=(8, 3))
    ax.axis("off")

    # Aggregate by method × task
    table_data: dict[str, dict[str, dict[str, list[float]]]] = {}
    for r in results:
        method = r.get("method", "unknown")
        task = r.get("task", "unknown")
        table_data.setdefault(method, {}).setdefault(task, {"acc": [], "cc": []})
        table_data[method][task]["acc"].append(r.get("accuracy", 0.0))
        table_data[method][task]["cc"].append(r.get("causal_consistency", 0.0))

    tasks = sorted({t for m in table_data.values() for t in m})
    methods = sorted(table_data.keys())

    col_labels = ["Method"] + [f"{t} Acc" for t in tasks] + [f"{t} CC" for t in tasks]
    cell_text = []
    for m in methods:
        row = [METHOD_LABELS.get(m, m)]
        for t in tasks:
            vals = table_data.get(m, {}).get(t, {}).get("acc", [0.0])
            row.append(f"{sum(vals)/max(len(vals),1):.1%}")
        for t in tasks:
            vals = table_data.get(m, {}).get(t, {}).get("cc", [0.0])
            row.append(f"{sum(vals)/max(len(vals),1):.3f}")
        cell_text.append(row)

    table = ax.table(cellText=cell_text, colLabels=col_labels, loc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.2, 1.5)
    ax.set_title(f"Table 1: Baseline Mechanism Preservation ({model_family})", pad=20)
    fig.savefig(output_path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)


def plot_exp2_table(
    results_dir: Path,
    output_path: Path,
    model_family: str = "qwen3",
) -> None:
    """Table 2: Four-method comparison — accuracy, CC, perplexity."""
    results = _load_results(results_dir / model_family)
    if not results:
        return

    fig, ax = plt.subplots(figsize=(10, 3))
    ax.axis("off")

    table_data: dict[str, dict[str, dict[str, list[float]]]] = {}
    for r in results:
        method = r.get("method", "unknown")
        task = r.get("task", "unknown")
        d = table_data.setdefault(method, {}).setdefault(task, {
            "acc": [], "cc": [], "ppl": [],
        })
        d["acc"].append(r.get("accuracy", 0.0))
        d["cc"].append(r.get("causal_consistency", 0.0))
        d["ppl"].append(r.get("perplexity", 0.0))

    tasks = sorted({t for m in table_data.values() for t in m})
    methods = ["standard_kd", "fitnets", "informed_fitnets", "rcid"]
    methods = [m for m in methods if m in table_data]

    col_labels = ["Method"]
    for t in tasks:
        col_labels += [f"{t} Acc", f"{t} CC", f"{t} PPL"]
    cell_text = []
    for m in methods:
        row = [METHOD_LABELS.get(m, m)]
        for t in tasks:
            d = table_data.get(m, {}).get(t, {"acc": [], "cc": [], "ppl": []})
            for key, fmt in [("acc", ".1%"), ("cc", ".3f"), ("ppl", ".1f")]:
                vals = d.get(key, [0.0])
                mean = sum(vals) / max(len(vals), 1)
                row.append(f"{mean:{fmt}}")
        cell_text.append(row)

    table = ax.table(cellText=cell_text, colLabels=col_labels, loc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1.0, 1.5)
    ax.set_title(f"Table 2: RCID Comparison ({model_family})", pad=20)
    fig.savefig(output_path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)


def plot_exp3_scatter(
    results_path: Path,
    output_path: Path,
) -> None:
    """Scatter plot: causal consistency vs OOD degradation."""
    with open(results_path) as f:
        data = json.load(f)

    points = data.get("scatter_data", [])
    if not points:
        return

    fig, ax = plt.subplots(figsize=(6, 5))
    for pt in points:
        method = pt["method"]
        color = METHOD_COLORS.get(method, "gray")
        label = METHOD_LABELS.get(method, method)
        ax.scatter(
            pt["causal_consistency"], pt["ood_degradation"],
            c=[color], label=label, s=40, alpha=0.7, edgecolors="black", linewidths=0.5,
        )

    # Deduplicate legend
    handles, labels = ax.get_legend_handles_labels()
    seen: dict[str, Any] = {}
    unique_handles, unique_labels = [], []
    for h, l in zip(handles, labels):
        if l not in seen:
            seen[l] = True
            unique_handles.append(h)
            unique_labels.append(l)
    ax.legend(unique_handles, unique_labels, loc="upper right")

    ax.set_xlabel("Causal Consistency (Pearson r)")
    ax.set_ylabel("OOD Degradation")
    ax.set_title("Exp 3: Mechanistic Consistency vs Robustness")
    fig.savefig(output_path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)


def plot_exp4_bar(
    results_dir: Path,
    output_path: Path,
) -> None:
    """Bar chart: information selectivity of h^T vs d^T."""
    results = _load_results(results_dir)
    if not results:
        return

    fig, ax = plt.subplots(figsize=(8, 5))
    tasks = []
    h_sel = []
    d_sel = []

    for r in results:
        task = r.get("task", "unknown")
        for layer_data in r.get("per_layer", {}).values():
            tasks.append(task)
            h_sel.append(layer_data.get("h_selectivity", 0.0))
            d_sel.append(layer_data.get("d_selectivity", 0.0))

    if not tasks:
        plt.close(fig)
        return

    import numpy as np
    x = np.arange(len(tasks))
    width = 0.35
    ax.bar(x - width / 2, h_sel, width, label=r"$h^T$ (raw)", color=PALETTE[0])
    ax.bar(x + width / 2, d_sel, width, label=r"$d^T$ (contrastive)", color=PALETTE[3])
    ax.set_xticks(x)
    ax.set_xticklabels(tasks, rotation=45, ha="right")
    ax.set_ylabel("Selectivity")
    ax.set_title("Exp 4: Information Purity")
    ax.legend()
    fig.savefig(output_path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)


def plot_cka_heatmap(
    cka_matrix_path: Path,
    output_path: Path,
    model_family: str = "qwen3",
) -> None:
    """CKA heatmap between teacher and student layers."""
    import torch
    cka = torch.load(cka_matrix_path, weights_only=True)  # (n_T, n_S)

    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(
        cka.numpy(), ax=ax, cmap="viridis", vmin=0, vmax=1,
        xticklabels=5, yticklabels=5,
    )
    ax.set_xlabel("Student Layer")
    ax.set_ylabel("Teacher Layer")
    ax.set_title(f"CKA Similarity ({model_family})")
    fig.savefig(output_path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)


def plot_cross_architecture_table(
    qwen3_dir: Path,
    llama3_dir: Path,
    output_path: Path,
) -> None:
    """Cross-architecture comparison table (Qwen3 vs LLaMA3)."""
    qwen_results = _load_results(qwen3_dir)
    llama_results = _load_results(llama3_dir)
    if not qwen_results and not llama_results:
        return

    def _aggregate(
        results: list[dict[str, Any]],
    ) -> dict[str, dict[str, float]]:
        agg: dict[str, dict[str, list[float]]] = {}
        for r in results:
            method = r.get("method", "unknown")
            d = agg.setdefault(method, {"acc": [], "cc": []})
            d["acc"].append(r.get("accuracy", 0.0))
            d["cc"].append(r.get("causal_consistency", 0.0))
        return {
            m: {
                "acc": sum(d["acc"]) / max(len(d["acc"]), 1),
                "cc": sum(d["cc"]) / max(len(d["cc"]), 1),
            }
            for m, d in agg.items()
        }

    qwen_agg = _aggregate(qwen_results)
    llama_agg = _aggregate(llama_results)

    fig, ax = plt.subplots(figsize=(8, 3))
    ax.axis("off")

    methods = sorted(set(list(qwen_agg.keys()) + list(llama_agg.keys())))
    col_labels = [
        "Method",
        "Qwen3 Acc", "Qwen3 CC",
        "LLaMA3 Acc", "LLaMA3 CC",
    ]
    cell_text = []
    for m in methods:
        q = qwen_agg.get(m, {"acc": 0, "cc": 0})
        l = llama_agg.get(m, {"acc": 0, "cc": 0})
        cell_text.append([
            METHOD_LABELS.get(m, m),
            f"{q['acc']:.1%}", f"{q['cc']:.3f}",
            f"{l['acc']:.1%}", f"{l['cc']:.3f}",
        ])

    table = ax.table(cellText=cell_text, colLabels=col_labels, loc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.2, 1.5)
    ax.set_title("Cross-Architecture Comparison", pad=20)
    fig.savefig(output_path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
