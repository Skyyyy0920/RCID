"""IOI 任务因果痕迹可视化分析。

生成三张图：
1. 热力图：所有 (layer, token_position) 的痕迹范数
2. PCA：top-5 检查点痕迹的前 3 个主成分
3. 余弦相似度矩阵：检查点之间的痕迹方向一致性

所有图片保存到 outputs/figures/。
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from sklearn.decomposition import PCA
from transformers import GPT2LMHeadModel, GPT2Tokenizer

# 让 src/ 下的包可导入
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from rcid.circuit.patching import extract_causal_imprints
from rcid.data.ioi import IOIDataset

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

FIGURES_DIR = PROJECT_ROOT / "outputs" / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

# 全局绘图样式
plt.rcParams.update({
    "figure.dpi": 150,
    "savefig.dpi": 200,
    "savefig.bbox": "tight",
    "font.size": 10,
})


# ======================================================================
# Step 0: 数据准备
# ======================================================================

def prepare_data(
    n_samples: int = 100,
    seed: int = 42,
    device: str | None = None,
) -> tuple[
    GPT2LMHeadModel,
    GPT2Tokenizer,
    torch.Tensor,   # clean_ids  (N, seq_len)
    torch.Tensor,   # corrupt_ids (N, seq_len)
    list[str],       # token_labels
]:
    """加载模型和 IOI 数据集，返回统一长度的 batch。"""
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    logger.info("Loading GPT-2 and IOI dataset...")
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    model = GPT2LMHeadModel.from_pretrained("gpt2")
    model.to(device)
    model.eval()
    for p in model.parameters():
        p.requires_grad = False

    dataset = IOIDataset(n_samples=n_samples, tokenizer=tokenizer, seed=seed)

    # 用 DataLoader collate 成统一长度
    loader = dataset.to_dataloader(batch_size=n_samples, shuffle=False)
    batch = next(iter(loader))

    clean_ids = batch["clean_ids"].to(device)      # (N, seq_len)
    corrupt_ids = batch["corrupt_ids"].to(device)  # (N, seq_len)

    # 用第一个样本的 token 文本做 x 轴标签
    first_sample = dataset.samples[0]
    token_ids = first_sample.clean_ids.tolist()
    token_labels = [
        tokenizer.decode([tid]).replace("\n", "\\n")
        for tid in token_ids
    ]

    seq_len = clean_ids.shape[1]
    logger.info(
        "Data ready: %d samples, seq_len=%d, device=%s",
        n_samples, seq_len, device,
    )

    return model, tokenizer, clean_ids, corrupt_ids, token_labels


# ======================================================================
# Plot 1: 痕迹范数热力图
# ======================================================================

def plot_imprint_heatmap(
    model: GPT2LMHeadModel,
    clean_ids: torch.Tensor,     # (N, seq_len)
    corrupt_ids: torch.Tensor,   # (N, seq_len)
    token_labels: list[str],
) -> np.ndarray:
    """绘制 (layer x token_pos) 的平均痕迹范数热力图。

    Returns:
        norm_matrix: (n_layers, seq_len) 的范数矩阵，供后续使用。
    """
    n_layers = model.config.n_layer
    seq_len = clean_ids.shape[1]

    logger.info("Extracting imprints for full heatmap (%d x %d)...", n_layers, seq_len)

    # 构建全部检查点
    all_checkpoints = [(l, t) for l in range(n_layers) for t in range(seq_len)]

    imprints = extract_causal_imprints(
        model, clean_ids, corrupt_ids, all_checkpoints,
    )

    # 构建范数矩阵
    norm_matrix = np.zeros((n_layers, seq_len))  # (n_layers, seq_len)
    for (layer, token_pos), d in imprints.items():
        norm_matrix[layer, token_pos] = d.norm(dim=-1).mean().item()

    # 绘制
    fig, ax = plt.subplots(figsize=(max(12, seq_len * 0.6), 6))

    im = ax.imshow(
        norm_matrix,
        aspect="auto",
        cmap="YlOrRd",
        interpolation="nearest",
        origin="lower",
    )

    # 坐标轴
    ax.set_xlabel("Token Position", fontsize=12)
    ax.set_ylabel("Layer", fontsize=12)
    ax.set_title(
        "Causal Imprint Norms — IOI Task (GPT-2)\n"
        r"$\| d_{l,t} \|_2$ averaged over contrastive pairs",
        fontsize=13,
    )

    ax.set_xticks(range(seq_len))
    ax.set_xticklabels(token_labels, rotation=75, ha="right", fontsize=7)
    ax.set_yticks(range(n_layers))
    ax.set_yticklabels([f"L{i}" for i in range(n_layers)], fontsize=9)

    cbar = fig.colorbar(im, ax=ax, shrink=0.8, pad=0.02)
    cbar.set_label("Mean L2 Norm", fontsize=10)

    # 标注 top-5 位置
    flat_indices = np.argsort(norm_matrix.ravel())[::-1][:5]
    for rank, flat_idx in enumerate(flat_indices):
        l, t = divmod(flat_idx, seq_len)
        ax.plot(t, l, "k*", markersize=10)
        ax.annotate(
            f"#{rank + 1}",
            (t, l),
            textcoords="offset points",
            xytext=(5, 5),
            fontsize=7,
            fontweight="bold",
            color="black",
        )

    fig.tight_layout()
    path = FIGURES_DIR / "imprint_heatmap.png"
    fig.savefig(path)
    logger.info("Saved: %s", path)
    plt.close(fig)

    return norm_matrix


# ======================================================================
# Plot 2: Top-k 痕迹的 PCA
# ======================================================================

def plot_imprint_pca(
    model: GPT2LMHeadModel,
    clean_ids: torch.Tensor,     # (N, seq_len)
    corrupt_ids: torch.Tensor,   # (N, seq_len)
    norm_matrix: np.ndarray,     # (n_layers, seq_len)
    top_k: int = 5,
) -> dict[tuple[int, int], torch.Tensor]:
    """对 top-k 检查点的痕迹做 PCA，可视化前 3 个主成分。

    Returns:
        top_imprints: {(layer, token_pos): (N, d_model)} 的 top-k 痕迹。
    """
    n_layers, seq_len = norm_matrix.shape

    # 选 top-k 检查点
    flat_indices = np.argsort(norm_matrix.ravel())[::-1][:top_k]
    top_checkpoints = [divmod(int(idx), seq_len) for idx in flat_indices]

    logger.info("Top-%d checkpoints for PCA: %s", top_k, top_checkpoints)

    # 提取这些检查点的痕迹
    top_imprints = extract_causal_imprints(
        model, clean_ids, corrupt_ids, top_checkpoints,
    )

    # 拼接所有痕迹，做整体 PCA
    all_vectors = []    # list of (N, d_model)
    labels = []         # 每个向量属于哪个检查点
    for (l, t) in top_checkpoints:
        d = top_imprints[(l, t)].cpu().numpy()  # (N, d_model)
        all_vectors.append(d)
        labels.extend([f"L{l},P{t}"] * d.shape[0])

    all_vectors_np = np.concatenate(all_vectors, axis=0)  # (top_k * N, d_model)

    pca = PCA(n_components=3)
    projected = pca.fit_transform(all_vectors_np)  # (top_k * N, 3)

    explained = pca.explained_variance_ratio_

    # 3D 散点图
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")

    # 每个检查点不同颜色
    colors = plt.cm.tab10(np.linspace(0, 1, top_k))
    n_samples = clean_ids.shape[0]

    for i, (l, t) in enumerate(top_checkpoints):
        start = i * n_samples
        end = start + n_samples
        ax.scatter(
            projected[start:end, 0],
            projected[start:end, 1],
            projected[start:end, 2],
            c=[colors[i]],
            label=f"L{l},P{t}",
            alpha=0.5,
            s=15,
        )

    ax.set_xlabel(f"PC1 ({explained[0]:.1%})", fontsize=10)
    ax.set_ylabel(f"PC2 ({explained[1]:.1%})", fontsize=10)
    ax.set_zlabel(f"PC3 ({explained[2]:.1%})", fontsize=10)
    ax.set_title(
        f"PCA of Top-{top_k} Causal Imprints — IOI Task\n"
        f"Total variance explained: {sum(explained):.1%}",
        fontsize=12,
    )
    ax.legend(fontsize=8, loc="upper left")

    fig.tight_layout()
    path = FIGURES_DIR / "imprint_pca_3d.png"
    fig.savefig(path)
    logger.info("Saved: %s", path)
    plt.close(fig)

    # 补充 2D 子图（PC1 vs PC2, PC1 vs PC3）
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for i, (l, t) in enumerate(top_checkpoints):
        start = i * n_samples
        end = start + n_samples
        for ax_idx, (dim_x, dim_y, lx, ly) in enumerate([
            (0, 1, f"PC1 ({explained[0]:.1%})", f"PC2 ({explained[1]:.1%})"),
            (0, 2, f"PC1 ({explained[0]:.1%})", f"PC3 ({explained[2]:.1%})"),
        ]):
            axes[ax_idx].scatter(
                projected[start:end, dim_x],
                projected[start:end, dim_y],
                c=[colors[i]],
                label=f"L{l},P{t}" if ax_idx == 0 else None,
                alpha=0.5,
                s=15,
            )
            axes[ax_idx].set_xlabel(lx, fontsize=10)
            axes[ax_idx].set_ylabel(ly, fontsize=10)

    axes[0].legend(fontsize=8)
    axes[0].set_title("PC1 vs PC2", fontsize=11)
    axes[1].set_title("PC1 vs PC3", fontsize=11)
    fig.suptitle(
        f"PCA Projections — Top-{top_k} Checkpoints",
        fontsize=13, y=1.02,
    )

    fig.tight_layout()
    path = FIGURES_DIR / "imprint_pca_2d.png"
    fig.savefig(path)
    logger.info("Saved: %s", path)
    plt.close(fig)

    return top_imprints


# ======================================================================
# Plot 3: 检查点间余弦相似度矩阵
# ======================================================================

def plot_cosine_similarity(
    top_imprints: dict[tuple[int, int], torch.Tensor],
) -> None:
    """绘制 top 检查点之间的平均痕迹方向余弦相似度矩阵。"""
    keys = sorted(top_imprints.keys())
    n = len(keys)
    eps = 1e-8

    # 计算每个检查点的平均痕迹方向（归一化后的均值）
    mean_directions: list[torch.Tensor] = []
    for key in keys:
        d = top_imprints[key]                       # (N, d_model)
        d_mean = d.mean(dim=0)                      # (d_model,)
        d_norm = d_mean / (d_mean.norm() + eps)     # (d_model,)
        mean_directions.append(d_norm)

    # 构建相似度矩阵
    sim_matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            cos_sim = (mean_directions[i] @ mean_directions[j]).item()
            sim_matrix[i, j] = cos_sim

    # 绘制
    labels = [f"L{l},P{t}" for l, t in keys]

    fig, ax = plt.subplots(figsize=(8, 7))

    im = sns.heatmap(
        sim_matrix,
        xticklabels=labels,
        yticklabels=labels,
        annot=True,
        fmt=".2f",
        cmap="RdBu_r",
        center=0,
        vmin=-1,
        vmax=1,
        square=True,
        linewidths=0.5,
        ax=ax,
        cbar_kws={"label": "Cosine Similarity", "shrink": 0.8},
    )

    ax.set_title(
        "Cosine Similarity of Mean Imprint Directions\n"
        "Between Top Checkpoints — IOI Task",
        fontsize=12,
    )
    ax.set_xlabel("Checkpoint", fontsize=11)
    ax.set_ylabel("Checkpoint", fontsize=11)

    fig.tight_layout()
    path = FIGURES_DIR / "imprint_cosine_similarity.png"
    fig.savefig(path)
    logger.info("Saved: %s", path)
    plt.close(fig)

    # 同时计算逐样本的相似度（更精细）
    _plot_pairwise_sample_similarity(top_imprints, keys)


def _plot_pairwise_sample_similarity(
    top_imprints: dict[tuple[int, int], torch.Tensor],
    keys: list[tuple[int, int]],
) -> None:
    """绘制每对检查点之间，逐样本余弦相似度的分布。"""
    eps = 1e-8
    n = len(keys)

    if n < 2:
        return

    # 选前 6 对最有代表性的组合
    pairs = []
    for i in range(n):
        for j in range(i + 1, n):
            pairs.append((i, j))
    pairs = pairs[:6]  # 限制子图数量

    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    axes_flat = axes.flatten()

    for ax_idx, (i, j) in enumerate(pairs):
        if ax_idx >= len(axes_flat):
            break

        d_i = top_imprints[keys[i]]  # (N, d_model)
        d_j = top_imprints[keys[j]]  # (N, d_model)

        # 逐样本余弦相似度
        d_i_norm = d_i / (d_i.norm(dim=-1, keepdim=True) + eps)  # (N, d_model)
        d_j_norm = d_j / (d_j.norm(dim=-1, keepdim=True) + eps)  # (N, d_model)
        cos_sims = (d_i_norm * d_j_norm).sum(dim=-1).cpu().numpy()  # (N,)

        li, ti = keys[i]
        lj, tj = keys[j]

        axes_flat[ax_idx].hist(cos_sims, bins=30, alpha=0.7, color="steelblue", edgecolor="white")
        axes_flat[ax_idx].axvline(
            cos_sims.mean(), color="red", linestyle="--", linewidth=1.5,
            label=f"mean={cos_sims.mean():.2f}",
        )
        axes_flat[ax_idx].set_xlabel("Cosine Similarity", fontsize=9)
        axes_flat[ax_idx].set_ylabel("Count", fontsize=9)
        axes_flat[ax_idx].set_title(
            f"L{li},P{ti} vs L{lj},P{tj}", fontsize=10,
        )
        axes_flat[ax_idx].legend(fontsize=8)
        axes_flat[ax_idx].set_xlim(-1, 1)

    # 隐藏多余子图
    for ax_idx in range(len(pairs), len(axes_flat)):
        axes_flat[ax_idx].set_visible(False)

    fig.suptitle(
        "Per-Sample Cosine Similarity Distributions\nBetween Checkpoint Pairs",
        fontsize=13,
    )
    fig.tight_layout()
    path = FIGURES_DIR / "imprint_cosine_distributions.png"
    fig.savefig(path)
    logger.info("Saved: %s", path)
    plt.close(fig)


# ======================================================================
# Main
# ======================================================================

def main() -> None:
    """运行完整的可视化流水线。"""
    logger.info("=" * 60)
    logger.info("IOI Causal Imprint Visualization")
    logger.info("=" * 60)

    model, tokenizer, clean_ids, corrupt_ids, token_labels = prepare_data(
        n_samples=100,
        seed=42,
    )

    # Plot 1: 热力图
    logger.info("\n--- Plot 1: Imprint Norm Heatmap ---")
    norm_matrix = plot_imprint_heatmap(
        model, clean_ids, corrupt_ids, token_labels,
    )

    # Plot 2: PCA
    logger.info("\n--- Plot 2: PCA of Top-5 Imprints ---")
    top_imprints = plot_imprint_pca(
        model, clean_ids, corrupt_ids, norm_matrix, top_k=5,
    )

    # Plot 3: 余弦相似度
    logger.info("\n--- Plot 3: Cosine Similarity Matrix ---")
    plot_cosine_similarity(top_imprints)

    logger.info("\n" + "=" * 60)
    logger.info("All figures saved to: %s", FIGURES_DIR)
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
