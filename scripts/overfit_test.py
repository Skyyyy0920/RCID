"""RCID 过拟合测试：10 个样本，200 epoch，验证实现正确性。

期望结果：
- L_total → 0
- IOI accuracy → 100%
- 因果痕迹余弦相似度 → 1

如果无法过拟合，说明实现有 bug。
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from torch import nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from transformers import GPT2Config, GPT2LMHeadModel, GPT2Tokenizer
from tqdm import tqdm

# 确保 src 在 path 中
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from rcid.data.ioi import IOIDataset
from rcid.circuit.patching import extract_causal_imprints
from rcid.alignment.procrustes import procrustes_align
from rcid.distillation.rcid_loss import RCIDLoss
from rcid.distillation.baselines import StandardKDLoss

logging.basicConfig(level=logging.INFO, format="%(name)s - %(message)s")
logger = logging.getLogger("overfit_test")

DEVICE = "cpu"
N_SAMPLES = 10
N_EPOCHS = 3000
LR = 1e-2
LAMBDA_KL = 1.0
LAMBDA_RCID = 5.0
TEMPERATURE = 4.0
OUTPUT_DIR = Path(__file__).resolve().parent.parent / "outputs" / "figures"


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # ── 1. 加载教师与 tokenizer ───────────────────────────────────────
    logger.info("Loading teacher model (gpt2)...")
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    teacher = GPT2LMHeadModel.from_pretrained("gpt2")
    teacher.to(DEVICE)
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad = False

    d_T = teacher.config.n_embd       # 768
    n_T = teacher.config.n_layer      # 12

    # ── 2. 初始化学生模型 ─────────────────────────────────────────────
    logger.info("Initializing student model (4L, 384d)...")
    student_cfg = GPT2Config(
        vocab_size=teacher.config.vocab_size,
        n_positions=1024,
        n_embd=384,
        n_layer=4,
        n_head=6,
        n_inner=384 * 4,
    )
    student = GPT2LMHeadModel(student_cfg)
    student.to(DEVICE)
    student.train()

    d_S = student.config.n_embd       # 384
    n_S = student.config.n_layer      # 4

    # ── 3. 生成 IOI 对比数据（10 个样本）──────────────────────────────
    logger.info("Generating %d IOI contrastive pairs...", N_SAMPLES)
    ioi_dataset = IOIDataset(n_samples=N_SAMPLES, tokenizer=tokenizer, seed=42)

    # 收集 batch（一次全给，不 shuffle）
    loader = ioi_dataset.to_dataloader(batch_size=N_SAMPLES, shuffle=False)
    batch = next(iter(loader))
    clean_ids = batch["clean_ids"].to(DEVICE)        # (10, seq_len)
    corrupt_ids = batch["corrupt_ids"].to(DEVICE)    # (10, seq_len)
    answer_token_id = batch["answer_token_id"]       # (10,)
    seq_len = clean_ids.shape[1]

    logger.info("Data: clean_ids %s, seq_len=%d", clean_ids.shape, seq_len)

    # ── 4. 选取检查点 & 层映射 ────────────────────────────────────────
    # 使用 IOI 的最后一个 token 位置（模型预测答案的位置）
    last_pos = seq_len - 1
    # 选几个教师层（中间和末尾）
    teacher_checkpoint_layers = [3, 7, 11]
    checkpoints = [(l, last_pos) for l in teacher_checkpoint_layers]

    # 简单层映射：均匀映射 teacher → student
    # T12 → S4: {3→1, 7→2, 11→3}
    layer_mapping = {3: 1, 7: 2, 11: 3}

    logger.info("Checkpoints: %s", checkpoints)
    logger.info("Layer mapping: %s", layer_mapping)

    # ── 5. 预计算教师因果痕迹 ─────────────────────────────────────────
    logger.info("Extracting teacher causal imprints...")
    teacher_imprints = extract_causal_imprints(
        teacher, clean_ids, corrupt_ids, checkpoints,
    )
    for k, v in teacher_imprints.items():
        logger.info("  Teacher imprint %s: norm=%.4f", k, v.norm(dim=-1).mean().item())

    # ── 6. 计算 Procrustes 对齐矩阵 W* ──────────────────────────────
    # 需要同时拥有教师和学生在相同数据上的痕迹来对齐
    # 用初始学生来做校准
    logger.info("Computing Procrustes alignment W*...")
    # 学生检查点用映射后的学生层索引
    student_checkpoints = [
        (layer_mapping[l], t) for l, t in checkpoints
    ]
    student.eval()
    student_imprints_init = extract_causal_imprints(
        student, clean_ids, corrupt_ids, student_checkpoints,
    )
    student.train()

    # 收集所有教师痕迹和对应的学生痕迹，拼接后做 Procrustes
    all_teacher_vecs = []
    all_student_vecs = []
    for cp in checkpoints:
        t_l, t_pos = cp
        s_l = layer_mapping[t_l]
        s_cp = (s_l, t_pos)
        all_teacher_vecs.append(teacher_imprints[cp])           # (N, d_T)
        all_student_vecs.append(student_imprints_init[s_cp])    # (N, d_S)

    source = torch.cat(all_student_vecs, dim=0)   # (N*C, d_S)
    target = torch.cat(all_teacher_vecs, dim=0)    # (N*C, d_T)
    W_star = procrustes_align(source, target, center=True)  # (d_T, d_S)
    logger.info("W* shape: %s", W_star.shape)

    # ── 7. 构建损失函数 ──────────────────────────────────────────────
    kd_loss_fn = StandardKDLoss(temperature=TEMPERATURE)
    rcid_loss_fn = RCIDLoss(
        W=W_star,
        checkpoints=checkpoints,
        layer_mapping=layer_mapping,
    )

    # ── 8. 训练循环（手动，便于精细追踪）────────────────────────────────
    optimizer = AdamW(student.parameters(), lr=LR)

    # 预计算教师 logits（教师冻结，不随训练变化）
    with torch.no_grad():
        cached_teacher_logits = teacher(clean_ids).logits.detach()  # (B, S, V)
    logger.info("Cached teacher logits: %s", cached_teacher_logits.shape)

    # 追踪指标
    history = {
        "loss_total": [],
        "loss_kl": [],
        "loss_rcid": [],
        "ioi_accuracy": [],
        "cosine_similarity": [],
    }

    EVAL_EVERY = 100  # 每 100 epoch 做完整评估（含 extract_causal_imprints）

    logger.info("Starting training: %d epochs, lr=%.1e", N_EPOCHS, LR)
    for epoch in tqdm(range(N_EPOCHS), desc="Overfit"):
        student.train()
        optimizer.zero_grad()

        # ── KL 损失（用缓存的教师 logits）──
        s_logits = student(clean_ids).logits          # (B, S, V)
        kl_loss = kd_loss_fn(cached_teacher_logits, s_logits)

        # ── RCID 损失 ──
        rcid_loss = rcid_loss_fn(
            teacher_imprints, student, clean_ids, corrupt_ids,
        )

        # ── 总损失 ──
        total_loss = LAMBDA_KL * kl_loss + LAMBDA_RCID * rcid_loss
        total_loss.backward()

        nn.utils.clip_grad_norm_(student.parameters(), 1.0)
        optimizer.step()

        # ── 评估指标（每 EVAL_EVERY 个 epoch）──
        if epoch % EVAL_EVERY == 0 or epoch == N_EPOCHS - 1:
            student.eval()
            with torch.no_grad():
                eval_logits = student(clean_ids).logits   # (B, S, V)
                pred_logits = eval_logits[:, last_pos, :]  # (B, V)
                pred_tokens = pred_logits.argmax(dim=-1)   # (B,)
                correct = (pred_tokens == answer_token_id.to(DEVICE)).float().mean().item()

                student_imprints_eval = extract_causal_imprints(
                    student, clean_ids, corrupt_ids, student_checkpoints,
                )

                cosines = []
                for cp in checkpoints:
                    t_l, t_pos = cp
                    s_l = layer_mapping[t_l]
                    s_cp = (s_l, t_pos)
                    d_T_vec = teacher_imprints[cp]
                    d_S_vec = student_imprints_eval[s_cp]
                    d_S_aligned = d_S_vec @ W_star.T
                    cos = F.cosine_similarity(d_S_aligned, d_T_vec, dim=-1)
                    cosines.append(cos.mean().item())

                mean_cosine = sum(cosines) / len(cosines)

            history["loss_total"].append(total_loss.item())
            history["loss_kl"].append(kl_loss.item())
            history["loss_rcid"].append(rcid_loss.item())
            history["ioi_accuracy"].append(correct)
            history["cosine_similarity"].append(mean_cosine)
        else:
            # 轻量记录（只记损失，acc/cos 用上次的值）
            history["loss_total"].append(total_loss.item())
            history["loss_kl"].append(kl_loss.item())
            history["loss_rcid"].append(rcid_loss.item())
            history["ioi_accuracy"].append(history["ioi_accuracy"][-1] if history["ioi_accuracy"] else 0.0)
            history["cosine_similarity"].append(history["cosine_similarity"][-1] if history["cosine_similarity"] else 0.0)

        if (epoch + 1) % 200 == 0 or epoch == 0:
            logger.info(
                "Epoch %d: loss=%.4f (kl=%.4f, rcid=%.4f), "
                "acc=%.1f%%, cos=%.4f",
                epoch + 1, total_loss.item(), kl_loss.item(),
                rcid_loss.item(),
                history["ioi_accuracy"][-1] * 100,
                history["cosine_similarity"][-1],
            )

    # ── 9. 最终指标报告 ──────────────────────────────────────────────
    final_loss = history["loss_total"][-1]
    final_acc = history["ioi_accuracy"][-1]
    final_cos = history["cosine_similarity"][-1]

    logger.info("=" * 60)
    logger.info("OVERFIT TEST RESULTS")
    logger.info("=" * 60)
    logger.info("Final L_total:           %.6f", final_loss)
    logger.info("Final IOI accuracy:      %.1f%%", final_acc * 100)
    logger.info("Final cosine similarity: %.4f", final_cos)
    logger.info("=" * 60)

    # 判断是否通过
    passed = True
    if final_loss > 0.5:
        logger.warning("FAIL: L_total %.4f > 0.5", final_loss)
        passed = False
    if final_acc < 0.8:
        logger.warning("FAIL: IOI accuracy %.1f%% < 80%%", final_acc * 100)
        passed = False
    if final_cos < 0.8:
        logger.warning("FAIL: Cosine similarity %.4f < 0.8", final_cos)
        passed = False

    if passed:
        logger.info("OVERFIT TEST PASSED")
    else:
        logger.warning("OVERFIT TEST FAILED — possible implementation bug!")

    # ── 10. 绘图 ─────────────────────────────────────────────────────
    _plot_curves(history)
    logger.info("Figure saved to %s", OUTPUT_DIR / "overfit_test.png")


def _plot_curves(history: dict[str, list[float]]) -> None:
    """绘制 3 子图：损失曲线、IOI 准确率、余弦相似度。"""
    epochs = list(range(1, len(history["loss_total"]) + 1))

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # --- 损失曲线 ---
    ax = axes[0]
    ax.plot(epochs, history["loss_total"], label="L_total", linewidth=2)
    ax.plot(epochs, history["loss_kl"], label="L_KL", linewidth=1.5, alpha=0.8)
    ax.plot(epochs, history["loss_rcid"], label="L_RCID", linewidth=1.5, alpha=0.8)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("Training Loss Curves")
    ax.legend()
    ax.set_yscale("log")
    ax.grid(True, alpha=0.3)

    # --- IOI 准确率 ---
    ax = axes[1]
    ax.plot(epochs, [a * 100 for a in history["ioi_accuracy"]],
            linewidth=2, color="green")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Accuracy (%)")
    ax.set_title("IOI Task Accuracy")
    ax.set_ylim(-5, 105)
    ax.axhline(y=100, color="gray", linestyle="--", alpha=0.5)
    ax.grid(True, alpha=0.3)

    # --- 余弦相似度 ---
    ax = axes[2]
    ax.plot(epochs, history["cosine_similarity"],
            linewidth=2, color="orange")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Cosine Similarity")
    ax.set_title("Causal Imprint Alignment")
    ax.set_ylim(-0.1, 1.1)
    ax.axhline(y=1.0, color="gray", linestyle="--", alpha=0.5)
    ax.grid(True, alpha=0.3)

    fig.suptitle(
        f"RCID Overfit Test ({N_SAMPLES} samples, {N_EPOCHS} epochs)",
        fontsize=14, fontweight="bold",
    )
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "overfit_test.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
