# RCID 论文实验实施指南

> **版本**: 2026-03-16
> **目标**: 从零到论文完整结果的全流程操作手册
> **硬件**: 4 × A100 80GB（最低 1 × A100 可串行完成）

---

## 目录

1. [环境准备](#一环境准备)
2. [项目概览与方法清单](#二项目概览与方法清单)
3. [Pilot Validation（首先运行）](#三pilot-validation首先运行)
4. [实验 1：现有蒸馏方法保留了 teacher 的机制吗？](#四实验-1现有蒸馏方法保留了-teacher-的机制吗)
5. [实验 2：RCID 在受控环境下改善机制传递](#五实验-2rcid-在受控环境下改善机制传递)
6. [实验 3：大规模蒸馏 + RCID 正则（核心实用性实验）](#六实验-3大规模蒸馏--rcid-正则核心实用性实验)
7. [实验 4：OOD 鲁棒性](#七实验-4ood-鲁棒性)
8. [实验 5：机制分析 + 跨架构泛化](#八实验-5机制分析--跨架构泛化)
9. [GPU 调度策略](#九gpu-调度策略)
10. [超参数速查表](#十超参数速查表)
11. [输出目录结构](#十一输出目录结构)
12. [故障排查](#十二故障排查)
13. [论文图表对应关系](#十三论文图表对应关系)

---

## 一、环境准备

### 1.1 依赖安装

```bash
cd RCID
pip install -e ".[eval]"
```

核心依赖（`pyproject.toml`）：

| 包 | 版本 | 用途 |
|---|------|------|
| torch | ≥2.1 | 训练/推理 |
| transformers | ≥4.45 | HuggingFace 模型 |
| accelerate | ≥0.27 | 模型加载优化 |
| datasets | ≥2.16 | 数据集加载 |
| omegaconf | ≥2.3 | YAML 配置 |
| wandb | ≥0.16 | 实验追踪（可选） |
| scipy | ≥1.11 | Procrustes/SVD |
| scikit-learn | ≥1.3 | Information Purity 探针 |
| matplotlib/seaborn | ≥3.8/0.13 | 论文图表 |
| lm-eval | ≥0.4 | Benchmark 评估（可选） |

### 1.2 模型下载

```bash
# Qwen3（主实验）
huggingface-cli download Qwen/Qwen3-8B
huggingface-cli download Qwen/Qwen3-0.6B

# LLaMA 3（跨架构验证，需要 HF token）
huggingface-cli login
huggingface-cli download meta-llama/Llama-3.1-8B
huggingface-cli download meta-llama/Llama-3.2-1B
```

### 1.3 环境检查

```bash
python scripts/check_environment.py
```

验证项：CUDA 可用、模型可加载、数据集可下载、各模块可导入。

### 1.4 模型规格速查

| 属性 | Qwen3-8B (T) | Qwen3-0.6B (S) | LLaMA-3-8B (T) | LLaMA-3.2-1B (S) |
|------|-------------|----------------|----------------|-------------------|
| HF 名称 | `Qwen/Qwen3-8B` | `Qwen/Qwen3-0.6B` | `meta-llama/Llama-3.1-8B` | `meta-llama/Llama-3.2-1B` |
| 层数 | 36 | 28 | 32 | 16 |
| d_model | 4096 | 1024 | 4096 | 2048 |
| 词表 | 151,936 | 151,936 | 128,256 | 128,256 |
| 显存（fp16） | ~16 GB | ~1.2 GB | ~16 GB | ~2 GB |

---

## 二、项目概览与方法清单

### 2.1 论文核心叙事

> "标准蒸馏让学生**表现得像**教师（输出分布对齐），
> RCID 正则让学生**思考得像**教师（内部推理机制对齐），两者互补。"

### 2.2 完整方法清单

本项目实现了 8 种蒸馏方法，分为三类：

**基线 KL 方法（单数据流）：**

| 方法代号 | 全称 | 原理 |
|---------|------|------|
| `standard_kd` | Forward KL | 标准 KL(P_T \|\| P_S)，全序列 per-token |
| `reverse_kl` | Reverse KL | KL(P_S \|\| P_T)，mode-seeking |
| `standard_kd_akl` | AKL (Wu et al.) | 自适应混合 forward/reverse KL |
| `standard_kd_klr` | KL-Ratio | 基于 KL 比值的自适应权重 |

**RCID 方法（双数据流：主 KL + 对比对正则）：**

| 方法代号 | 全称 | 原理 |
|---------|------|------|
| `standard_kd_rcid` | RCID | KL + 因果检查点对比差值匹配 |
| `standard_kd_fitnets` | FitNets | KL + 全层表示匹配 |
| `standard_kd_informed_fitnets` | Informed FitNets | KL + 因果位置表示匹配 |

**SaGD 方法（单数据流 + 预计算教师显著性）：**

| 方法代号 | 全称 | 原理 |
|---------|------|------|
| `standard_kd_sagd` | SaGD | KL(加权) + 显著性对齐损失 |

### 2.3 两种训练器

| 训练器 | 文件 | 用途 | 数据类型 |
|--------|------|------|---------|
| `UnifiedTrainer` | `distillation/trainer.py` | Toy data 实验 (Exp 1-2) | ContrastiveDataset |
| `ScalableDistillationTrainer` | `distillation/scalable_trainer.py` | 大规模蒸馏 (Exp 3-5) | InstructionDataset |

### 2.4 损失函数清单

| 损失 | 文件 | 用途 |
|------|------|------|
| `StandardKDLoss` | `baselines.py` | 全序列 KL 蒸馏 |
| `FitNetsLoss` | `baselines.py` | 全层 MSE 表示匹配 |
| `InformedFitNetsLoss` | `baselines.py` | 因果位置 MSE 表示匹配 |
| `RCIDLoss` | `rcid_loss.py` | 因果检查点对比差值匹配（余弦距离） |
| `OCIDLoss` | `ocid_loss.py` | 输出层因果干预蒸馏 |
| `AKLLoss` | `adaptive_kl_losses.py` | 自适应 KL 混合 |
| `KLRatioLoss` | `adaptive_kl_losses.py` | KL 比值自适应 |
| `SaliencyAlignmentLoss` | `saliency.py` | 显著性余弦距离对齐 |

---

## 三、Pilot Validation（首先运行）

> **目标**：验证模型、数据、因果分析 pipeline 均可用。每种模型约 30 分钟。

### 3.1 运行命令

```bash
# Step 1: Qwen3（必须先通过）
python scripts/pilot_validation.py --model_family qwen3 --device cuda:0

# Step 2: LLaMA 3（Qwen3 通过后再运行）
python scripts/pilot_validation.py --model_family llama3 --device cuda:0
```

### 3.2 验证步骤与通过标准

| 步骤 | 内容 | 通过标准 | 失败处理 |
|------|------|---------|---------|
| 1 | 加载 teacher，IOI 准确率 | >95% | 检查模型路径/HF token |
| 2 | 单 token 名字池大小 | ≥20 个名字 | 检查 tokenizer 兼容性 |
| 3 | 各层对比差值范数分布 | 高层 > 底层 | 检查 IOI 模板是否正确 |
| 4 | 最佳检查点 activation patching | Δ > 0.01 | 检查 hook 注册逻辑 |
| 5 | Student baseline 准确率 | 仅记录，无阈值 | — |

### 3.3 预期输出示例

```
[PASS] Step 1: Teacher IOI accuracy = 97.2% (50 samples)
[PASS] Step 2: Name pool size = 34 single-token names
[PASS] Step 3: Layer norm gradient — top layers > bottom layers
         Layer  2: mean_norm = 0.012
         Layer 12: mean_norm = 0.045
         Layer 24: mean_norm = 0.189
         Layer 34: mean_norm = 0.342
[PASS] Step 4: Activation patching delta = 0.156 at (layer=30, pos=7)
[INFO] Step 5: Student baseline accuracy = 12.0% (random chance)
```

### 3.4 判定逻辑

- **Qwen3 步骤 1-4 全部 PASS** → 继续所有实验
- **Qwen3 任一步骤 FAIL** → **停止**，排查后重试
- **LLaMA 3 FAIL** → 可继续 Qwen3 实验，LLaMA 实验推迟

---

## 四、实验 1：现有蒸馏方法保留了 teacher 的机制吗？

> **论文主线 A 的核心证据。**
> **假说**：标准蒸馏产生的 student 即使准确率高，内部推理机制可能与 teacher 不一致。

### 4.1 实验设计

```
数据: IOI + Factual Probing (+ WinoGrande 可选)
方法: standard_kd (全序列 KL)
种子: 42, 123, 456
模型: Qwen3-8B → Qwen3-0.6B

流程:
  1. Teacher 上搜索因果检查点（Read）
  2. 用 StandardKD 蒸馏 student（3 seeds × 3 tasks = 9 runs）
  3. 每个 student 在每个检查点做因果干预（Write）
  4. 计算 causal_consistency = Pearson(Δ_T, Δ_S)
```

### 4.2 运行命令

```bash
# 方式 1: 使用实验脚本（推荐）
python scripts/run_exp1.py --model_family qwen3 --device cuda:0

# 方式 2: 使用 run_all.py 并行调度
python scripts/run_all.py --exp 1 --model_family qwen3 --gpus 0,1,2,3
```

### 4.3 训练超参数

```yaml
epochs: 20
batch_size: 16
lr: 5e-5
optimizer: AdamW
weight_decay: 0.01
scheduler: cosine
warmup_ratio: 0.05
grad_clip: 1.0
temperature: 2.0
kl_mode: sequence    # 全序列 KL（强制）
fp16: true
```

### 4.4 评估指标

| 指标 | 计算方式 | 预期结果 |
|------|---------|---------|
| Task Accuracy | student 在 clean 输入上的正确率 | 接近 teacher（>80%） |
| Causal Consistency (CC) | Pearson(Δ_T, Δ_S) 在所有检查点上 | **低**（<0.5） |
| Logit Diff Mean | logit(correct) - logit(wrong) 的均值 | 正值 |

### 4.5 预期结论

> StandardKD 的 student 准确率接近 teacher（>80%），
> 但因果一致性显著低于 1.0（CC < 0.5），
> 证明输出匹配不等于机制匹配。

### 4.6 输出位置

```
outputs/results/exp1/{qwen3|llama3}/
├── standard_kd/
│   ├── seed_42/
│   │   ├── ioi/
│   │   │   ├── student_checkpoint.pt
│   │   │   ├── training_log.json      # 训练历史
│   │   │   ├── eval_accuracy.json     # task accuracy
│   │   │   └── causal_consistency.json # CC 分数
│   │   ├── factual/
│   │   └── winogrande/
│   ├── seed_123/
│   └── seed_456/
└── teacher_checkpoints.json  # 因果检查点列表
```

---

## 五、实验 2：RCID 在受控环境下改善机制传递

> **论文主线 B 的核心证据。**
> **假说**：RCID 通过在因果位置匹配对比差值（而非完整表示），更有效地传递 teacher 的推理机制。

### 5.1 实验设计

```
数据: IOI + Factual Probing + WinoGrande
方法: standard_kd, fitnets, informed_fitnets, rcid (共 4 种)
种子: 42, 123, 456
总 runs: 4 methods × 3 tasks × 3 seeds = 36

因素拆解:
  FitNets → Informed FitNets = "选对位置"的贡献
  Informed FitNets → RCID = "匹配对比差值 vs 完整表示"的贡献
```

### 5.2 运行命令

```bash
# 完整实验 2（36 runs）
python scripts/run_exp2.py --model_family qwen3 --device cuda:0

# 并行调度（4 GPUs）
python scripts/run_all.py --exp 2 --model_family qwen3 --gpus 0,1,2,3
```

### 5.3 各方法损失函数配置

| 方法 | 主损失 | 正则项 | 匹配目标 | 匹配位置 |
|------|--------|--------|---------|---------|
| standard_kd | 全序列 KL | — | — | — |
| fitnets | 全序列 KL | MSE | h_T (完整表示) | 全层映射 |
| informed_fitnets | 全序列 KL | MSE | h_T_clean (完整表示) | 因果检查点 |
| rcid | 全序列 KL | 余弦距离 | d_T = h_clean - h_corrupt (对比差值) | 因果检查点 |

### 5.4 RCID 训练流程详解

```
每个 training step:
  1. Teacher forward (no_grad):
     - clean_input  → t_logits, t_clean_residuals
     - corrupt_input → t_corrupt_residuals
     - d_T = t_clean - t_corrupt (对比差值, detached)

  2. Student forward (with grad):
     - clean_input  → s_logits, s_clean_residuals
     - corrupt_input → s_corrupt_residuals
     - d_S = s_clean - s_corrupt

  3. 损失计算:
     - L_KL = StandardKDLoss(t_logits, s_logits, mask=attention_mask)
     - L_RCID = mean over checkpoints:
         aligned = d_S @ W.T  (Procrustes 投影)
         L = ||normalize(aligned) - normalize(d_T)||^2
     - total = L_KL + λ_RCID * L_RCID

  4. backward() + optimizer.step()
```

### 5.5 前置步骤：检查点搜索 + 对齐矩阵

```python
# 伪代码：在蒸馏前执行
from rcid.circuit.patching import extract_contrastive_differences
from rcid.circuit.checkpoint_selection import select_checkpoints
from rcid.alignment.cka import cka_matrix
from rcid.alignment.layer_matching import match_layers
from rcid.alignment.procrustes import compute_procrustes_matrices

# 1. 提取各层对比差值（Read 操作）
teacher_diffs = extract_contrastive_differences(
    teacher, adapter, clean_ids, corrupt_ids, layers=range(36))

# 2. 搜索因果检查点
checkpoints = select_checkpoints(
    teacher_diffs, dataset, top_k=10, diversity_ratio=0.5)
# → [(layer, token_pos), ...] e.g. [(30, 7), (28, 3), ...]

# 3. CKA 层匹配
teacher_reps = extract_residuals(teacher, ...)  # 所有层
student_reps = extract_residuals(student, ...)
cka_scores = cka_matrix(teacher_reps, student_reps)
layer_mapping = match_layers(cka_scores, strategy="greedy")
# → {30: 22, 28: 20, ...}

# 4. Procrustes 对齐矩阵
W_matrices = compute_procrustes_matrices(
    teacher_diffs, student_diffs, layer_mapping)
# → {30: W(4096, 1024), 28: W(4096, 1024), ...}
```

### 5.6 评估指标

| 指标 | 计算方式 | 论文位置 |
|------|---------|---------|
| Task Accuracy | student 正确率 | Table 1 |
| Causal Consistency (CC) | Pearson(Δ_T, Δ_S) 均值 | Table 1 |
| Information Purity | 探针准确率(任务标签) - 探针准确率(控制标签) | Table 1 |

### 5.7 预期结果

| 方法 | Accuracy | CC | Info Purity |
|------|----------|-----|-------------|
| StandardKD | 高 | 低 | 低 |
| FitNets | 高 | 中 | 中 |
| Informed FitNets | 高 | 中-高 | 中 |
| **RCID** | **高** | **最高** | **最高** |

### 5.8 输出位置

```
outputs/results/exp2/{qwen3}/
├── standard_kd/seed_42/{ioi,factual,winogrande}/
├── fitnets/seed_42/{ioi,factual,winogrande}/
├── informed_fitnets/seed_42/{ioi,factual,winogrande}/
├── rcid/seed_42/{ioi,factual,winogrande}/
└── summary_table.json  # 汇总所有方法的指标
```

---

## 六、实验 3：大规模蒸馏 + RCID 正则（核心实用性实验）

> **论文主线 C 的核心证据。**
> **假说**：将 RCID 作为正则项嫁接到大规模 KL 蒸馏上，可以在真实 benchmark 上带来可测量的提升。

### 6.1 实验设计

```
数据:
  主数据流: Dolly-15K（约 12.8K 训练样本）
  RCID 数据流: 自动构造 ~5K 对比对
  SaGD 数据: 预计算 teacher 显著性

方法对比:
  standard_kd                  — 纯 KL
  reverse_kl                   — 反向 KL
  standard_kd_akl              — AKL 自适应
  standard_kd_klr              — KL-Ratio 自适应
  standard_kd_rcid             — KL + RCID 正则
  standard_kd_fitnets          — KL + FitNets 正则
  standard_kd_informed_fitnets — KL + InformedFitNets 正则
  standard_kd_sagd             — KL(加权) + 显著性对齐

种子: 42, 123, 456
```

### 6.2 前置准备（按顺序执行）

#### Step 1: 生成对比对（RCID 方法需要）

```bash
python scripts/generate_contrastive_pairs.py \
    --model_name Qwen/Qwen3-8B \
    --output_dir data/contrastive_pairs/ \
    --max_entity_pairs 2500 \
    --max_number_pairs 2500 \
    --device cuda:0 \
    --seed 42
```

**输出**：

```
data/contrastive_pairs/
├── entity_swap.json         # ~2500 对
├── number_perturb.json      # ~2500 对
└── generation_summary.json  # 元数据
```

**耗时**：约 30-60 分钟（取决于 teacher 推理速度）

#### Step 2: 预计算 teacher 显著性（SaGD 方法需要）

```bash
python scripts/precompute_teacher_saliency.py \
    --model_family qwen3 \
    --data_source "databricks/databricks-dolly-15k" \
    --max_seq_len 512 \
    --batch_size 4 \
    --saliency_temperature 2.0 \
    --output_path data/teacher_saliency_qwen3.pt \
    --device cuda:0
```

**输出格式**：

```python
# data/teacher_saliency_qwen3.pt
{
    "saliency": [Tensor(L_0,), Tensor(L_1,), ...],  # 每个样本一个向量
    "metadata": {
        "model": "Qwen/Qwen3-8B",
        "dataset": "databricks/databricks-dolly-15k",
        "n_samples": 12859,
        "saliency_temperature": 2.0,
        "max_seq_len": 512,
    }
}
```

**耗时**：约 2-4 小时（需要对每个样本做一次 teacher forward + backward）

> **关键**：预计算时的 `data_source`、`max_seq_len`、`tokenizer` 必须与训练时完全一致，
> 否则 index → saliency 的映射会静默错位。

#### Step 3: 搜索因果检查点 + 计算 Procrustes 矩阵（RCID/FitNets 方法需要）

这一步在 `run_large_scale_distill.py` 中会自动执行（如果 `--method` 需要）。
也可以手动预计算以节省重复开销。

### 6.3 运行大规模蒸馏

#### 方式 1: 逐个运行（最可控）

```bash
# Standard KD 基线
python scripts/run_large_scale_distill.py \
    --model_family qwen3 \
    --method standard_kd \
    --epochs 3 \
    --batch_size 8 \
    --gradient_accumulation 4 \
    --lr 2e-5 \
    --max_seq_len 512 \
    --temperature 2.0 \
    --seed 42 \
    --output_dir outputs/large_scale \
    --device cuda:0

# Reverse KL
python scripts/run_large_scale_distill.py \
    --model_family qwen3 \
    --method reverse_kl \
    --seed 42 \
    --device cuda:1

# AKL
python scripts/run_large_scale_distill.py \
    --model_family qwen3 \
    --method standard_kd_akl \
    --akl_mu 0.5 \
    --seed 42 \
    --device cuda:2

# KL-Ratio (token-level)
python scripts/run_large_scale_distill.py \
    --model_family qwen3 \
    --method standard_kd_klr \
    --klr_granularity token \
    --seed 42 \
    --device cuda:3
```

```bash
# RCID
python scripts/run_large_scale_distill.py \
    --model_family qwen3 \
    --method standard_kd_rcid \
    --contrastive_pairs_path data/contrastive_pairs/ \
    --lambda_rcid 0.1 \
    --rcid_every_n_steps 5 \
    --top_k_checkpoints 10 \
    --seed 42 \
    --device cuda:0

# SaGD
python scripts/run_large_scale_distill.py \
    --model_family qwen3 \
    --method standard_kd_sagd \
    --teacher_saliency_path data/teacher_saliency_qwen3.pt \
    --lambda_sal 0.5 \
    --sagd_every_n_steps 5 \
    --sagd_tau_w 1.0 \
    --seed 42 \
    --device cuda:1
```

#### 方式 2: 使用 paper experiments 脚本

```bash
# 运行已命名的实验配置
python scripts/run_paper_experiments.py --experiment forward_kl --device cuda:0
python scripts/run_paper_experiments.py --experiment reverse_kl --device cuda:1
python scripts/run_paper_experiments.py --experiment akl --device cuda:2
python scripts/run_paper_experiments.py --experiment klr_token --device cuda:3

# SaGD 系列
python scripts/run_paper_experiments.py --experiment sagd --device cuda:0
python scripts/run_paper_experiments.py --experiment sagd_loss_only --device cuda:1
python scripts/run_paper_experiments.py --experiment sagd_reweight_only --device cuda:2
```

#### 方式 3: 批量并行运行

```bash
# 使用 shell 脚本一键运行
bash scripts/run_all_experiments.sh

# 或分阶段运行
bash scripts/run_phase2.sh cuda:0
```

### 6.4 训练超参数

```yaml
large_scale_training:
  epochs: 3
  batch_size: 8                      # per GPU
  gradient_accumulation_steps: 4     # effective batch = 32
  lr: 2e-5
  weight_decay: 0.01
  warmup_ratio: 0.03
  max_grad_norm: 1.0
  max_seq_len: 512
  temperature: 2.0
  fp16: true
  save_every_n_epochs: 1

  # RCID 正则
  lambda_rcid: 0.1                   # grid search: [0.01, 0.05, 0.1, 0.5, 1.0]
  rcid_every_n_steps: 5             # 每 5 步计算一次 RCID 损失

  # SaGD
  lambda_sal: 0.5                    # 显著性对齐损失权重
  sagd_every_n_steps: 5             # 每 5 步做一次显著性重加权
  sagd_tau_w: 1.0                    # 权重温度
  saliency_temperature: 2.0         # 显著性 → 分布的温度
```

### 6.5 λ_RCID 超参数搜索

```bash
# 逐个搜索 λ 值（优先级从高到低）
for lambda in 0.1 0.05 0.5 0.01 1.0; do
    python scripts/run_large_scale_distill.py \
        --model_family qwen3 \
        --method standard_kd_rcid \
        --contrastive_pairs_path data/contrastive_pairs/ \
        --lambda_rcid $lambda \
        --seed 42 \
        --output_dir outputs/large_scale/lambda_search/lambda_${lambda} \
        --device cuda:0
done
```

### 6.6 ROUGE-L 评估（训练后自动执行）

训练结束后，脚本自动在 Dolly 验证集和测试集上计算 ROUGE-L：

```
eval_results.json:
{
    "rouge_l_f": 0.234,     # F1
    "rouge_l_p": 0.287,     # Precision
    "rouge_l_r": 0.221,     # Recall
    "num_samples": 500,
    "num_empty_generations": 3
}
```

手动重新评估：

```bash
python scripts/run_paper_experiments.py \
    --experiment forward_kl \
    --eval_only \
    --device cuda:0
```

### 6.7 Benchmark 评估（MMLU/GSM8K/ARC 等）

```bash
# 评估蒸馏后的 checkpoint
python scripts/eval_benchmarks.py \
    --model_path outputs/large_scale/qwen3/standard_kd/seed_42/student_final.pt \
    --model_name Qwen/Qwen3-0.6B \
    --benchmarks mmlu,gsm8k,arc_challenge,hellaswag,winogrande,truthfulqa_mc2 \
    --batch_size 8 \
    --device cuda:0

# 评估 base student（未蒸馏基线）
python scripts/eval_benchmarks.py \
    --model_name Qwen/Qwen3-0.6B \
    --benchmarks mmlu,gsm8k,arc_challenge \
    --device cuda:0
```

Benchmark 配置：

| Benchmark | Few-shot | 评估方式 |
|-----------|----------|---------|
| MMLU | 5-shot | 多选准确率 |
| GSM8K | 8-shot | 精确匹配 |
| ARC-Challenge | 25-shot | 多选准确率 |
| HellaSwag | 10-shot | 多选准确率 |
| WinoGrande | 5-shot | 多选准确率 |
| TruthfulQA | 0-shot | MC2 分数 |

### 6.8 SaGD 消融实验

三个变体验证 SaGD 两个组件的贡献：

| 配置名 | 重加权 | L_sal | tau_w | lambda_sal | 目的 |
|--------|--------|-------|-------|-----------|------|
| `sagd` | ✅ | ✅ | 1.0 | 0.5 | 完整方法 |
| `sagd_loss_only` | ≈关闭 | ✅ | 100.0 | 0.5 | 隔离 L_sal 贡献 |
| `sagd_reweight_only` | ✅ | ❌ | 0.1 | 0.0 | 隔离重加权贡献 |

运行消融：

```bash
python scripts/run_paper_experiments.py --experiment sagd --device cuda:0
python scripts/run_paper_experiments.py --experiment sagd_loss_only --device cuda:1
python scripts/run_paper_experiments.py --experiment sagd_reweight_only --device cuda:2
```

### 6.9 预期结果

| 方法 | ROUGE-L | MMLU | GSM8K | CC (on contrastive subset) |
|------|---------|------|-------|---------------------------|
| Standard KD | 基线 | 基线 | 基线 | 低 |
| Reverse KL | ≈基线 | ≈基线 | ≈基线 | 低 |
| AKL | ≈基线 | ≈基线 | ≈基线 | 低 |
| KL-Ratio | ≈基线或略优 | ≈基线 | ≈基线 | 低 |
| **KD + RCID** | **≈基线或略优** | **≈基线** | **可能略优** | **显著更高** |
| KD + SaGD | ≈基线或略优 | ≈基线 | ≈基线 | 中 |

> **关键论证**：即使 benchmark 提升不显著，CC 的提升也有价值——
> 证明 RCID 确实改善了内部机制对齐，这对可解释性和鲁棒性有意义。

### 6.10 输出位置

```
outputs/large_scale/qwen3/{method}/seed_{seed}/
├── config.json
├── student_final.pt
├── student_epoch1.pt / student_epoch2.pt  # 中间 checkpoint
├── training_log.json
├── training_stats.jsonl
├── eval_results.json      # ROUGE-L
└── test_generations.json  # 生成样本
```

---

## 七、实验 4：OOD 鲁棒性

> **假说**：保留 teacher 推理机制的 student 在分布外数据上更鲁棒。

### 7.1 实验设计

```
数据: 收集实验 2+3 中所有 student
评估:
  1. 在 Dolly 训练分布上计算准确率 (ID)
  2. 在 OOD 变体上计算准确率 (OOD)
  3. degradation = ID_accuracy - OOD_accuracy
  4. 绘制 (CC, degradation) 散点图

OOD 变体:
  IOI: unseen_names, different_templates, longer_context
  Factual: different_relations, different_entities
  WinoGrande: longer_sentences, multiple_distractors
```

### 7.2 运行命令

```bash
python scripts/run_exp3.py --model_family qwen3 --device cuda:0
```

### 7.3 预期结果

- CC 与 OOD degradation 呈负相关（Pearson r < -0.5）
- 即：因果一致性越高 → OOD 性能下降越少 → 更鲁棒

### 7.4 输出位置

```
outputs/results/exp3/
├── ood_results.json       # 每个 student 的 (CC, OOD_degradation)
└── scatter_plot.png       # CC vs degradation 散点图
```

---

## 八、实验 5：机制分析 + 跨架构泛化

> **论文主线 D 的核心证据。**

### 8.1 Part A: 信息纯度分析

```
目的: 验证对比差值 (d_T) 比完整表示 (h_T) 包含更纯净的任务相关信息

方法:
  1. 在 teacher 的因果检查点提取 d_T 和 h_T
  2. 训练 LogisticRegression 探针：
     - 任务标签: 预测正确答案 (task accuracy)
     - 控制标签: 预测随机标签 (control accuracy)
  3. selectivity = task_accuracy - control_accuracy

预期: d_T 的 selectivity > h_T 的 selectivity
```

运行：

```bash
python scripts/run_exp4.py --model_family qwen3 --device cuda:0
```

### 8.2 Part B: 大规模蒸馏模型的因果分析

```
目的: 验证实验 3 中 RCID 正则确实改善了内部机制对齐

方法:
  1. 加载实验 3 中各方法的最佳 student checkpoint
  2. 在对比对子集上计算因果一致性
  3. 计算信息纯度

预期: KD+RCID 的 CC 和 Info Purity 显著优于纯 KD
```

### 8.3 Part C: 跨架构验证（LLaMA 3）

```
目的: 证明 RCID 不依赖于特定架构

模型: LLaMA-3-8B → LLaMA-3.2-1B
方法: standard_kd, rcid (仅核心对比)
任务: IOI, Factual Probing
种子: 42, 123, 456
总 runs: 2 × 2 × 3 = 12
```

运行：

```bash
# LLaMA 3 Pilot
python scripts/pilot_validation.py --model_family llama3 --device cuda:0

# 跨架构实验
python scripts/run_exp5_cross_arch.py --device cuda:0

# 或使用 run_all.py
python scripts/run_all.py --exp 5 --model_family llama3 --gpus 0,1
```

### 8.4 输出位置

```
outputs/results/exp4/
├── information_purity.json    # d_T vs h_T 的 selectivity 对比
└── bar_chart.png

outputs/results/exp5_cross_arch/
├── llama3/
│   ├── standard_kd/seed_42/{ioi,factual}/
│   └── rcid/seed_42/{ioi,factual}/
└── cross_arch_table.json      # Qwen3 vs LLaMA3 结果对比
```

---

## 九、GPU 调度策略

### 9.1 Toy Data 实验（实验 1+2, Qwen3）

```
时间估算: 每个 run 约 20-30 分钟
总 runs: 36 (实验 2)
4 GPU 并行: ~5 小时

GPU 分配:
  GPU 0: seed=42,  method=standard_kd → fitnets (6 runs)
  GPU 1: seed=42,  method=informed_fitnets → rcid (6 runs)
  GPU 2: seed=123, method=standard_kd → fitnets (6 runs)
  GPU 3: seed=123, method=informed_fitnets → rcid (6 runs)
  (然后 seed=456 的 12 runs)
```

### 9.2 大规模蒸馏（实验 3, Qwen3）

```
时间估算: 每个 run 约 3-5 小时（3 epochs on Dolly-15K）
总 runs: 8 methods × 3 seeds = 24 (或先跑 seed=42 的 8 个)
4 GPU 并行: ~6-8 小时 per seed

推荐执行顺序:
  Round 1 (seed=42):
    GPU 0: standard_kd
    GPU 1: standard_kd_rcid (lambda=0.1)
    GPU 2: standard_kd_sagd
    GPU 3: reverse_kl

  Round 2 (seed=42):
    GPU 0: standard_kd_akl
    GPU 1: standard_kd_klr
    GPU 2: standard_kd_fitnets
    GPU 3: standard_kd_informed_fitnets

  Round 3-6: seed=123, 456 重复
```

### 9.3 λ_RCID 搜索

```
时间估算: 5 个 λ 值 × 3-5 小时 = 15-25 小时（串行）
4 GPU 并行: ~4-7 小时

  GPU 0: lambda=0.01
  GPU 1: lambda=0.05
  GPU 2: lambda=0.1 (默认)
  GPU 3: lambda=0.5
  (然后) GPU 0: lambda=1.0
```

### 9.4 跨架构验证（实验 5, LLaMA 3）

```
时间估算: 12 runs × 20-30 分钟 = 4-6 小时
4 GPU 并行: ~2 小时
```

### 9.5 显存估算

| 方法 | Teacher (fp16) | Student (fp32) | 其他 | 总计 |
|------|---------------|----------------|------|------|
| standard_kd | ~16 GB | ~2.4 GB | ~5 GB | ~24 GB |
| standard_kd_rcid | ~16 GB | ~2.4 GB | ~8 GB (hooks) | ~27 GB |
| standard_kd_sagd | ~16 GB | ~2.4 GB + 2.4 GB (2次forward) | ~5 GB | ~26 GB |

所有方法均可在单张 A100 80GB 上运行。

---

## 十、超参数速查表

### 10.1 Toy Data 训练（实验 1+2）

| 参数 | 值 | 备注 |
|------|-----|------|
| epochs | 20 | |
| batch_size | 16 | |
| lr | 5e-5 | |
| optimizer | AdamW | |
| weight_decay | 0.01 | |
| scheduler | cosine | |
| warmup_ratio | 0.05 | |
| grad_clip | 1.0 | |
| temperature | 2.0 | KL 温度 |
| kl_mode | sequence | 全序列 KL |
| lambda_rcid | 1.0 | RCID 正则权重 |
| fp16 | true | |

### 10.2 大规模蒸馏训练（实验 3）

| 参数 | 值 | 备注 |
|------|-----|------|
| epochs | 3 | |
| batch_size | 8 | per GPU |
| gradient_accumulation | 4 | effective batch = 32 |
| lr | 2e-5 | |
| weight_decay | 0.01 | |
| warmup_ratio | 0.03 | |
| max_grad_norm | 1.0 | |
| max_seq_len | 512 | |
| temperature | 2.0 | |
| fp16 | true | |

### 10.3 方法特有参数

| 参数 | 适用方法 | 默认值 | 搜索范围 |
|------|---------|--------|---------|
| lambda_rcid | RCID/FitNets/InformedFitNets | 0.1 | [0.01, 0.05, 0.1, 0.5, 1.0] |
| rcid_every_n_steps | RCID 方法 | 5 | — |
| top_k_checkpoints | RCID 方法 | 10 | — |
| diversity_ratio | 检查点选择 | 0.5 | — |
| lambda_sal | SaGD | 0.5 | — |
| sagd_tau_w | SaGD | 1.0 | [0.1, 1.0, 100.0] (消融) |
| sagd_every_n_steps | SaGD | 5 | [1, 5] |
| saliency_temperature | SaGD | 2.0 | — |
| akl_mu | AKL | 0.5 | — |
| klr_granularity | KL-Ratio | token | [token, batch] |
| klr_beta | KL-Ratio (batch) | 0.99 | [0.0, 0.9, 0.95, 0.99, 0.999] |

### 10.4 检查点选择参数

| 参数 | 值 | 含义 |
|------|-----|------|
| top_k | 10 | 选择 top-10 因果检查点 |
| diversity_ratio | 0.5 | 50% 来自未修改位置，50% 来自修改位置 |
| min_layer | n_layers // 3 | 只考虑高层（底层差异太 trivial） |

---

## 十一、输出目录结构

```
RCID/
├── data/                                      # 预处理数据
│   ├── contrastive_pairs/                     # 自动生成的对比对
│   │   ├── entity_swap.json
│   │   ├── number_perturb.json
│   │   └── generation_summary.json
│   ├── teacher_saliency_qwen3.pt             # Qwen3 teacher 显著性缓存
│   └── teacher_saliency_llama3.pt            # LLaMA3 teacher 显著性缓存
│
├── outputs/
│   ├── results/                               # Toy data 实验结果
│   │   ├── exp1/qwen3/                        # 实验 1
│   │   ├── exp2/qwen3/                        # 实验 2
│   │   ├── exp3/                              # 实验 4 (OOD)
│   │   ├── exp4/                              # 实验 5a (Info Purity)
│   │   └── exp5_cross_arch/llama3/            # 实验 5c (跨架构)
│   │
│   ├── large_scale/                           # 大规模蒸馏结果
│   │   ├── qwen3/
│   │   │   ├── standard_kd/seed_42/
│   │   │   │   ├── config.json
│   │   │   │   ├── student_final.pt
│   │   │   │   ├── student_epoch{1,2,3}.pt
│   │   │   │   ├── training_log.json
│   │   │   │   ├── training_stats.jsonl
│   │   │   │   ├── eval_results.json
│   │   │   │   └── test_generations.json
│   │   │   ├── standard_kd_rcid/seed_42/
│   │   │   ├── standard_kd_sagd/seed_42/
│   │   │   └── .../
│   │   └── llama3/
│   │
│   ├── paper/                                 # Paper experiments 输出
│   │   ├── forward_kl/
│   │   ├── sagd/
│   │   └── .../
│   │
│   └── large_scale/lambda_search/             # λ_RCID 搜索
│       ├── lambda_0.01/
│       ├── lambda_0.05/
│       ├── lambda_0.1/
│       ├── lambda_0.5/
│       └── lambda_1.0/
```

---

## 十二、故障排查

### 12.1 常见问题

| 问题 | 原因 | 解决方案 |
|------|------|---------|
| CUDA OOM | batch 太大 / 显存不足 | 减小 `--batch_size`，增大 `--gradient_accumulation` |
| SaGD 方法报错 | 未提供 teacher saliency 路径 | 先运行 `precompute_teacher_saliency.py` |
| RCID 对齐失败 | 对比对文件不存在 | 先运行 `generate_contrastive_pairs.py` |
| Saliency 缓存不匹配 | 预计算时参数不一致 | 确保 `data_source`、`max_seq_len`、tokenizer 完全一致 |
| ROUGE-L 评估缺失 | 数据集无 val/test split | 检查 Dolly split 是否正确生成 |
| LLaMA 3 加载失败 | 缺少 HF 访问权限 | `huggingface-cli login` 并申请 Llama 访问权限 |
| FP16 softmax 溢出 | Qwen3 词表过大 (151K) | 代码已处理：loss 计算中 `.float()` 上转 |
| 训练 loss NaN | 学习率过大 / 梯度爆炸 | 降低 `--lr`，检查 `--max_grad_norm` |
| CC = NaN | 某个检查点 Δ 全为零 | `causal_consistency.py` 已处理退化情况 |

### 12.2 训练监控

```bash
# 实时查看训练日志
tail -f outputs/large_scale/qwen3/standard_kd/seed_42/training_stats.jsonl | python -m json.tool

# 关键指标:
# - loss: 应稳定下降
# - lr: 应符合 cosine schedule
# - sagd/jsd: SaGD 方法中 JSD 均值（应 > 0）
# - sagd/max_weight: 最大样本权重（不应远 > 10）
# - sagd/cos_sim: 显著性余弦相似度（应逐渐接近 1）
# - rcid_loss: RCID 损失（应下降）
```

### 12.3 Checkpoint 恢复

当前实现支持 epoch 级 checkpoint（`student_epoch{N}.pt`），但不支持 step 级恢复。
如果训练中断：

```bash
# 手动加载最近 checkpoint 继续训练
# 需修改脚本加载 student_epoch2.pt 并设置 epochs=1（只跑剩余 epoch）
```

---

## 十三、论文图表对应关系

### 13.1 主要表格

| 论文位置 | 内容 | 数据来源 |
|---------|------|---------|
| Table 1 | 实验 2 结果：4 方法 × 3 任务的 Accuracy / CC / Info Purity | `outputs/results/exp2/` |
| Table 2 | 实验 3 结果：大规模蒸馏 Benchmark 分数 | `outputs/large_scale/` + `eval_benchmarks.py` |
| Table 3 | 实验 5 结果：跨架构（LLaMA 3）对比 | `outputs/results/exp5_cross_arch/` |
| Table A1 | λ_RCID 搜索结果 | `outputs/large_scale/lambda_search/` |
| Table A2 | SaGD 消融结果 | `outputs/paper/sagd*/` |

### 13.2 主要图表

| 论文位置 | 内容 | 数据来源 | 生成脚本 |
|---------|------|---------|---------|
| Fig 1 | 因果一致性：StandardKD 低 CC 示意图 | 实验 1 | `visualization/paper_figures.py` |
| Fig 2 | CKA 层匹配热力图 | 实验 2 前置步骤 | `visualization/paper_figures.py` |
| Fig 3 | CC vs OOD degradation 散点图 | 实验 4 | `visualization/paper_figures.py` |
| Fig 4 | Info Purity: d_T vs h_T 对比柱状图 | 实验 5a | `visualization/paper_figures.py` |
| Fig A1 | 训练曲线（loss, CC per epoch） | 实验 2+3 | `visualization/paper_figures.py` |

### 13.3 生成论文图表

```bash
# 所有实验完成后，生成完整论文图表
python -m rcid.visualization.paper_figures \
    --results_dir outputs/ \
    --output_dir outputs/figures/ \
    --dpi 300
```

---

## 附录 A：完整执行时间线

```
Day 1 (准备):
  □ 环境安装 + 模型下载 (2h)
  □ Pilot validation — Qwen3 (0.5h)
  □ Pilot validation — LLaMA3 (0.5h)
  □ 生成对比对 (1h)
  □ 预计算 teacher saliency (3h)

Day 2 (Toy Data 实验):
  □ 实验 1: StandardKD × 3 tasks × 3 seeds (3h, 4 GPUs)
  □ 实验 2: 4 methods × 3 tasks × 3 seeds (5h, 4 GPUs)
  □ 分析结果，确认 RCID CC > StandardKD CC

Day 3-4 (大规模蒸馏):
  □ 实验 3 Round 1: seed=42, 8 methods (8h, 4 GPUs)
  □ 实验 3 Round 2: seed=123, 8 methods (8h, 4 GPUs)
  □ 实验 3 Round 3: seed=456, 8 methods (8h, 4 GPUs)

Day 5 (Benchmark + 分析):
  □ 所有 checkpoint 的 Benchmark 评估 (8h, 4 GPUs)
  □ λ_RCID 搜索 (如未完成) (5h)
  □ SaGD 消融实验 (3h)

Day 6 (补充实验):
  □ 实验 4: OOD 鲁棒性 (2h)
  □ 实验 5a: 信息纯度 (2h)
  □ 实验 5b: 大规模模型因果分析 (2h)
  □ 实验 5c: LLaMA 3 跨架构 (4h)

Day 7 (图表 + 整理):
  □ 生成所有论文图表
  □ 汇总结果表格
  □ 检查异常值，必要时重跑
```

---

## 附录 B：关键实现约束清单

以下是代码中已实现的关键约束，实验执行时无需额外处理，但需了解：

| 约束 | 实现位置 | 说明 |
|------|---------|------|
| 全序列 KL | `StandardKDLoss` | 所有方法强制使用，不可 answer_only |
| Teacher 始终 eval + no_grad | `ScalableDistillationTrainer.__init__` | teacher.eval() |
| Teacher 痕迹 detached | `_compute_rcid_loss` | t_clean/t_corrupt detached |
| W 矩阵为 buffer | `RCIDLoss.__init__` | register_buffer 冻结 |
| Student 保留梯度 | `extract_residuals_with_grad` | 不 detach |
| 学生显著性 create_graph=True | `scalable_trainer.py:443` | 二阶梯度 |
| 权重 detached | `scalable_trainer.py:462` | weights.detach() |
| JSD 上界 clamp | `saliency.py:213` | clamp(0, log2) |
| Padding 不泄漏概率 | `saliency.py:174` | masked_fill(-inf) |
| Hook try/finally 清理 | 所有 hook 代码 | 异常安全 |
| FP32 上转避免溢出 | `baselines.py:30` | .float() |
| 数值稳定 eps | 全局 | norm.clamp(min=1e-8) |

---

## 附录 C：SaGD 训练流程详解

```
每个 training step (when use_sagd and step % sagd_every == 0):

  1. 主数据流:
     - Teacher forward (no_grad): t_logits = teacher(ids)
     - Student forward (with grad): s_logits = student(ids)

  2. 显著性计算:
     - t_sal = cache[indices]                    # 从预计算缓存查找 (B, L)
     - s_sal = saliency_computer.compute(        # 学生在线计算 (B, L)
         student, ids, mask, labels_mask,
         create_graph=True)                      # 保留计算图用于 L_sal 反传

  3. JSD 重加权:
     - t_dist = to_distribution(t_sal, ...)      # softmax 归一化到 prompt 位置
     - s_dist = to_distribution(s_sal.detach())  # detach: JSD 不需要梯度
     - jsd = divergence(t_dist, s_dist)          # (B,) Jensen-Shannon 散度
     - weights = softmax(jsd / tau_w) * B        # (B,) 均值=1 的样本权重

  4. 重加权 KL 损失:
     - shifted_mask = shift(attention_mask)      # 对齐 logit[j] → token[j+1]
     - per_sample_kl = KL(t_logits, s_logits, shifted_mask)  # (B,)
     - kl_loss = mean(weights.detach() * per_sample_kl)

  5. 显著性对齐损失:
     - sal_loss = mean(1 - cos_sim(t_sal, s_sal))  # L_sal: 余弦距离

  6. 总损失:
     - total_loss = kl_loss + lambda_sal * sal_loss
     - total_loss.backward()  # 梯度通过 s_sal → student params (二阶)
```

非 SaGD 步（step % sagd_every != 0）：

```
  total_loss = StandardKDLoss(t_logits, s_logits, mask=attention_mask)
  total_loss.backward()  # 标准一阶梯度
```

---

## 附录 D：RCID 损失公式

$$\mathcal{L} = \underbrace{\mathcal{L}_{\text{KL}}^{\text{seq}}}_{\text{全序列 KL}} + \lambda \cdot \underbrace{\frac{1}{|\mathcal{C}|} \sum_{(l,t) \in \mathcal{C}} \left\| \frac{W^* d_{\hat{l},t}^{S}}{\|W^* d_{\hat{l},t}^{S}\|} - \frac{d_{l,t}^{T}}{\|d_{l,t}^{T}\|} \right\|^2}_{\text{RCID 正则}}$$

其中：
- $d = h_{\text{clean}} - h_{\text{corrupt}}$：对比差值
- $W^*$：冻结 Procrustes 矩阵 $(d_T, d_S)$
- $\mathcal{C}$：因果检查点集合 $\{(l, t)\}$
- $\hat{l} = \text{layer\_mapping}[l]$：学生对应层

代码实现（`rcid_loss.py`）：
```python
d_S = s_clean[s_layer][:, pos, :] - s_corrupt[s_layer][:, pos, :]  # (B, d_S)
aligned = d_S @ W.t()                                               # (B, d_T)
aligned_n = F.normalize(aligned, dim=-1, eps=1e-8)
d_T_n = F.normalize(d_T, dim=-1, eps=1e-8)
loss = (aligned_n - d_T_n).pow(2).sum(dim=-1).mean()               # scalar
```
