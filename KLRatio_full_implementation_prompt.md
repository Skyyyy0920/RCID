# KL-Ratio Adaptive Distillation: 完整实现 Prompt

## 项目概述

这是一个 LLM 知识蒸馏论文项目。我们提出了 **KL-Ratio Adaptive Distillation**：用 FKL 和 RKL 的比值作为自适应混合信号，零额外开销，超越 AKL (Wu et al., COLING 2025)。

现在需要：
1. **清理代码**：删除所有与 RCID、PADD 相关的旧代码，保留基础设施
2. **整洁实现**：重新实现 KL-Ratio method + AKL baseline
3. **完整实验**：跑论文所需的全部实验（main table + ablation + analysis）

## 模型配置

- Teacher: Qwen/Qwen3-8B (36 layers, d_model=4096)
- Student: Qwen/Qwen3-0.6B (28 layers, d_model=1024)
- 数据: tatsu-lab/alpaca 52K
- 评估: lm-eval-harness (GSM8K, MMLU, ARC-Challenge, HellaSwag, WinoGrande, TruthfulQA)
- 硬件: 4× A100 80GB

## 已有结果（后续实验必须与此可比）

训练超参数（所有实验必须保持一致）：
- epochs: 3, batch_size: 8, gradient_accumulation_steps: 4 (effective batch 32)
- lr: 2e-5, warmup_ratio: 0.03, max_seq_len: 512
- temperature: 2.0, fp16: true, seed: 42

部分已有结果：
```
Method                  | GSM8K  | MMLU   | ARC-C  
forward_kl              | 36.09  |   ?    |   ?    
reverse_kl              | 34.80  |   ?    |   ?    
jeffreys (0.5 FKL+RKL)  | 37.68  |   ?    |   ?    
AKL (Wu et al.)         | 33.74  |  0.00  |  0.00  ← 崩了
KLR-token (ours)        | 33.97  | 46.25  | 41.04  
KLR-batch-ema (ours)    | 37.00  | 46.10  | 41.55  ← 最好
```

---

## 第一步：代码清理

### 需要删除的文件

```bash
# RCID 相关（因果可解释性方法，已放弃）
rm -f src/rcid/circuit/patching.py
rm -f src/rcid/circuit/checkpoint_selection.py
rm -f src/rcid/alignment/cka.py
rm -f src/rcid/alignment/procrustes.py
rm -f src/rcid/distillation/rcid_loss.py
rm -f src/rcid/data/generated_contrastive.py
rm -f src/rcid/eval/causal_consistency.py
rm -f src/rcid/eval/information_purity.py
rm -f src/rcid/eval/ood_robustness.py
rm -f src/rcid/eval/task_accuracy.py
rm -f scripts/generate_contrastive_pairs.py
rm -f scripts/sweep_rcid_parallel.sh
rm -f scripts/run_exp1.py scripts/run_exp2.py scripts/run_exp3.py scripts/run_exp4.py scripts/run_exp5.py
rm -f scripts/run_full_pipeline.sh

# PADD 相关（Phase 1 方法，已被 KL-Ratio 替代）
rm -f src/rcid/distillation/padd_loss.py
rm -f src/rcid/distillation/padd_analysis.py
rm -f scripts/run_padd_distill.py
rm -f scripts/run_padd_phase1.sh

# 旧实验输出（可选，看磁盘空间）
# rm -rf outputs/phase1/
```

### 需要保留的文件

```
src/rcid/
├── distillation/
│   ├── scalable_trainer.py    # 主训练循环（需要清理 RCID/PADD 路由，只保留新方法）
│   ├── baselines.py           # StandardKDLoss, ReverseKDLoss 等（保留）
│   └── adaptive_kl_losses.py  # AKLLoss + KLRatioLoss（已有，保留）
├── models/                    # 模型加载（保留，不动）
├── data/
│   └── instruction_dataset.py # Alpaca 数据加载（保留）
└── eval/                      # 只保留 benchmark 评估相关

scripts/
├── run_large_scale_distill.py # 蒸馏主脚本（保留）
├── eval_benchmarks.py         # lm-eval 评估（保留）
└── run_phase2_experiments.py  # Phase 2 脚本（已有，保留）

configs/                       # 保留
CLAUDE.md                      # 保留
```

### 清理 scalable_trainer.py

在 `scalable_trainer.py` 中：
1. 删除所有 `standard_kd_rcid` 方法的代码路径（搜索 `rcid`）
2. 删除所有 `standard_kd_padd` 方法的代码路径（搜索 `padd`）
3. 保留 `standard_kd`（forward KL baseline）
4. 保留 `standard_kd_akl` 和 `standard_kd_klr`（新方法）
5. 删除 RCID loss 的 import 和初始化
6. 删除 PADD loss 的 import 和初始化
7. 删除对比对（contrastive pair）相关的数据加载逻辑
8. 删除 RCID 相关的 logging

清理后，trainer 只支持三个 method：
- `standard_kd`: 标准 forward KL
- `standard_kd_akl`: AKL baseline
- `standard_kd_klr`: KL-Ratio（我们的方法）

另外需要在 `standard_kd` 路径中加一个 `reverse_kl` 选项（通过 config 控制），或者新增一个 `reverse_kl` method，使得实验中也能跑 reverse KL baseline。

如果原有代码中 reverse KL 是通过 baselines.py 的 ReverseKDLoss 实现的，确保它仍然可用。

---

## 第二步：验证 adaptive_kl_losses.py

检查已有的 `src/rcid/distillation/adaptive_kl_losses.py`，确认：

1. **AKLLoss** 实现正确：
   - teacher logits 除以 temperature 后 softmax
   - 按累积概率 ≥ μ 划分 head/tail
   - 计算 head/tail 的 L1 gap
   - α = g_head / (g_head + g_tail + ε)
   - loss = α * FKL + (1-α) * RKL
   - α detached

2. **KLRatioLoss** 实现正确：
   - Token-level: α_t = FKL_t / (FKL_t + RKL_t + ε) per position
   - Batch-level + EMA: α_ema = β * α_ema_prev + (1-β) * α_inst
   - α_ema 用 register_buffer 存储
   - 初始化 α_ema = 0.5

3. 两者的接口一致：`forward(teacher_logits, student_logits, mask) → (loss, stats_dict)`

如果有 bug 导致 AKL 结果异常（MMLU=0），需要修复。可能的问题：
- AKL 的排序操作导致梯度断裂
- α 没有 detach 导致梯度爆炸
- head/tail mask 的 cumsum 实现有误
- temperature 应用错误

**重要**：先单独跑 AKL 的单元测试确认实现无误。如果确认实现正确而 AKL 结果就是差，那这本身就是论文的一个发现。

---

## 第三步：实验设计

### 实验 1: Main Comparison Table（论文 Table 1）

**目的**：完整对比所有方法在 6 个 benchmark 上的表现。

```python
experiments = {
    # Baselines
    "forward_kl":    {"method": "standard_kd"},
    "reverse_kl":    {"method": "standard_kd", "reverse_kl": True},  
    # 或者 {"method": "reverse_kl"} 取决于 trainer 支持
    "jeffreys":      {"method": "standard_kd_klr", 
                      "klr_granularity": "fixed", "klr_fixed_alpha": 0.5},
    # 如果没有 fixed 模式，可以直接实现一个 JeffreysLoss 
    # 或者用 KLRatioLoss 但初始化 alpha=0.5, beta=1.0（永不更新）
    
    # Literature baseline
    "akl":           {"method": "standard_kd_akl", "akl_mu": 0.5},
    
    # Our methods
    "klr_token":     {"method": "standard_kd_klr", "klr_granularity": "token"},
    "klr_batch_ema": {"method": "standard_kd_klr", "klr_granularity": "batch", "klr_beta": 0.99},
}

benchmarks = ["gsm8k", "mmlu", "arc_challenge", "hellaswag", "winogrande", "truthfulqa"]
```

**注意**：forward_kl, reverse_kl, jeffreys 之前只有 GSM8K 分数，需要补跑 MMLU, ARC-C, HellaSwag, WinoGrande, TruthfulQA。如果 Phase 1 的 checkpoint 还在（outputs/phase1/），可以直接用 eval_benchmarks.py 评估，不用重新训练。

**4 GPU 并行方案**：
```
Round 1:  GPU0: forward_kl    GPU1: reverse_kl    GPU2: jeffreys    GPU3: akl
Round 2:  GPU0: klr_token     GPU1: klr_batch_ema
```

如果部分实验已有 checkpoint，只需补评估。

### 实验 2: Ablation — EMA β 的影响（论文 Table 2）

**目的**：验证 EMA 平滑的重要性及 β 的鲁棒性。

```python
beta_experiments = {
    "klr_no_ema":      {"klr_granularity": "batch", "klr_beta": 0.0},   # 无 EMA，等同于 batch-level instant
    "klr_beta_0.9":    {"klr_granularity": "batch", "klr_beta": 0.9},
    "klr_beta_0.95":   {"klr_granularity": "batch", "klr_beta": 0.95},
    "klr_beta_0.99":   {"klr_granularity": "batch", "klr_beta": 0.99},  # 默认
    "klr_beta_0.999":  {"klr_granularity": "batch", "klr_beta": 0.999},
}
```

评估只需 GSM8K（足以看趋势），如果结果有趣再补跑其他 benchmark。

**4 GPU 并行**：5 个实验 → Round 1 跑 4 个，Round 2 跑 1 个。

### 实验 3: Ablation — Token-level vs Batch-level（论文 Table 3）

已有数据（klr_token vs klr_batch_ema），不需要额外跑。直接用实验 1 的数据呈现。

### 实验 4: Analysis — α 轨迹可视化（论文 Figure 1）

**目的**：展示 α_ema 在训练过程中的变化轨迹，验证 "自动 curriculum" 性质。

**实现**：在 KLRatioLoss 中已经有 stats dict 输出 alpha_ema。需要在训练循环中保存每个 step 的：
- alpha_ema
- fkl_mean
- rkl_mean
- training_loss

保存到 JSON 文件，然后用 matplotlib 画图。

```python
# 画图脚本：scripts/plot_alpha_trajectory.py
# 输入：outputs/phase2/klr_batch_ema/training_stats.json
# 输出：
#   Figure 1a: α_ema vs training step（展示 α 随训练的变化）
#   Figure 1b: FKL_mean 和 RKL_mean vs training step（展示两个 KL 的变化）
#   Figure 1c: α_ema 在不同 β 下的对比
```

### 实验 5: Analysis — Training Loss Curve 对比（论文 Figure 2）

**目的**：对比不同方法的训练 loss 收敛速度。

所有方法的 training loss 应该在训练过程中被记录。画一张图对比：
- forward_kl
- jeffreys
- AKL（如果能跑通）
- KLR-batch-ema

---

## 第四步：完整实验脚本

### 新建 `scripts/run_all_experiments.sh`

```bash
#!/usr/bin/env bash
# 论文完整实验：4-GPU 并行
set -euo pipefail
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

OUTPUT_DIR="${OUTPUT_DIR:-outputs/paper}"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
LOG_DIR="${OUTPUT_DIR}/logs"
mkdir -p "$LOG_DIR"

run_exp() {
    local exp_name="$1"
    local gpu_id="$2"
    local log_file="${LOG_DIR}/${exp_name}.log"
    echo "[GPU ${gpu_id}] Starting: ${exp_name}"
    CUDA_VISIBLE_DEVICES="$gpu_id" python "${SCRIPT_DIR}/run_paper_experiments.py" \
        --experiment "$exp_name" \
        --device "cuda:0" \
        --output_dir "$OUTPUT_DIR" \
        > "$log_file" 2>&1
    local status=$?
    if [ $status -eq 0 ]; then
        echo "[GPU ${gpu_id}] ✓ ${exp_name}"
    else
        echo "[GPU ${gpu_id}] ✗ ${exp_name} (see ${log_file})"
    fi
    return $status
}

# ============================================================
# Part 1: Main Table (Exp 1)
# ============================================================
echo "===== Part 1: Main Comparison ====="

# Round 1: 4 parallel
run_exp forward_kl   0 &
run_exp reverse_kl   1 &
run_exp jeffreys     2 &
run_exp akl          3 &
wait
echo "Round 1 done"

# Round 2: 2 parallel
run_exp klr_token     0 &
run_exp klr_batch_ema 1 &
wait
echo "Round 2 done"

# ============================================================
# Part 2: Beta Ablation (Exp 2)
# ============================================================
echo "===== Part 2: Beta Ablation ====="

run_exp klr_no_ema     0 &
run_exp klr_beta_0.9   1 &
run_exp klr_beta_0.95  2 &
run_exp klr_beta_0.999 3 &
wait
echo "Beta ablation done"

# ============================================================
# Part 3: Generate plots
# ============================================================
echo "===== Part 3: Plots ====="
python "${SCRIPT_DIR}/plot_paper_figures.py" --output_dir "$OUTPUT_DIR"

echo ""
echo "===== ALL DONE ====="
```

### 新建 `scripts/run_paper_experiments.py`

主实验脚本，功能：

```python
"""
论文实验统一入口。
用法：
    python scripts/run_paper_experiments.py --experiment forward_kl --device cuda:0 --output_dir outputs/paper
    python scripts/run_paper_experiments.py --experiment klr_batch_ema --device cuda:0 --output_dir outputs/paper
"""

import argparse

# 实验配置表
EXPERIMENT_CONFIGS = {
    # --- Main Table ---
    "forward_kl": {
        "method": "standard_kd",
        "description": "Standard forward KL baseline",
    },
    "reverse_kl": {
        "method": "reverse_kl",  # 或通过 config flag 控制
        "description": "Reverse KL baseline",
    },
    "jeffreys": {
        "method": "standard_kd_klr",
        "klr_granularity": "batch",
        "klr_beta": 1.0,        # beta=1.0 → alpha 永远是初始值 0.5 → Jeffreys
        "description": "Jeffreys divergence (fixed α=0.5)",
    },
    "akl": {
        "method": "standard_kd_akl",
        "akl_mu": 0.5,
        "description": "AKL (Wu et al., COLING 2025)",
    },
    "klr_token": {
        "method": "standard_kd_klr",
        "klr_granularity": "token",
        "description": "KL-Ratio token-level (ours)",
    },
    "klr_batch_ema": {
        "method": "standard_kd_klr",
        "klr_granularity": "batch",
        "klr_beta": 0.99,
        "description": "KL-Ratio batch + EMA (ours)",
    },
    
    # --- Beta Ablation ---
    "klr_no_ema": {
        "method": "standard_kd_klr",
        "klr_granularity": "batch",
        "klr_beta": 0.0,
        "description": "KL-Ratio batch, no EMA (β=0)",
    },
    "klr_beta_0.9": {
        "method": "standard_kd_klr",
        "klr_granularity": "batch",
        "klr_beta": 0.9,
        "description": "KL-Ratio batch, β=0.9",
    },
    "klr_beta_0.95": {
        "method": "standard_kd_klr",
        "klr_granularity": "batch",
        "klr_beta": 0.95,
        "description": "KL-Ratio batch, β=0.95",
    },
    "klr_beta_0.999": {
        "method": "standard_kd_klr",
        "klr_granularity": "batch",
        "klr_beta": 0.999,
        "description": "KL-Ratio batch, β=0.999",
    },
}

# 共用训练超参数（所有实验必须一致）
TRAINING_CONFIG = {
    "teacher_model": "Qwen/Qwen3-8B",
    "student_model": "Qwen/Qwen3-0.6B",
    "dataset": "tatsu-lab/alpaca",
    "epochs": 3,
    "batch_size": 8,
    "gradient_accumulation_steps": 4,
    "lr": 2e-5,
    "warmup_ratio": 0.03,
    "max_seq_len": 512,
    "temperature": 2.0,
    "fp16": True,
    "seed": 42,
}

# 评估 benchmark 列表
EVAL_BENCHMARKS = ["gsm8k", "mmlu", "arc_challenge", "hellaswag", "winogrande", "truthfulqa_mc2"]
EVAL_BATCH_SIZE = 4  # 降低避免 OOM
```

脚本的 main 逻辑：
1. 解析命令行：`--experiment`, `--device`, `--output_dir`, `--eval_only`(可选，只跑评估)
2. 合并 EXPERIMENT_CONFIGS[experiment] 和 TRAINING_CONFIG
3. 设置输出路径：`{output_dir}/{experiment}/`
4. 如果不是 eval_only：
   a. 初始化 trainer（参考 run_large_scale_distill.py 的方式）
   b. 训练，每 100 步记录 stats 到 `training_stats.jsonl`
   c. 保存最终 checkpoint
5. 运行 lm-eval 评估所有 benchmark
6. 保存 eval_results.json

**关键**：训练过程中的 stats 记录格式：
```jsonl
{"step": 100, "loss": 2.34, "alpha_mean": 0.48, "fkl_mean": 1.12, "rkl_mean": 1.22, "lr": 1.5e-5}
{"step": 200, "loss": 2.01, "alpha_mean": 0.45, "fkl_mean": 0.98, "rkl_mean": 1.15, "lr": 1.8e-5}
...
```

### 新建 `scripts/plot_paper_figures.py`

画论文图表：

```python
"""
生成论文所需的所有图表。
用法：python scripts/plot_paper_figures.py --output_dir outputs/paper
"""

# Figure 1: α_ema trajectory during training
# - X 轴：training step
# - Y 轴：α_ema
# - 一条线：klr_batch_ema 的 α 轨迹
# - 标注：training 初期 α>0.5（FKL 主导），后期 α<0.5（RKL 主导）
# - 如果有多个 β 的实验，画多条线对比

# Figure 2: FKL and RKL during training
# - X 轴：training step
# - 双 Y 轴或同一 Y 轴：FKL_mean 和 RKL_mean
# - 展示：FKL 下降更快（先拟合 head），RKL 下降更慢（tail 更难）

# Figure 3: Training loss curves comparison
# - 对比 forward_kl, jeffreys, klr_batch_ema 的训练 loss

# Figure 4: Beta ablation bar chart
# - X 轴：β values (0, 0.9, 0.95, 0.99, 0.999)
# - Y 轴：GSM8K accuracy
# - 展示 β=0.99 附近最好

# 保存到 {output_dir}/figures/
```

### 新建或修改 `scripts/eval_only.sh`

用于对已有 checkpoint 补跑评估（不重新训练）：

```bash
#!/usr/bin/env bash
# 对已有 checkpoint 补跑所有 benchmark
# 用法: bash scripts/eval_only.sh outputs/phase1/standard_kd cuda:0

CHECKPOINT_DIR="$1"
DEVICE="${2:-cuda:0}"

python scripts/run_paper_experiments.py \
    --experiment forward_kl \
    --device "$DEVICE" \
    --output_dir "$CHECKPOINT_DIR" \
    --eval_only \
    --checkpoint_path "${CHECKPOINT_DIR}/student_final"
```

---

## 第五步：Jeffreys 的实现方式

有两种方式实现 Jeffreys baseline，选一种：

**方式 A**（推荐）：在 KLRatioLoss 中加一个 `fixed_alpha` 参数
```python
class KLRatioLoss(nn.Module):
    def __init__(self, temperature=2.0, granularity='batch', beta=0.99, fixed_alpha=None):
        # 如果 fixed_alpha 不为 None，则始终使用该值，忽略 ratio 计算
        self.fixed_alpha = fixed_alpha
    
    def forward(self, ...):
        if self.fixed_alpha is not None:
            alpha = self.fixed_alpha
        else:
            # 正常的 ratio 计算
            ...
```

**方式 B**：beta=1.0 使得 EMA 永不更新，始终是初始值 0.5。

方式 A 更清晰。Jeffreys 实验配置为 `{"klr_granularity": "batch", "fixed_alpha": 0.5}`。

同理，reverse KL 可以实现为 `fixed_alpha=0.0`。这样所有方法统一在 KLRatioLoss 框架下：

| 方法 | 实现 |
|------|------|
| Forward KL | StandardKDLoss (baselines.py) |
| Reverse KL | KLRatioLoss(fixed_alpha=0.0) 或 ReverseKDLoss |
| Jeffreys | KLRatioLoss(fixed_alpha=0.5) |
| KLR-token | KLRatioLoss(granularity='token') |
| KLR-batch-ema | KLRatioLoss(granularity='batch', beta=0.99) |

---

## 第六步：修复 AKL 的潜在问题

AKL 出现 MMLU=0, ARC-C=0，可能原因：

1. **checkpoint 保存/加载问题**：评估时加载的不是正确的模型
   - 检查：`outputs/phase2/akl_mu0.5/` 下是否有有效的 checkpoint
   - 检查：checkpoint 大小是否合理（~1.2GB for 0.6B model）

2. **训练 loss 爆炸**：α 的 grad 没 detach，导致梯度不稳定
   - 检查：training_stats 中 loss 是否有 NaN 或突然飙升
   - 修复：确认 α 计算中所有中间量都 detach

3. **排序导致的数值问题**：在 fp16 下对 150K 词表排序可能有问题
   - 修复：排序前转 float32

4. **评估脚本路径错误**：可能评估了未训练的原始模型

先检查 log 确定原因，再决定是否需要修复。如果 AKL 实现确实正确但结果就是差，那就如实报告——这说明 AKL 在更大规模模型上不鲁棒。

---

## 第七步：输出文件结构

最终目录结构应该是：

```
outputs/paper/
├── logs/                           # 每个实验的完整 log
│   ├── forward_kl.log
│   ├── reverse_kl.log
│   └── ...
├── forward_kl/
│   ├── student_final/              # HF 格式 checkpoint
│   ├── training_stats.jsonl        # 训练过程统计
│   └── eval_results.json           # 评估结果
├── reverse_kl/
│   └── ...
├── jeffreys/
│   └── ...
├── akl/
│   └── ...
├── klr_token/
│   └── ...
├── klr_batch_ema/
│   └── ...
├── klr_no_ema/                     # beta ablation
│   └── ...
├── klr_beta_0.9/
│   └── ...
├── klr_beta_0.95/
│   └── ...
├── klr_beta_0.999/
│   └── ...
├── figures/
│   ├── alpha_trajectory.pdf
│   ├── fkl_rkl_curves.pdf
│   ├── loss_comparison.pdf
│   └── beta_ablation.pdf
└── summary_table.txt               # 汇总表格
```

---

## 代码规范

1. **类型标注**：所有函数签名完整标注
2. **Tensor shape 注释**：`# (batch, seq, vocab)` 格式
3. **数值稳定性**：F.log_softmax，eps=1e-8
4. **梯度管理**：teacher detach，alpha detach
5. **文件长度**：每个 .py ≤ 300 行
6. **日志格式**：JSONL for stats, JSON for final results

## 验证清单

实现完成后，逐项检查：

- [ ] `python -c "from rcid.distillation.adaptive_kl_losses import AKLLoss, KLRatioLoss; print('OK')"` 
- [ ] `python -c "from rcid.distillation.baselines import StandardKDLoss; print('OK')"`
- [ ] scalable_trainer.py 中无 `rcid_loss`、`padd_loss` 的 import
- [ ] scalable_trainer.py 支持 `standard_kd`、`standard_kd_akl`、`standard_kd_klr` 三个 method
- [ ] KLRatioLoss 的 fixed_alpha=0.5 模式输出的 loss ≈ 0.5*FKL + 0.5*RKL
- [ ] KLRatioLoss 的 batch EMA 模式，alpha_ema 随 forward 调用更新
- [ ] training_stats.jsonl 每 100 步记录一行
- [ ] eval_results.json 包含所有 6 个 benchmark 的分数
- [ ] 所有实验的训练超参数完全一致

## 开始步骤

1. **先做代码清理**（删文件 + 清理 trainer）
2. **检查 adaptive_kl_losses.py**（确认实现正确）
3. **加 fixed_alpha 支持到 KLRatioLoss**
4. **确保 training_stats.jsonl 输出**（每 100 步记录 alpha, fkl, rkl, loss）
5. **实现 run_paper_experiments.py**（统一实验入口）
6. **实现 run_all_experiments.sh**（4-GPU 并行）
7. **实现 plot_paper_figures.py**（画图）
8. **跑单元测试 + sanity check**
9. **开始正式实验**
