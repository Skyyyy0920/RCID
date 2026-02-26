# PADD Implementation Prompt for Claude Code

## 项目背景

这是一个 LLM 知识蒸馏研究项目，原本研究 RCID（基于因果可解释性的蒸馏），现在转向一个新方法 PADD（Position-Adaptive Divergence Distillation）。项目已有完整的模型加载、数据处理、训练循环和评估基础设施。

**模型配置**：
- Teacher: Qwen/Qwen3-8B (36 layers, d_model=4096)
- Student: Qwen/Qwen3-0.6B (28 layers, d_model=1024)
- 数据: Alpaca-52K 指令数据集
- 评估: lm-eval (MMLU, GSM8K, ARC-C, HellaSwag, WinoGrande, TruthfulQA)

## 新方法：PADD（Position-Adaptive Divergence Distillation）

### 核心 Idea

标准 KD 对所有 token 位置用同一个 forward KL 散度。但不同位置的 teacher 分布特征不同，最优的知识传递方式也不同：

- **Teacher 确信的位置**（entropy 低）：teacher 有一个明确答案。Reverse KL 更有效——它惩罚 student 在 teacher 认为不可能的地方放概率（mode-seeking），帮助 student 精确学会确定性知识。
- **Teacher 不确定的位置**（entropy 高）：dark knowledge 蕴含在多个选项的相对概率中。Forward KL 更有效——它迫使 student 覆盖所有 teacher 认为可能的选项（mode-covering），保留 dark knowledge。

PADD 根据每个位置的 teacher entropy 自适应混合 forward KL 和 reverse KL。

### 数学公式

对每个 token 位置 t，计算自适应权重：

```
alpha(t) = sigmoid((H(p_T(t)) - mu_H) / tau)
```

其中：
- H(p_T(t)) 是 teacher 在位置 t 的 entropy
- mu_H 是当前 batch 中所有有效位置的平均 teacher entropy
- tau 是温度超参数，控制 forward/reverse KL 的混合锐度

位置 t 的 loss：

```
L(t) = alpha(t) * KL(p_T || p_S) + (1 - alpha(t)) * KL(p_S || p_T)
```

总 loss（带 attention mask）：

```
L_PADD = sum(L(t) * mask(t)) / sum(mask(t))
```

当 tau → ∞ 时，alpha → 0.5 对所有位置，退化为 JSD（安全基线）。

### KL 计算注意事项

Forward KL: KL(p_T || p_S) = sum(p_T * log(p_T / p_S))
Reverse KL: KL(p_S || p_T) = sum(p_S * log(p_S / p_T))

两个 KL 都需要对 temperature-scaled logits 计算。设蒸馏温度为 T（默认2.0）：
```
p_T = softmax(teacher_logits / T)
p_S = softmax(student_logits / T)
```

Loss 要乘以 T^2（标准 KD 的 convention）。

数值稳定性：用 log_softmax 计算，避免显式计算概率后取 log。

## 需要做的修改

### 1. 新建文件：`src/rcid/distillation/padd_loss.py`

实现 PADDLoss 类：

```python
class PADDLoss(nn.Module):
    """Position-Adaptive Divergence Distillation Loss.
    
    根据 teacher entropy 自适应混合 forward KL 和 reverse KL。
    Teacher 确信的位置偏向 reverse KL（mode-seeking），
    Teacher 不确定的位置偏向 forward KL（mode-covering）。
    
    Args:
        temperature: KD 温度（默认 2.0）
        tau: 自适应混合的温度参数（默认 1.0）
        alpha_min: alpha 的下限 clamp（默认 0.1）
        alpha_max: alpha 的上限 clamp（默认 0.9）
    """
    
    def forward(self, teacher_logits, student_logits, mask=None):
        """
        Args:
            teacher_logits: (batch, seq_len, vocab_size)
            student_logits: (batch, seq_len, vocab_size)
            mask: (batch, seq_len) attention mask, 1 for valid positions
            
        Returns:
            loss: scalar
            stats: dict with alpha_mean, forward_kl_mean, reverse_kl_mean, teacher_entropy_mean
        """
```

关键设计要求：
- 返回 loss 和 stats dict（用于 logging）
- stats 包含：alpha_mean（平均混合系数）、forward_kl_mean、reverse_kl_mean、teacher_entropy_mean
- 数值稳定：用 F.log_softmax 计算，不要显式 softmax 后取 log
- mask 处理：只在有效位置计算 loss 和 entropy 统计
- alpha clamp 到 [alpha_min, alpha_max] 避免极端值
- mu_H 在每个 batch 的有效位置上计算（不是全局的）

### 2. 修改文件：`src/rcid/distillation/scalable_trainer.py`

在 ScalableDistillationTrainer 中添加 PADD 支持。

当前 trainer 通过 config 中的 method 字段路由到不同方法。需要添加一个新 method：`standard_kd_padd`。

修改点：
- `__init__` 中：如果 method 是 `standard_kd_padd`，初始化 PADDLoss
- 训练循环中：替换 kl_loss 的计算
  - 原来：`kl_loss = kd_loss_fn(t_logits, s_logits, mask=attn_mask)` 
  - PADD：`kl_loss, padd_stats = padd_loss_fn(t_logits, s_logits, mask=attn_mask)`
- logging 中：记录 padd_stats 中的 alpha_mean 等指标

**不要改动现有方法的代码路径**。standard_kd、standard_kd_rcid 等保持不变。

查看现有的 scalable_trainer.py 了解训练循环结构、loss 计算方式、logging 方式，然后做最小修改。

### 3. 新建文件：`scripts/run_padd_distill.py`

Phase 1 快速验证脚本。参考现有的 `scripts/run_large_scale_distill.py` 的结构。

实验矩阵：
```python
methods = [
    "standard_kd",           # baseline: 纯 forward KL
    "standard_kd_padd",      # PADD: adaptive forward/reverse KL
]

# PADD 超参搜索
padd_tau_values = [0.5, 1.0, 2.0, 5.0]

# 额外 ablation baselines（作为 PADDLoss 的特殊配置）
# pure_reverse_kl: alpha 固定为 0（全部 reverse KL）
# fixed_jsd: alpha 固定为 0.5（JSD）
```

训练超参数沿用现有配置（参见 CLAUDE.md 10.4 节）：
- epochs: 3
- batch_size: 8, gradient_accumulation_steps: 4
- lr: 2e-5, warmup_ratio: 0.03
- max_seq_len: 512
- temperature: 2.0
- fp16: true

脚本功能：
- 接受命令行参数：--method, --tau, --device, --output_dir, --seed
- 训练完成后自动运行 lm-eval 评估（MMLU, GSM8K, ARC-C 三个核心 benchmark）
- 保存训练 loss 曲线和 PADD stats（alpha 分布随训练的变化）
- 支持多 seed 运行

### 4. 新建文件：`scripts/run_padd_phase1.sh`

一键运行 Phase 1 所有实验的 shell 脚本：

```bash
# Phase 1: 快速验证（预计 2-3 天，单卡 A100）
# 
# 实验：
# 1. standard_kd (baseline)               x1
# 2. PADD tau=0.5                          x1
# 3. PADD tau=1.0                          x1  
# 4. PADD tau=2.0                          x1
# 5. PADD tau=5.0                          x1
# 6. pure reverse KL (ablation)            x1
# 7. fixed JSD (ablation)                  x1
#
# 每个实验约 3-4 小时（Alpaca-52K, 3 epochs, A100）
# 评估约 30 分钟 / 模型
#
# 判断标准：任何 tau 在任何 benchmark 上超过 standard_kd → Phase 2
```

### 5. 新建文件：`src/rcid/distillation/padd_analysis.py`

MI 分析工具（Phase 3 用，但现在就建好接口）：

```python
def analyze_alpha_distribution(trainer_stats: dict) -> dict:
    """分析训练过程中 alpha 的分布变化。"""
    
def compare_logit_lens_trajectory(
    teacher, student_padd, student_baseline, 
    tokenizer, samples, adapter, device
) -> dict:
    """用 logit lens 比较 PADD student 和 baseline student 的预测形成过程。
    
    对每个 sample：
    - 提取 teacher 逐层 logit lens 预测
    - 提取 PADD student 逐层预测
    - 提取 baseline student 逐层预测
    - 计算每层的 teacher-student 预测一致性
    """

def analyze_representation_structure(
    teacher, student_padd, student_baseline,
    samples, adapter, device
) -> dict:
    """比较两个 student 在残差流表示空间上和 teacher 的结构相似度。"""
```

## 不要修改的文件

- `src/rcid/models/` — 模型加载和 adapter 保持不变
- `src/rcid/data/` — 数据加载保持不变
- `src/rcid/circuit/` — 因果分析工具保持不变（MI 分析阶段会用）
- `src/rcid/alignment/` — Procrustes 等保持不变
- `src/rcid/distillation/rcid_loss.py` — 保持不变
- `src/rcid/distillation/baselines.py` — 保持不变（里面的 StandardKDLoss 作为 baseline）
- `src/rcid/eval/` — 评估工具保持不变
- `tests/` — 现有测试保持不变
- `configs/` — 现有配置保持不变

## 代码规范

遵循 CLAUDE.md 第十二节的编码规范：

1. **类型标注**：所有函数签名完整标注
2. **Tensor shape 注释**：每个 tensor 操作标注 shape
3. **断言检查**：关键维度和数值检查
4. **数值稳定性**：eps=1e-8，用 log_softmax 而非 softmax+log
5. **文件长度**：每个 .py 不超过 300 行
6. **梯度流管理**：teacher 始终 eval + no_grad，student 保留梯度

## 验证方法

实现完成后，请做以下检查：

1. `python -c "from rcid.distillation.padd_loss import PADDLoss; print('Import OK')"` — 确认导入正常
2. 写一个简单的单元测试验证 PADDLoss：
   - 随机 teacher/student logits，确认 loss 是有限正数
   - 当 tau 很大时（如 100），alpha 接近 0.5，loss 接近 JSD
   - 当 teacher_logits == student_logits 时，loss 接近 0
   - mask 正确过滤 padding 位置
   - stats dict 包含所有预期的 key
3. 确认现有的 `standard_kd` 方法路径不受影响

## 开始步骤

1. 先 `cat src/rcid/distillation/scalable_trainer.py` 了解现有训练循环结构
2. 先 `cat src/rcid/distillation/baselines.py` 了解现有 loss 的接口
3. 再 `cat scripts/run_large_scale_distill.py` 了解现有实验脚本结构
4. 然后按上述要求实现新文件和修改
