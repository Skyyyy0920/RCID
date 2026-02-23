# RCID Project — CLAUDE.md

> **项目名称**: RCID (Residual Causal Imprint Distillation)
> **核心问题**: 蒸馏后的 student 内部推理机制是否与 teacher 一致？如何让它一致？
> **核心定位**: RCID 是一个**即插即用的机制对齐正则项**，嫁接到标准蒸馏流程上，让学生不仅"表现得像"教师，也"思考得像"教师。
> **模型配对（主实验）**: Qwen3-8B (teacher) → Qwen3-0.6B (student)
> **模型配对（泛化验证）**: LLaMA-3-8B (teacher) → LLaMA-3.2-1B (student)
> **硬件**: 4 × A100 80GB

---

## 一、论文的四条主线

### 主线 A：发现问题

现有蒸馏方法产出的 student 即使任务准确率接近 teacher，
其内部推理机制可能完全不同。我们提出 **mechanistic consistency** 指标来量化这个现象。

### 主线 B：解决问题

我们提出 RCID，一种对比差值引导的蒸馏正则项。通过在因果关键位置匹配 teacher 和 student
的残差流对比差值（而非原始表示），RCID 在因果一致性、信息纯度、OOD 鲁棒性上均优于现有方法。

### 主线 C：实用性验证（大规模蒸馏）

将 RCID 作为正则项嫁接到大规模 KL 蒸馏上，通过自动构造的对比对，
在 MMLU、GSM8K、ARC 等真实 benchmark 上带来可测量的提升。

### 主线 D：跨架构泛化

在 LLaMA 3 上重复核心实验，证明方法不依赖于特定架构。

### 论文核心叙事

> "标准蒸馏让学生**表现得像**教师（输出分布对齐），
> RCID 正则让学生**思考得像**教师（内部推理机制对齐），两者互补。"

1. 我们发现标准蒸馏不保留教师的推理机制（实验 1）
2. 提出 RCID 方法并在受控环境下验证其有效性（实验 2）
3. 展示将 RCID 作为正则项嫁接到大规模蒸馏上可以提升学生在真实 benchmark 上的表现（实验 3，核心实用性实验）
4. 通过 OOD 鲁棒性实验证明机制保留带来更好的泛化（实验 4）
5. 通过因果分析和跨架构验证解释了提升的来源（实验 5）

---

## 二、核心概念

### 2.1 两种残差流操作

**Read（提取对比差值）**：读取残差流值，不改变模型行为。用于检查点搜索和蒸馏训练。
```python
h_clean = hook_read(model, clean_input, layer, pos)
h_corrupt = hook_read(model, corrupt_input, layer, pos)
d = h_clean - h_corrupt  # 对比差值 (contrastive difference)
```

**Write（因果干预 / activation patching）**：替换残差流值并继续前向传播。用于因果一致性评估。
```python
logits_patched = patch_and_run(model, clean_input, corrupt_value, layer, pos)
delta = logit_diff(original) - logit_diff(patched)  # 因果效应
```

### 2.2 因果一致性指标

对 teacher 和 student 施加相同的因果干预（Write），比较行为变化：

$$\text{CausalConsistency} = \text{Pearson}(\Delta_T, \Delta_S)$$

- $\Delta_T$：对 teacher 在检查点做 patching 后的输出变化
- $\Delta_S$：对 student 在对应位置做同样 patching 后的输出变化
- student 的 patching 值来自 student 自身的 corrupt 前向传播（不是 teacher 的）

### 2.3 RCID 损失（正则项形式）

**完整损失函数：**

$$\mathcal{L} = \underbrace{\mathcal{L}_{\text{KL}}^{\text{seq-level}}}_{\text{全序列 KL 蒸馏}} + \lambda \cdot \underbrace{\frac{1}{|\mathcal{C}|} \sum_{(l,t) \in \mathcal{C}} \left\| \frac{W^* d_{\hat{l},t}^{S}}{\|W^* d_{\hat{l},t}^{S}\|} - \frac{d_{l,t}^{T}}{\|d_{l,t}^{T}\|} \right\|^2}_{\text{RCID 因果差值匹配正则项}}$$

其中：
- $\mathcal{L}_{\text{KL}}^{\text{seq-level}}$ 是在**所有 token 位置**上的 per-token KL 散度（主损失，保证通用能力）
- $d = h_{\text{clean}} - h_{\text{corrupt}}$ 是对比差值
- $W^*$ 是冻结的 Procrustes 对齐矩阵
- $\mathcal{C}$ 是因果检查点集合

**两个数据流：**
- **主数据流**：大规模指令数据（Alpaca-52K 等），计算全序列 KL
- **RCID 数据流**：少量自动构造的对比对（~5K），计算因果差值匹配

### 2.4 因果检查点选择

搜索 teacher 内部所有 (层, token位置) 组合，按残差流差值范数排序选 top-k。

**多样性约束**：
- **被修改位置**：clean 和 corrupt 在该位置放了不同 token。底层差异是 trivial 的（embedding 不同），只有高层差异反映模型加工。
- **未被修改位置**：clean 和 corrupt 在该位置 token 相同。任何差异都来自模型内部信息传播，是干净的信号。
- 检查点选择确保两类位置都有代表。

---

## 三、KL Loss 规范（关键设计决策）

### 3.1 全序列 KL（强制）

**所有蒸馏方法必须使用全序列 per-token KL，不再仅在 answer_pos 计算。**

理由：
1. 与主流蒸馏方法（DistilBERT、MiniLM、TinyBERT、SDFT/SDPO）一致
2. 全序列 KL 保护学生的语言建模能力，避免只学到单个位置的输出
3. RCID 的价值通过附加正则项体现，而非通过削弱基线来人为放大
4. 大规模蒸馏场景下真实指令数据没有单一 answer position

### 3.2 StandardKDLoss 接口

```python
class StandardKDLoss(nn.Module):
    def forward(
        self,
        teacher_logits: torch.Tensor,  # (batch, seq_len, vocab) 或 (batch, vocab)
        student_logits: torch.Tensor,  # 同 teacher
        mask: torch.Tensor | None = None,  # (batch, seq_len)，1=有效，0=padding
    ) -> torch.Tensor:
        # 支持两种输入形状：
        # - 3D (batch, seq, vocab)：全序列 KL，用 mask 过滤 padding
        # - 2D (batch, vocab)：单位置 KL（向后兼容 toy data）
```

### 3.3 Trainer KL 模式

```yaml
kl_mode: "sequence"   # 全序列 KL（默认，用于所有正式实验）
# kl_mode: "answer_only"  # 仅 answer position（仅用于调试/向后兼容）
```

UnifiedTrainer（toy data）和 ScalableDistillationTrainer（大规模数据）均已实现此逻辑。

---

## 四、模型架构

### 4.1 架构抽象层

所有代码通过 ModelAdapter 抽象层与模型交互，不直接硬编码架构细节。
支持 Qwen3 和 LLaMA 3 两种架构。

```python
class ModelAdapter(ABC):
    @abstractmethod
    def get_layers(self, model) -> nn.ModuleList: ...
    @abstractmethod
    def get_embed_tokens(self, model) -> nn.Embedding: ...
    @abstractmethod
    def get_lm_head(self, model) -> nn.Linear: ...
    @abstractmethod
    def get_residual_hook_point(self, model, layer_idx) -> nn.Module: ...
    @abstractmethod
    def parse_layer_output(self, output) -> torch.Tensor: ...
    @abstractmethod
    def get_num_layers(self, model) -> int: ...
    @abstractmethod
    def get_hidden_size(self, model) -> int: ...
```

### 4.2 Qwen3 架构细节

```yaml
Qwen3-8B (Teacher):
  hf_name: Qwen/Qwen3-8B
  layers: model.model.layers[0..35]    # 36 层
  d_model: 4096
  n_heads: 32 (GQA: 32 query heads, 4 KV heads)
  vocab: 151936
  embed: model.model.embed_tokens
  lm_head: model.lm_head
  norm: RMSNorm (Pre-Norm) + QK-Norm
  position: RoPE
  activation: SwiGLU
  layer_output: output[0]  # (batch, seq, 4096)

Qwen3-0.6B (Student):
  hf_name: Qwen/Qwen3-0.6B
  layers: model.model.layers[0..27]    # 28 层
  d_model: 1024
  n_heads: 16 (GQA: 16 query heads, 8 KV heads)
  vocab: 151936 (共享词表)
  其余同上
```

### 4.3 LLaMA 3 架构细节

```yaml
LLaMA-3-8B (Teacher):
  hf_name: meta-llama/Llama-3.1-8B
  layers: model.model.layers[0..31]    # 32 层
  d_model: 4096
  n_heads: 32 (GQA: 32 query heads, 8 KV heads)
  vocab: 128256
  layer_output: output[0]  # (batch, seq, 4096)

LLaMA-3.2-1B (Student):
  hf_name: meta-llama/Llama-3.2-1B
  layers: model.model.layers[0..15]    # 16 层
  d_model: 2048
  n_heads: 32 (GQA: 32 query heads, 8 KV heads)
  vocab: 128256 (共享词表)
```

### 4.4 跨架构关键差异对比

| 特性 | Qwen3 | LLaMA 3 |
|------|-------|---------|
| GQA 分组 | 32Q/4KV (teacher), 16Q/8KV (student) | 32Q/8KV (统一) |
| QK-Norm | ✅ 有 | ❌ 无 |
| 词表大小 | 151,936 | 128,256 |
| Teacher 层数 | 36 | 32 |
| Student 层数 | 28 | 16 |
| Teacher d_model | 4096 | 4096 |
| Student d_model | 1024 | 2048 |

两个模型族的 HuggingFace 接口几乎一致，Adapter 差异仅在配置参数。
**Tokenizer 不同**，IOI 名字池等需要针对每个 tokenizer 分别验证。

---

## 五、数据集体系

### 5.1 两类数据

```
对比对数据（用于 RCID 正则 + 可解释性实验）:
  手工构造: IOI, Factual Probing, WinoGrande, Simple Math
  自动构造: EntitySwap, NumberPerturb, LLMGenerate（从大规模数据生成）

大规模指令数据（用于主 KL 蒸馏）:
  Alpaca-52K（初始验证）
  SlimOrca 100K 子集（扩展验证）
```

### 5.2 IOI（Indirect Object Identification）

```
Clean:   "When Mary and John went to the store, John gave a drink to"  → Mary
Corrupt: "When Mary and John went to the store, Mary gave a drink to"  → ???
区别仅在 S2 位置。
```

名字池需在**对应 tokenizer** 中验证为单 token。

### 5.3 Factual Knowledge Probing

```
Clean:   "The capital of France is"            → Paris
Corrupt: "The capital of Germany is"           → Berlin
区别仅在国家名。
```

### 5.4 WinoGrande

```
Clean:   "The trophy doesn't fit in the suitcase because it is too big."  → it = trophy
Corrupt: "The trophy doesn't fit in the suitcase because it is too small." → it = suitcase
区别仅在关键形容词。
```

### 5.5 通用对比对接口

所有对比数据集继承 `ContrastiveDataset` 基类：

```python
class ContrastiveDataset:
    clean_ids: torch.Tensor           # (N, seq_len)
    corrupt_ids: torch.Tensor         # (N, seq_len)
    answer_pos: torch.Tensor          # (N,)
    correct_token_id: torch.Tensor    # (N,)
    wrong_token_id: torch.Tensor      # (N,)
    key_positions: dict[str, torch.Tensor]
    is_modified: dict[str, bool]
    model_family: str
```

### 5.6 大规模指令数据

`InstructionDataset` 用于主 KL 蒸馏数据流：

```python
class InstructionDataset(torch.utils.data.Dataset):
    """从 HuggingFace 加载 Alpaca/SlimOrca 等指令数据。"""
    def __getitem__(self, idx) -> dict:
        return {
            "input_ids": torch.Tensor,      # (seq_len,)
            "attention_mask": torch.Tensor,  # (seq_len,)
            "labels_mask": torch.Tensor,     # (seq_len,) response tokens=1
        }
```

Alpaca 格式 prompt 模板：
```
Below is an instruction that describes a task.

### Instruction:
{instruction}

### Response:
{output}
```

---

## 六、自动对比对构造

### 6.1 三种生成器

| 生成器 | 原理 | 适用场景 | 适用 Benchmark |
|--------|------|---------|---------------|
| `EntitySwapGenerator` | NER + 实体替换 | 事实知识 | TriviaQA, NQ |
| `NumberPerturbGenerator` | 正则匹配 + 数字扰动 | 数学推理 | GSM8K, MATH |
| `LLMGenerator` | 教师 LLM 自动生成最小改动变体 | 通用 | 任意 |

所有生成器继承 `ContrastivePairGenerator` 基类：

```python
class ContrastivePairGenerator(ABC):
    def generate(self, text: str) -> list[tuple[str, str]]: ...
    def validate_pair(self, clean: str, corrupt: str) -> bool: ...
    def batch_generate(self, texts: list[str], max_pairs_per_text: int = 3) -> list[tuple[str, str]]: ...
```

### 6.2 质量验证

`ContrastivePairValidator` 确保每个对比对满足：
1. **教师输出改变**：teacher 在 clean 和 corrupt 上的 top-1 预测不同
2. **编辑距离小**：token 级别差异 ≤ 5
3. **长度一致**：两个序列 token 数差 ≤ 2
4. **因果效应存在**：teacher 残差流中存在非 trivial 的对比差值

### 6.3 生成流程

```
大规模指令数据 → 三种生成器 → 候选对比对 → 质量验证 → 过滤后的对比对 JSON
                                                              ↓
                                                    GeneratedContrastiveDataset
                                                              ↓
                                                        RCID 训练
```

生成脚本：`scripts/generate_contrastive_pairs.py`

---

## 七、实验设计（五个实验）

### 实验 1：现有蒸馏方法保留了 teacher 的机制吗？

**主线 A 的核心证据。数据：IOI + Factual Probing。**

```
1. 在 teacher 上搜索因果检查点（Read）
2. 用 StandardKD（全序列 KL）蒸馏 student（3 seeds）
3. 对每个 student，在每个检查点做因果干预（Write）
4. 计算 causal_consistency = Pearson(Δ_T, Δ_S)
预期：student 准确率接近 teacher，但因果一致性低。
```

### 实验 2：RCID 在受控环境下改善机制传递

**主线 B 的核心证据。数据：IOI + Factual Probing + WinoGrande。**

4 种方法对比（均使用全序列 KL）：
- StandardKD（只看输出）
- FitNets（所有层匹配完整表示）
- Informed FitNets（因果位置匹配完整表示）
- RCID（因果位置匹配对比差值）

评估指标：task accuracy、causal consistency、information purity。

因素拆解：
- FitNets → Informed FitNets = "选对位置"的贡献
- Informed FitNets → RCID = "匹配对比差值"的贡献

### 实验 3：大规模蒸馏 + RCID 正则（核心实用性实验）

**主线 C 的核心证据。数据：Alpaca-52K + 自动对比对。**

```
方法对比：
  standard_kd                  — 全序列 KL only
  standard_kd_rcid             — KL + λ·RCID（对比差值匹配）
  standard_kd_fitnets          — KL + FitNets（全层表示匹配）
  standard_kd_informed_fitnets — KL + InformedFitNets（因果位置表示匹配）

RCID 对比对：从 Alpaca 数据中自动构造 ~5K 对
每 N 步从对比对数据中采样一个 batch 计算 RCID loss

评估（真实 benchmark）：
  MMLU (5-shot), GSM8K (8-shot), ARC-Challenge (25-shot),
  HellaSwag, WinoGrande, TruthfulQA

评估（可解释性，在对比对子集上）：
  因果一致性, 信息纯度

预期：RCID 正则在某些能力（特别是推理类）上带来提升
```

### 实验 4：OOD 鲁棒性

收集实验 2、3 中所有 student 的 (causal_consistency, ood_degradation)。

- 在 Alpaca 上训练，在其他领域指令数据上评估性能衰减
- 对比 Standard KD vs Standard KD + RCID 的 OOD degradation
- 如果正相关 → 保留 teacher 机制 → 更鲁棒的泛化

### 实验 5：机制分析 + 跨架构泛化

**主线 D 的核心证据。**

- 对实验 3 中大规模蒸馏的模型做因果分析（在对比对子集上计算因果一致性）
- 证明 RCID 正则确实改善了内部机制对齐
- 用信息纯度指标验证对比差值 vs 完整表示的优势

**跨架构验证（LLaMA 3）**：在 LLaMA-3-8B → LLaMA-3.2-1B 上复现实验 2 核心部分（IOI + Factual Probing，StandardKD vs RCID）。

---

## 八、核心模块实现规范

### 8.1 模型适配器 (`models/adapter.py`)

```python
class Qwen3Adapter(ModelAdapter):
    def get_layers(self, model): return model.model.layers
    def get_residual_hook_point(self, model, layer_idx): return model.model.layers[layer_idx]
    def parse_layer_output(self, output): return output[0]  # (batch, seq, d_model)
    def get_embed_tokens(self, model): return model.model.embed_tokens
    def get_lm_head(self, model): return model.lm_head
    def get_num_layers(self, model): return len(model.model.layers)
    def get_hidden_size(self, model): return model.config.hidden_size

# LLaMA3Adapter 接口完全相同
```

### 8.2 因果干预 (`circuit/intervention.py`)

```python
def patch_and_run(model, adapter, clean_input, patch_value, layer, token_pos):
    """Pearl do-operator in transformer residual stream."""
    # hook → patch → forward → unhook

def compute_causal_effect(model, adapter, clean_input, corrupt_input,
                          layer, token_pos, answer_pos, correct_id, wrong_id):
    """Δ = logit_diff(original) - logit_diff(patched)"""
```

### 8.3 RCID 损失 (`distillation/rcid_loss.py`)

```python
class RCIDLoss(nn.Module):
    """因果检查点上匹配 teacher/student 对比差值方向。
    W: buffer, 冻结 Procrustes 矩阵 (d_T, d_S)
    教师痕迹: 预计算, detached
    学生前向: 保留梯度
    """
    def forward(self, teacher_imprints, student_clean_residuals, student_corrupt_residuals):
        # 对每个 checkpoint: d_S = h_clean - h_corrupt, aligned = d_S @ W.T
        # loss = ||normalize(aligned) - normalize(d_T)||^2
```

### 8.4 大规模蒸馏训练器 (`distillation/scalable_trainer.py`)

```python
class ScalableDistillationTrainer:
    """双数据流训练器：全序列 KL + 可选 RCID 正则。

    主数据流: InstructionDataset → 全序列 KL 蒸馏
    RCID 数据流: GeneratedContrastiveDataset → 因果差值匹配（每 N 步）

    方法路由:
      standard_kd              → 无对比对, lambda=0       → 纯 KL
      standard_kd_rcid         → 对比对 + checkpoints     → KL + RCID
      standard_kd_fitnets      → 对比对 + 全层 W          → KL + FitNets
      standard_kd_informed_fitnets → 对比对 + checkpoints + W → KL + InformedFitNets
    """
    def train(self) -> dict[str, list[float]]:
        for epoch:
            for step, main_batch in main_loader:
                kl_loss = kd_loss_fn(t_logits, s_logits, mask=attn_mask)
                rcid_loss = 0
                if use_rcid and step % rcid_every == 0:
                    rcid_loss = _compute_rcid_loss(next(rcid_iter))
                total = kl_loss + lambda_rcid * rcid_loss
                total.backward(); optimizer.step(); scheduler.step()
```

支持 fp16 (torch.amp.autocast + GradScaler)、gradient accumulation、cosine scheduler with warmup。

### 8.5 Toy Data 训练器 (`distillation/trainer.py`)

```python
class UnifiedTrainer:
    """用于 toy data 实验（IOI/Factual/WinoGrande）的训练器。
    支持 standard_kd, fitnets, informed_fitnets, rcid 四种方法。
    KL 模式由 config["kl_mode"] 控制，默认 "sequence"。
    接受可选的 tokenizer 参数用于生成 attention mask。
    """
```

### 8.6 Procrustes 对齐 (`alignment/procrustes.py`)

```python
def procrustes_align(source, target) -> torch.Tensor:
    """W* = argmin ||target - source @ W^T||_F
    source: (N, d_S), target: (N, d_T) → W: (d_T, d_S)"""
```

---

## 九、项目结构

```
rcid/
├── CLAUDE.md
├── pyproject.toml
├── configs/
│   ├── master.yaml
│   ├── qwen3.yaml
│   ├── llama3.yaml
│   ├── large_scale.yaml                    # 大规模蒸馏配置
│   ├── exp1_mechanism_preservation.yaml
│   ├── exp2_rcid_comparison.yaml
│   ├── exp3_robustness_correlation.yaml
│   ├── exp4_information_purity.yaml
│   └── exp5_cross_architecture.yaml
├── src/
│   └── rcid/
│       ├── __init__.py
│       ├── models/
│       │   ├── __init__.py
│       │   ├── adapter.py                  # ModelAdapter 基类 + Qwen3Adapter + LLaMA3Adapter
│       │   ├── teacher.py
│       │   └── student.py
│       ├── circuit/
│       │   ├── __init__.py
│       │   ├── patching.py                 # Read: extract_contrastive_differences
│       │   ├── intervention.py             # Write: patch_and_run, compute_causal_effect
│       │   ├── checkpoint_selection.py
│       │   └── contrastive.py              # ContrastiveDataset 基类
│       ├── alignment/
│       │   ├── __init__.py
│       │   ├── procrustes.py
│       │   ├── layer_matching.py
│       │   └── cka.py
│       ├── distillation/
│       │   ├── __init__.py
│       │   ├── rcid_loss.py                # RCID loss
│       │   ├── baselines.py                # StandardKDLoss (全序列KL), FitNets, InformedFitNets
│       │   ├── trainer.py                  # UnifiedTrainer (toy data 实验)
│       │   └── scalable_trainer.py         # ScalableDistillationTrainer (大规模蒸馏)
│       ├── data/
│       │   ├── __init__.py
│       │   ├── ioi.py
│       │   ├── factual_probing.py
│       │   ├── winogrande.py
│       │   ├── simple_math.py
│       │   ├── instruction_dataset.py      # 大规模指令数据加载 (Alpaca/SlimOrca)
│       │   ├── contrastive_generators.py   # 三种自动对比对生成器
│       │   ├── contrastive_validator.py    # 对比对质量验证
│       │   └── generated_contrastive.py    # 从 JSON 加载自动生成的对比对
│       ├── eval/
│       │   ├── __init__.py
│       │   ├── causal_consistency.py
│       │   ├── task_accuracy.py
│       │   ├── ood_robustness.py
│       │   ├── information_purity.py
│       │   └── perplexity.py
│       └── visualization/
│           ├── __init__.py
│           └── paper_figures.py
├── scripts/
│   ├── pilot_validation.py
│   ├── run_exp1.py
│   ├── run_exp2.py
│   ├── run_exp3.py                         # OOD 鲁棒性相关性
│   ├── run_exp4.py
│   ├── run_exp5_cross_arch.py
│   ├── run_all.py
│   ├── generate_contrastive_pairs.py       # 从大规模数据生成对比对
│   ├── run_large_scale_distill.py          # 大规模蒸馏实验 (实验 3)
│   ├── eval_benchmarks.py                  # Benchmark 评估 (MMLU/GSM8K/ARC 等)
│   ├── run_full_pipeline.sh                # 一键运行完整大规模蒸馏流程
│   └── check_environment.py                # 环境依赖检查
├── tests/
│   ├── conftest.py                         # TinyTransformerModel, TinyAdapter
│   ├── test_models.py
│   ├── test_data.py
│   ├── test_circuit.py
│   ├── test_alignment.py
│   ├── test_distillation.py
│   ├── test_eval.py
│   ├── test_large_scale.py                 # 大规模蒸馏集成测试
│   └── test_integration.py
└── outputs/
    └── results/
        ├── exp1/
        ├── exp2/
        ├── exp3/
        ├── exp4/
        ├── exp5_cross_arch/
        └── large_scale/                    # 大规模蒸馏结果
```

---

## 十、实验配置

### 10.1 模型

```yaml
# 主实验
qwen3_teacher:
  name: Qwen/Qwen3-8B
  n_layers: 36
  d_model: 4096

qwen3_student:
  name: Qwen/Qwen3-0.6B
  n_layers: 28
  d_model: 1024

# 泛化验证
llama3_teacher:
  name: meta-llama/Llama-3.1-8B
  n_layers: 32
  d_model: 4096

llama3_student:
  name: meta-llama/Llama-3.2-1B
  n_layers: 16
  d_model: 2048
```

### 10.2 实验矩阵

```yaml
# Toy data 实验 (Qwen3) — 实验 1+2
methods: [standard_kd, fitnets, informed_fitnets, rcid]
tasks: [ioi, factual_probing, winogrande]
seeds: [42, 123, 456]
# 4 methods × 3 tasks × 3 seeds = 36 runs

# 大规模蒸馏 (Qwen3) — 实验 3
large_scale_methods: [standard_kd, standard_kd_rcid, standard_kd_fitnets, standard_kd_informed_fitnets]
large_scale_data: [alpaca_52k]
lambda_rcid_search: [0.01, 0.05, 0.1, 0.5, 1.0]
seeds: [42, 123, 456]
# 先跑 lambda=0.1 的 standard_kd vs standard_kd_rcid，再展开

# 泛化验证 (LLaMA 3) — 实验 5
llama3_methods: [standard_kd, rcid]
llama3_tasks: [ioi, factual_probing]
llama3_seeds: [42, 123, 456]
# 2 × 2 × 3 = 12 runs
```

### 10.3 Toy Data 训练超参数

```yaml
toy_training:
  epochs: 20
  batch_size: 16
  lr: 5e-5
  optimizer: adamw
  scheduler: cosine
  grad_clip: 1.0
  lambda_kl: 1.0
  lambda_rcid: 1.0
  kl_mode: sequence
  temperature: 2.0
  fp16: true
```

### 10.4 大规模蒸馏超参数

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
  kl_mode: sequence
  temperature: 2.0
  lambda_rcid: 0.1                   # grid search: [0.01, 0.05, 0.1, 0.5, 1.0]
  rcid_every_n_steps: 5
  rcid_data_size: 5000
  fp16: true
  save_every_n_epochs: 1
```

### 10.5 多 GPU 策略

```yaml
# 不做分布式训练（模型不需要），而是做实验级并行
# Toy data (Qwen3)
gpu_0: seed=42, method=standard_kd → fitnets → ...
gpu_1: seed=42, method=informed_fitnets → rcid → ...
gpu_2: seed=123, ...
gpu_3: seed=456, ...

# 大规模蒸馏 (Qwen3)
gpu_0: standard_kd, seed=42
gpu_1: standard_kd_rcid (lambda=0.1), seed=42
gpu_2: standard_kd_rcid (lambda=0.05), seed=42
gpu_3: standard_kd_rcid (lambda=0.5), seed=42
```

---

## 十一、Pilot Validation（最先运行！）

```python
# scripts/pilot_validation.py
# 参数: --model_family qwen3 | llama3
# 目标：每种模型 30 分钟内完成

# 1. 加载 teacher，在 IOI 上测试准确率 → 预期 > 95%
# 2. 用对应 tokenizer 检查名字池 → 预期 ≥ 20 个单 token 名字
# 3. 构造 10 个 IOI 对比对，提取各层因果差值范数 → 预期高层 > 底层
# 4. 在一个检查点做 activation patching → 预期 Δ > 0 且显著
# 5. 加载 student（未蒸馏），测试 baseline 准确率
```

---

## 十二、编码规范

### 12.1 类型标注（强制）
所有函数签名完整标注。

### 12.2 Tensor Shape 注释（强制）
```python
residual = adapter.parse_layer_output(output)[:, token_pos, :]  # (batch, d_model)
```

### 12.3 断言检查
```python
assert teacher_imprint.dim() == 2, f"Expected 2D, got {teacher_imprint.dim()}D"
assert loss.isfinite(), f"Loss is {loss.item()}"
```

### 12.4 Hook 生命周期管理
```python
handle = hook_point.register_forward_hook(hook_fn)
try:
    model(input_ids)
finally:
    handle.remove()
```

### 12.5 梯度流管理
- 教师模型：始终 eval + no_grad
- 教师痕迹：预计算后 detach
- W 矩阵：注册为 buffer
- 学生前向传播：保留梯度

### 12.6 数值稳定性
```python
eps = 1e-8
norm = x.norm(dim=-1, keepdim=True).clamp(min=eps)
```

### 12.7 不硬编码架构
所有模型交互通过 ModelAdapter，禁止直接写 `model.model.layers[l]`（除 adapter 内部外）。

### 12.8 文件长度
每个 `.py` 不超过 300 行。

### 12.9 模型族参数化
涉及模型名、层数、维度的地方从 config 读取或通过 adapter 查询。
数据集类接受 tokenizer 参数，不假设特定词表。

---

## 十三、相关工作定位

**vs Wu et al. (2022) Causal Distillation (DIITO)**：
DIITO 用 interchange intervention + 预定义因果模型。
RCID 用 activation patching + 自动检查点搜索。RCID 不需要预先知道 teacher 的因果结构。

**vs Prakash et al. (2025) Circuit Distillation**：
他们用 CKA 匹配电路组件的完整激活。
我们用 Procrustes 匹配对比差值（过滤了任务无关信息）。
他们只在 toy task 上验证；我们扩展到大规模蒸馏 + 真实 benchmark。

**vs Dunefsky et al. (2025) Distilled Circuits**：
纯分析工作。我们既诊断（实验 1）又治疗（RCID），且验证实用性（实验 3）。

**vs 标准蒸馏方法（DistilBERT、TinyBERT、MiniLM）**：
它们只做输出/表示匹配。RCID 额外对齐因果机制，作为正则项互补。

---

## 十四、实施优先级

```
P0（基础修复 — 已完成 ✅）:
  ✅ 修复 StandardKDLoss → 全序列 KL + mask 支持
  ✅ 修复 UnifiedTrainer → kl_mode 配置 + tokenizer 参数
  ✅ 实现 ScalableDistillationTrainer
  ✅ 实现 InstructionDataset、GeneratedContrastiveDataset
  ✅ 实现 contrastive_generators + contrastive_validator
  ✅ 实现 run_large_scale_distill.py + eval_benchmarks.py + generate_contrastive_pairs.py

P1（Toy data 实验验证 — 当前阶段）:
  Pilot validation（Qwen3 + LLaMA 3）
  重跑 toy data（验证全序列 KL 下 RCID 仍有效）
  运行实验 1 + 2（IOI 先行，扩展到 Factual + WinoGrande）

P2（大规模蒸馏实验运行）:
  从 Alpaca 生成 ~5K 对比对
  Standard KD 基线 on Alpaca-52K
  Standard KD + RCID on Alpaca-52K
  λ_RCID 超参数搜索：[0.01, 0.05, 0.1, 0.5, 1.0]
  MMLU/GSM8K/ARC/HellaSwag/WinoGrande/TruthfulQA 评估

P3（完善论文 — Qwen3）:
  OOD 鲁棒性实验
  因果分析（大规模蒸馏模型）
  信息纯度验证

P4（跨架构泛化 — LLaMA 3）:
  LLaMA 3 数据集适配
  实验 2 核心部分复现（IOI + Factual）
  跨架构对比表

P5（论文材料）:
  论文图表、结果分析、讨论章节
```

---

## 十五、风险与应对

### 风险 1：RCID 在大规模蒸馏上无提升
**应对**：
- λ_RCID grid search [0.01, 1.0]
- 分能力维度分析（可能在数学推理有效、其他无效 → 仍是有趣发现）
- 即使性能提升不大，因果一致性提升 + 可解释性仍有价值

### 风险 2：自动对比对质量不够
**应对**：
- 严格质量验证 pipeline (ContrastivePairValidator)
- 回退到半自动（基于 benchmark 结构化构造）
- GSM8K/ARC 等本身有结构化格式，易做最小改动

### 风险 3：计算资源不够
**应对**：
- 先用 Alpaca-52K（小数据）快速验证
- fp16 + gradient checkpointing
- RCID 正则只需少量对比对，不显著增加计算

---

## 十六、符号速查

| 符号 | 含义 | 代码 |
|------|------|------|
| $h_{l,t}$ | 第 $l$ 层第 $t$ 位置的残差流 | `residual[layer][:, pos, :]` |
| $d_{l,t}^T$ | 教师对比差值 | `teacher_diff` |
| $d_{l,t}^S$ | 学生对比差值 | `student_diff` |
| $W^*$ | 冻结 Procrustes 矩阵 (d_T, d_S) | `W` (buffer) |
| $\hat{l}$ | 学生中与教师层 $l$ 匹配的层 | `layer_mapping[l]` |
| $\Delta_T$ | teacher patching 后行为变化 | `delta_T` |
| $\Delta_S$ | student patching 后行为变化 | `delta_S` |
| CC | 因果一致性 | `causal_consistency` |
| $\lambda$ | RCID 正则权重 | `lambda_rcid` |

---

## 十七、依赖

```toml
[project]
name = "rcid"
requires-python = ">=3.10"
dependencies = [
    "torch>=2.1",
    "transformers>=4.45",
    "accelerate>=0.27",
    "datasets>=2.16",
    "omegaconf>=2.3",
    "wandb>=0.16",
    "scipy>=1.11",
    "scikit-learn>=1.3",
    "matplotlib>=3.8",
    "seaborn>=0.13",
    "pytest>=7.4",
    "tqdm>=4.66",
]

[project.optional-dependencies]
eval = ["lm-eval>=0.4"]  # EleutherAI lm-evaluation-harness
```

---

## 十八、论文结构映射

```
Section 1: Introduction
  → 四条主线叙事：发现问题 → 解决方法 → 实用验证 → 跨架构泛化

Section 2: Background & Mechanistic Consistency Metric
  → 2.1-2.2（Read/Write + 因果一致性指标）

Section 3: Method (RCID)
  → 2.3-2.4（RCID 损失 + 检查点选择 + Procrustes 对齐）
  → 6.1-6.3（自动对比对构造）

Section 4: Experiments
  4.1 Setup: 两个模型族、toy data + 大规模数据
  4.2 Exp1: 现有方法机制不一致
  4.3 Exp2: RCID 改善机制传递（toy data，因素拆解）
  4.4 Exp3: 大规模蒸馏 + RCID 正则（核心实用性结果）
  4.5 Exp4: OOD 鲁棒性
  4.6 Exp5: 因果分析 + 跨架构泛化（LLaMA 3）

Section 5: Discussion
  → 贡献总结、局限性、未来方向（MoE、更大规模）

Appendix:
  → 自动对比对构造细节
  → 超参数搜索
  → 更多可视化
```

---

## 十九、贡献列表

1. **Mechanistic Consistency Metric**：首次提出量化蒸馏中推理机制保留程度的指标
2. **RCID Loss**：基于因果差值的机制对齐正则项，可即插即用到标准蒸馏流程
3. **自动对比对构造**：通过实体替换、数字扰动、LLM 生成三种方式使方法可扩展到大规模数据
4. **双重验证**：在 toy data 上验证可解释性（因果一致性、信息纯度），在大规模 benchmark 上验证实用性（MMLU、GSM8K、ARC）
5. **跨架构泛化**：在 Qwen3 和 LLaMA 3 两种架构上验证方法有效性
