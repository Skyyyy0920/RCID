# RCID 项目：下一步完整方案

## 一、现状诊断

### 1.1 已完成的工作

项目已搭建了完整的代码框架，包括：
- **ModelAdapter 抽象层**：支持 Qwen3 和 LLaMA3 两种架构
- **核心模块**：patching.py（Read）、intervention.py（Write）、checkpoint_selection.py、procrustes.py
- **蒸馏方法**：StandardKD、FitNets、InformedFitNets、RCID 四种方法的 loss 和统一 trainer
- **评估体系**：causal_consistency、task_accuracy、perplexity、information_purity
- **数据集**：IOI、Factual Probing、WinoGrande 的对比数据集接口
- **实验脚本**：5 个实验的运行脚本

### 1.2 核心问题诊断

经过深入分析代码和瓶颈文档，我识别出以下关键问题：

#### 问题 1：KL Loss 仅针对 answer position 的最后一个 token

当前 trainer 的 `_compute_loss` 中：

```python
batch_idx = torch.arange(bs, device=clean.device)
t_at_ans = t_logits[batch_idx, answer_pos]  # (batch, vocab)
s_at_ans = s_logits[batch_idx, answer_pos]   # (batch, vocab)
kd_loss = self.kd_loss_fn(t_at_ans, s_at_ans)
```

**这是一个严重的设计缺陷。** 主流蒸馏方法（如 DistilBERT、MiniLM、TinyBERT 以及最新的 SDFT/SDPO 等）都是在**所有 token 位置**上计算 KL loss（sequence-level KL），而不是仅在单个 answer token 上。仅在最后一个 token 上做 KL 蒸馏会导致：

- 学生模型只学到了"在该位置输出正确答案"，但丢失了序列中间的语言建模能力
- 与主流蒸馏实践不一致，reviewer 会质疑实验的公平性
- 在大规模蒸馏场景下完全不适用（真实的 instruction-following 数据没有单一 answer position）

#### 问题 2：训练数据仅为 toy data，无法泛化

目前所有蒸馏实验都只在 IOI/Factual/WinoGrande 等合成对比数据集上训练。这些数据集的特点是：
- 句子短、模式单一
- 训练集和测试集的分布高度相似
- 无法验证 RCID 在真实场景下的效用

#### 问题 3：对比对需要手工构造，不可扩展

每个任务都需要设计 clean/corrupt pair 的构造规则，限制了方法的通用性。

---

## 二、重新定位 RCID

### 2.1 新定位：即插即用的机制对齐正则项

RCID 不应该是一个独立的蒸馏方法，而应该是一个**可插拔的正则项**，嫁接到任何标准蒸馏流程上。新的总损失函数为：

$$\mathcal{L} = \underbrace{\mathcal{L}_{\text{KL}}^{\text{seq-level}}}_{\text{在大规模指令数据上，全序列 KL}} + \lambda \cdot \underbrace{\mathcal{L}_{\text{RCID}}}_{\text{在少量对比对上，因果差值匹配}}$$

**叙事**："标准蒸馏让学生**表现得像**教师（输出分布对齐），RCID 正则让学生**思考得像**教师（内部推理机制对齐），两者互补。"

### 2.2 两阶段实验设计

| 阶段 | 目的 | 数据 | 评估 |
|------|------|------|------|
| 阶段一：可解释性验证 | 证明 RCID 改善内部机制 | IOI 等 toy data | 因果一致性、信息纯度 |
| 阶段二：实用性验证 | 证明 RCID 提升真实性能 | 大规模指令数据 + 自动对比对 | MMLU, GSM8K, ARC 等 benchmark |

---

## 三、代码修改方案

### 3.1 修复 KL Loss：全序列蒸馏

**核心修改**：KL loss 应该在所有 token 位置上计算，而不是仅在 answer position。

#### 3.1.1 修改 `StandardKDLoss`

```python
class StandardKDLoss(nn.Module):
    """Standard KD via sequence-level KL divergence."""
    
    def __init__(self, temperature: float = 2.0) -> None:
        super().__init__()
        self.temperature = temperature

    def forward(
        self,
        teacher_logits: torch.Tensor,  # (batch, seq_len, vocab)
        student_logits: torch.Tensor,  # (batch, seq_len, vocab)
        mask: torch.Tensor | None = None,  # (batch, seq_len), 1 for valid tokens
    ) -> torch.Tensor:
        T = self.temperature
        # 在所有有效 token 位置上计算 KL
        t_probs = F.softmax(teacher_logits / T, dim=-1)
        s_log_probs = F.log_softmax(student_logits / T, dim=-1)
        
        # per-token KL: (batch, seq_len)
        per_token_kl = F.kl_div(
            s_log_probs, t_probs, reduction="none"
        ).sum(dim=-1)  # sum over vocab
        
        if mask is not None:
            per_token_kl = per_token_kl * mask
            loss = per_token_kl.sum() / mask.sum().clamp(min=1)
        else:
            loss = per_token_kl.mean()
        
        return loss * (T * T)
```

#### 3.1.2 修改 Trainer 的 `_compute_loss`

```python
def _compute_loss(self, clean, corrupt, answer_pos, s_layers, bs, 
                  lambda_kl, lambda_method, history):
    # Teacher forward (no grad)
    with torch.no_grad():
        t_logits = self.teacher(clean).logits  # (batch, seq, vocab)

    # Student clean forward + residuals
    if self.method in ("fitnets", "informed_fitnets", "rcid") and s_layers:
        s_clean_cache, s_logits = self._collect_student_residuals(clean, s_layers)
    else:
        s_logits = self.student(clean).logits
        s_clean_cache = {}

    # 全序列 KL loss（修改后）
    # 对于 toy data：可以使用 attention mask 或全序列
    # 对于大规模数据：使用标准的 causal LM mask
    kd_loss = self.kd_loss_fn(t_logits, s_logits, mask=self.get_loss_mask(clean))

    # RCID loss 不变 — 仍在特定对比对的因果检查点上计算
    # ...（method-specific loss 部分保持不变）
```

#### 3.1.3 向后兼容

为保留 toy data 实验的可比性，可以增加一个配置选项：

```yaml
kl_mode: "sequence"  # "sequence" (全序列) | "answer_only" (仅 answer position)
```

但**所有正式实验**（包括 toy data 上的实验）都应该使用 `sequence` 模式，因为：
1. 这与主流蒸馏方法一致，确保公平比较
2. 全序列 KL 保护学生的语言建模能力
3. RCID 的价值通过附加正则项体现，而非通过削弱基线来人为放大

### 3.2 新增大规模蒸馏训练器

需要新建一个面向大规模数据的训练器，与现有的 toy-data trainer 分离：

```python
class ScalableDistillationTrainer:
    """大规模蒸馏训练器，支持 RCID 正则项。
    
    主数据流：在大规模指令数据上做全序列 KL 蒸馏
    辅助数据流：在对比对数据上做 RCID 正则
    两者交替采样
    """
    
    def __init__(
        self,
        teacher, student,
        teacher_adapter, student_adapter,
        main_dataset,         # 大规模指令数据（HuggingFace Dataset）
        contrastive_dataset,  # 对比对数据（ContrastiveDataset）
        config,
        # RCID 相关
        checkpoints=None,
        layer_mapping=None,
        W_matrices=None,
    ):
        ...
    
    def train_step(self, batch):
        """每个 step: 
        1. 从 main_dataset 采样一个 batch，计算 KL loss
        2. 每 N 步从 contrastive_dataset 采样，计算 RCID loss
        3. 合并 loss，反向传播
        """
        ...
```

### 3.3 数据流设计

```
┌──────────────────────────────────────────┐
│           大规模指令数据                    │
│    (Alpaca-52K / OpenOrca / SlimOrca)     │
│                                          │
│    每个样本: (instruction, response)       │
│    KL Loss: 全序列 teacher-student        │
└────────────────┬─────────────────────────┘
                 │
                 ▼
        ┌────────────────┐
        │  Total Loss =  │
        │  L_KL + λ·L_RCID │
        └────────┬───────┘
                 │
                 ▲
┌────────────────┴─────────────────────────┐
│         自动构造的对比对数据                 │
│    (从大规模数据中自动生成)                  │
│                                          │
│    每个样本: (clean, corrupt, checkpoints) │
│    RCID Loss: 因果差值方向匹配              │
└──────────────────────────────────────────┘
```

---

## 四、自动化对比对构造

这是让 RCID 可扩展的关键。我推荐以下三种互补的方法：

### 4.1 方法 A：实体替换（适用于事实知识类）

**原理**：用 NER 自动找到关键实体，替换为同类别的其他实体。

**实现方案**：

```python
class EntitySwapContrastiveGenerator:
    """基于实体替换的对比对自动生成器。"""
    
    def __init__(self, teacher, tokenizer, ner_model="dslim/bert-base-NER"):
        self.teacher = teacher
        self.tokenizer = tokenizer
        self.ner_pipeline = pipeline("ner", model=ner_model)
        # 预建实体替换表
        self.entity_pool = {
            "PER": ["Alice", "Bob", "Charlie", ...],
            "ORG": ["Google", "Apple", "Microsoft", ...],
            "LOC": ["France", "Germany", "Japan", ...],
        }
    
    def generate(self, text: str) -> list[tuple[str, str]]:
        """返回 [(clean_text, corrupt_text), ...] 对比对列表"""
        entities = self.ner_pipeline(text)
        pairs = []
        for ent in entities:
            category = ent["entity_group"]
            replacements = [e for e in self.entity_pool[category] 
                          if e != ent["word"]]
            for replacement in random.sample(replacements, min(3, len(replacements))):
                corrupt = text.replace(ent["word"], replacement)
                # 验证教师模型输出确实改变
                if self._teacher_output_changed(text, corrupt):
                    pairs.append((text, corrupt))
        return pairs
    
    def _teacher_output_changed(self, clean, corrupt) -> bool:
        """验证教师在 clean 和 corrupt 上的输出（top-1 token）不同"""
        with torch.no_grad():
            clean_ids = self.tokenizer(clean, return_tensors="pt")
            corrupt_ids = self.tokenizer(corrupt, return_tensors="pt")
            clean_pred = self.teacher(**clean_ids).logits[:, -1].argmax(-1)
            corrupt_pred = self.teacher(**corrupt_ids).logits[:, -1].argmax(-1)
        return clean_pred.item() != corrupt_pred.item()
```

**适用 Benchmark**：TriviaQA, NaturalQuestions, 事实知识相关任务

### 4.2 方法 B：关键数字/条件扰动（适用于数学推理类）

```python
class MathContrastiveGenerator:
    """基于数字扰动的对比对生成器。"""
    
    def generate(self, text: str) -> list[tuple[str, str]]:
        # 用正则找到所有数字
        numbers = re.findall(r'\b\d+\b', text)
        pairs = []
        for num in numbers:
            # 替换为附近的数字
            for delta in [-3, -1, 1, 3]:
                new_num = str(int(num) + delta)
                corrupt = text.replace(num, new_num, 1)
                if self._teacher_output_changed(text, corrupt):
                    pairs.append((text, corrupt))
        return pairs
```

**适用 Benchmark**：GSM8K, MATH

### 4.3 方法 C：LLM 自动生成对比对（最通用）

这是最灵活也最推荐的方法，可以覆盖任意任务类型：

```python
class LLMContrastiveGenerator:
    """使用教师 LLM 自动生成最小改动对比对。"""
    
    PROMPT_TEMPLATE = """Given the following text, generate a minimally modified version 
where changing only 1-2 key words leads to a different correct answer/conclusion.

Original: {text}

Requirements:
1. Change only 1-2 words
2. The change must cause the answer/conclusion to differ
3. Return ONLY the modified text, nothing else

Modified:"""
    
    def generate(self, text: str) -> tuple[str, str] | None:
        prompt = self.PROMPT_TEMPLATE.format(text=text)
        corrupt = self.teacher.generate(prompt)
        
        # 质量验证
        if not self._is_minimal_change(text, corrupt):
            return None
        if not self._teacher_output_changed(text, corrupt):
            return None
        return (text, corrupt)
    
    def _is_minimal_change(self, clean, corrupt, max_edit_distance=5) -> bool:
        """确保改动足够小"""
        clean_tokens = clean.split()
        corrupt_tokens = corrupt.split()
        diff_count = sum(1 for a, b in zip(clean_tokens, corrupt_tokens) if a != b)
        return diff_count <= 3 and abs(len(clean_tokens) - len(corrupt_tokens)) <= 1
```

### 4.4 对比对质量验证 Pipeline

无论用哪种方法，都需要一个统一的质量验证流程：

```python
class ContrastivePairValidator:
    """对比对质量验证器。"""
    
    def validate(self, clean: str, corrupt: str) -> dict:
        return {
            "teacher_output_changed": self._check_output_change(clean, corrupt),
            "edit_distance_small": self._check_edit_distance(clean, corrupt),
            "tokens_aligned": self._check_token_alignment(clean, corrupt),
            "causal_effect_exists": self._check_causal_effect(clean, corrupt),
        }
    
    def _check_causal_effect(self, clean, corrupt) -> bool:
        """验证教师模型在某些因果检查点上存在非 trivial 的对比差值"""
        diffs = extract_contrastive_differences(
            self.teacher, self.adapter, 
            clean_ids, corrupt_ids, 
            layers=self.sample_layers
        )
        max_norm = max(d.norm(dim=-1).mean().item() for d in diffs.values())
        return max_norm > self.threshold
```

### 4.5 推荐策略：分能力构造

| 能力维度 | 对比对构造方式 | 训练数据来源 | 评估 Benchmark |
|---------|-------------|------------|--------------|
| 事实知识 | 实体替换（方法 A） | TriviaQA/NQ 训练集 | TriviaQA, NQ |
| 数学推理 | 数字扰动（方法 B） | GSM8K 训练集 | GSM8K, MATH |
| 常识推理 | LLM 生成（方法 C） | HellaSwag/ARC 训练集 | HellaSwag, ARC |
| 指令遵循 | LLM 生成（方法 C） | Alpaca 指令数据 | IFEval |

---

## 五、完整实验方案

### 实验 1（保留）：发现问题 — 现有蒸馏不保留机制

**目的**：建立动机，证明标准蒸馏的 student 虽然准确率接近 teacher，但内部机制不一致。

- **数据**：IOI + Factual Probing
- **方法**：StandardKD（全序列 KL）
- **评估**：task accuracy vs. causal consistency
- **预期**：accuracy 高但 causal consistency 低 → 揭示问题

**修改点**：KL loss 改为全序列，确保基线公平。

### 实验 2（保留）：RCID 在受控环境下改善机制传递

**目的**：在 toy data 上验证 RCID 的因果机制对齐效果。

- **数据**：IOI + Factual Probing + WinoGrande
- **方法**：StandardKD, FitNets, Informed FitNets, RCID
- **评估**：task accuracy, causal consistency, information purity
- **因素拆解**：
  - FitNets → Informed FitNets = 选对位置的贡献
  - Informed FitNets → RCID = 匹配对比差值的贡献

### 实验 3（新增，核心）：大规模蒸馏 + RCID 正则

**目的**：证明 RCID 作为正则项在真实 benchmark 上带来提升。

#### 3a. 实验设置

**模型配对**：
- 主实验：Qwen3-8B → Qwen3-0.6B
- 泛化验证：LLaMA-3-8B → LLaMA-3.2-1B

**主蒸馏数据**（两个选项，按资源选择）：

| 数据集 | 规模 | 优势 | 劣势 |
|-------|------|------|------|
| Alpaca-52K | 52K | 小，训练快 | 质量一般 |
| SlimOrca | 500K+ | 质量高，覆盖广 | 训练慢 |
| OpenHermes-2.5 | 1M | 最大 | 可能过大 |

**推荐**：使用 **Alpaca-52K** 作为初始验证，成功后扩展到 **SlimOrca 的 100K 子集**。

**RCID 对比对数据**：
- 从训练数据中自动构造，每种能力维度约 1K-5K 对
- 总量约 5K-20K 对比对（相对于主数据的 10-40%）

#### 3b. 方法对比

| 方法 | 描述 |
|------|------|
| Standard KD | 全序列 KL loss |
| Standard KD + FitNets | KL + 全层表示匹配 |
| Standard KD + Informed FitNets | KL + 因果位置表示匹配 |
| **Standard KD + RCID** | KL + 因果位置对比差值匹配 |

#### 3c. 评估

**实用性评估**（真实 benchmark）：

| Benchmark | 能力维度 | 评估方式 |
|-----------|---------|---------|
| MMLU | 综合知识 | 5-shot accuracy |
| GSM8K | 数学推理 | exact match |
| ARC-Challenge | 科学推理 | accuracy |
| HellaSwag | 常识推理 | accuracy |
| TriviaQA | 事实知识 | exact match |
| WinoGrande | 代词消解 | accuracy |

**可解释性评估**（在对比对子集上）：
- 因果一致性（Causal Consistency）
- 信息纯度（Information Purity）

#### 3d. 训练超参数

```yaml
# 大规模蒸馏配置
large_scale_training:
  main_data: "tatsu-lab/alpaca"  # 或 SlimOrca
  epochs: 3
  batch_size: 8  # per GPU
  gradient_accumulation: 4  # effective batch = 32
  lr: 2e-5
  scheduler: cosine_with_warmup
  warmup_ratio: 0.03
  max_seq_len: 512
  
  kl_temperature: 2.0
  kl_mode: "sequence"  # 全序列 KL
  
  # RCID 正则
  rcid_enabled: true
  lambda_rcid: 0.1  # 需要 grid search: [0.01, 0.05, 0.1, 0.5, 1.0]
  rcid_batch_ratio: 0.2  # 每 5 个主 batch 插入 1 个 RCID batch
  rcid_data_size: 5000  # 自动构造的对比对数量
  
  # 对比对自动构造
  contrastive_generation:
    method: "entity_swap+number_perturb+llm_generate"
    per_capability: 1000  # 每种能力维度 1000 对
    validation_threshold: 0.1  # 因果差值范数阈值
```

### 实验 4（新增）：OOD 鲁棒性

**目的**：证明 RCID 正则过的学生在 distribution shift 下更鲁棒。

- 在 Alpaca 上训练，在 **其他领域的指令数据** 上评估
- 对比 Standard KD vs Standard KD + RCID 的性能衰减

### 实验 5（保留）：为什么 RCID 有效 — 机制分析

**目的**：用因果一致性指标分析实验 3 中的大规模蒸馏模型。

- 从大规模蒸馏模型中，抽样对比对
- 计算因果一致性，证明 RCID 确实改善了内部机制

---

## 六、实施优先级和时间线

### Phase 0：修复基础（预计 2-3 天）

1. **修复 KL Loss**：改为全序列 KL，保留 answer_only 模式作为可选项
2. **修复 Trainer**：支持 attention mask，支持可变长度序列
3. **重跑 toy data 实验**（实验 1+2）验证全序列 KL 下 RCID 仍然有效

### Phase 1：对比对自动构造（预计 3-5 天）

1. 实现 `EntitySwapContrastiveGenerator`
2. 实现 `MathContrastiveGenerator`
3. 实现 `LLMContrastiveGenerator`
4. 实现 `ContrastivePairValidator`
5. 从 Alpaca 数据中生成 5K 对比对，验证质量

### Phase 2：大规模蒸馏 + RCID（预计 5-7 天）

1. 实现 `ScalableDistillationTrainer`
2. Standard KD 基线：在 Alpaca-52K 上蒸馏 Qwen3-8B → Qwen3-0.6B
3. Standard KD + RCID：加入 RCID 正则
4. λ_RCID 超参数搜索
5. 在 MMLU/GSM8K/ARC/HellaSwag 上评估

### Phase 3：完善实验 + 分析（预计 3-5 天）

1. OOD 鲁棒性实验
2. 因果分析（在大规模蒸馏模型上）
3. 跨架构泛化（LLaMA3）
4. 消融实验：对比对数量 vs 效果、λ_RCID 敏感性

### Phase 4：论文撰写（预计 5-7 天）

---

## 七、论文叙事重构

### 新的 Story

> 1. **发现**（实验 1）：标准蒸馏不保留教师的推理机制——即使学生的任务准确率接近教师，其内部因果结构可能完全不同。
>
> 2. **方法**（RCID）：我们提出残差因果痕迹蒸馏，通过在因果关键位置匹配教师和学生的对比差值方向，引导学生不仅学到教师的输出分布，也学到教师的内部推理机制。
>
> 3. **受控验证**（实验 2）：在受控环境下，RCID 显著提升因果一致性和信息纯度。
>
> 4. **实用验证**（实验 3，核心）：将 RCID 作为即插即用的正则项嫁接到大规模蒸馏上，通过自动构造的对比对，在 MMLU/GSM8K/ARC 等真实 benchmark 上带来可测量的提升。
>
> 5. **解释**（实验 5）：通过因果分析揭示提升的来源——RCID 正则确实改善了学生在关键推理位置上的内部机制对齐。

### 贡献列表

1. **Mechanistic Consistency Metric**：首次提出量化蒸馏中推理机制保留程度的指标
2. **RCID Loss**：基于因果差值的机制对齐正则项，可即插即用到标准蒸馏中
3. **自动对比对构造**：通过实体替换、数字扰动、LLM 生成三种方式自动构造对比对，使方法可扩展
4. **实证**：在 toy data 上验证可解释性，在大规模 benchmark 上验证实用性

---

## 八、风险评估与应对

### 风险 1：RCID 在大规模蒸馏上没有提升

**可能原因**：λ_RCID 没调好、对比对质量不够、大规模 KL 已经隐式传递了足够的机制信息。

**应对**：
- 从 λ_RCID = 0.01 到 1.0 做 grid search
- 如果整体无提升，看分能力维度：RCID 可能在某些能力上有提升（如数学推理），在其他能力上没效果 → 这本身也是一个有趣的发现
- 即使性能提升不大，因果一致性的提升 + 可解释性分析仍然是有价值的贡献

### 风险 2：自动构造的对比对质量不够

**应对**：
- 质量验证 pipeline 严格过滤
- 如果自动方法不够好，回退到半自动方法（基于已有 benchmark 数据集的结构化构造）
- GSM8K/ARC 等数据集本身就有结构化的问答格式，比较容易做最小改动

### 风险 3：计算资源不够

**应对**：
- 先用 Alpaca-52K（小数据），快速验证
- 使用 fp16 + gradient checkpointing
- RCID 正则只需要少量对比对，不显著增加计算量

---

## 九、具体代码修改清单

### 必须修改的文件：

| 文件 | 修改内容 |
|------|---------|
| `src/rcid/distillation/baselines.py` | `StandardKDLoss` 支持全序列 KL + mask |
| `src/rcid/distillation/trainer.py` | `_compute_loss` 改为全序列 KL；新增 `ScalableDistillationTrainer` |
| `src/rcid/circuit/contrastive.py` | `ContrastiveDataset` 支持可变长度和 padding |
| `tests/test_distillation.py` | 更新测试覆盖新的 KL 模式 |

### 必须新增的文件：

| 文件 | 内容 |
|------|------|
| `src/rcid/data/contrastive_generators.py` | 三种自动对比对生成器 |
| `src/rcid/data/contrastive_validator.py` | 对比对质量验证 |
| `src/rcid/data/instruction_dataset.py` | 大规模指令数据加载器 |
| `src/rcid/distillation/scalable_trainer.py` | 大规模蒸馏训练器 |
| `scripts/generate_contrastive_pairs.py` | 对比对生成脚本 |
| `scripts/run_large_scale_distill.py` | 大规模蒸馏实验脚本 |
| `scripts/eval_benchmarks.py` | Benchmark 评估脚本（调用 lm-eval-harness） |
| `configs/large_scale.yaml` | 大规模蒸馏配置 |

---

## 十、立即可执行的下一步

**今天就可以开始做的事情（按优先级排序）**：

1. **修复 `StandardKDLoss`** → 支持全序列 KL + mask（约 30 分钟）

2. **修复 `trainer.py` 的 `_compute_loss`** → 传入全序列 logits 而非单个 answer position 的 logits（约 1 小时）

3. **在 IOI toy data 上用新的全序列 KL 重跑 StandardKD 和 RCID** → 验证修复后 RCID 仍然有效（约 2-3 小时训练）

4. **实现 `EntitySwapContrastiveGenerator`** → 从 Alpaca 数据中自动生成 1K 对比对（约 1 天）

5. **实现 `ScalableDistillationTrainer`** → 在 Alpaca-52K 上跑 Standard KD vs Standard KD + RCID（约 2-3 天）
