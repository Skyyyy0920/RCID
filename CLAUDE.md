# RCID Project — CLAUDE.md

> **项目名称**: RCID (Residual Causal Imprint Distillation)
> **核心问题**: 蒸馏后的 student 内部推理机制是否与 teacher 一致？如何让它一致？
> **模型配对（主实验）**: Qwen3-8B (teacher) → Qwen3-0.6B (student)
> **模型配对（泛化验证）**: LLaMA-3-8B (teacher) → LLaMA-3.2-1B (student)
> **硬件**: 4 × A100 80GB

---

## 一、论文的两条主线

### 主线 A：发现问题

现有蒸馏方法产出的 student 即使任务准确率接近 teacher，
其内部推理机制可能完全不同。我们提出 **mechanistic consistency** 指标来量化这个现象。

### 主线 B：解决问题

我们提出 RCID，一种对比差值引导的蒸馏方法。通过在因果关键位置匹配 teacher 和 student
的残差流对比差值（而非原始表示），RCID 在任务准确率、因果一致性、OOD 鲁棒性上均优于现有方法。

### 主线 C：跨架构泛化

在 LLaMA 3 上重复核心实验，证明方法不依赖于特定架构，从 GQA 分组策略到 QK-Norm 的有无均不影响有效性。

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

### 2.3 RCID 损失

$$\mathcal{L} = \mathcal{L}_{\text{KL}} + \lambda \cdot \frac{1}{|\mathcal{C}|} \sum_{(l,t) \in \mathcal{C}} \left\| \frac{W^* d_{\hat{l},t}^{S}}{\|W^* d_{\hat{l},t}^{S}\|} - \frac{d_{l,t}^{T}}{\|d_{l,t}^{T}\|} \right\|^2$$

其中 $d = h_{\text{clean}} - h_{\text{corrupt}}$ 是对比差值，$W^*$ 是冻结的 Procrustes 对齐矩阵。

### 2.4 因果检查点选择

搜索 teacher 内部所有 (层, token位置) 组合，按残差流差值范数排序选 top-k。

**多样性约束**：
- **被修改位置**：clean 和 corrupt 在该位置放了不同 token。底层差异是 trivial 的（embedding 不同），只有高层差异反映模型加工。
- **未被修改位置**：clean 和 corrupt 在该位置 token 相同。任何差异都来自模型内部信息传播，是干净的信号。
- 检查点选择确保两类位置都有代表。

---

## 三、模型架构

### 3.1 架构抽象层

所有代码通过 ModelAdapter 抽象层与模型交互，不直接硬编码架构细节。
支持 Qwen3 和 LLaMA 3 两种架构。

```python
class ModelAdapter(ABC):
    """统一接口，屏蔽 Qwen3/LLaMA3 等架构差异。"""

    @abstractmethod
    def get_layers(self, model) -> nn.ModuleList:
        """返回 transformer 层列表。"""
        ...

    @abstractmethod
    def get_embed_tokens(self, model) -> nn.Embedding:
        """返回词嵌入层。"""
        ...

    @abstractmethod
    def get_lm_head(self, model) -> nn.Linear:
        """返回语言模型头。"""
        ...

    @abstractmethod
    def get_residual_hook_point(self, model, layer_idx) -> nn.Module:
        """返回第 layer_idx 层残差流的 hook 挂载点。"""
        ...

    @abstractmethod
    def parse_layer_output(self, output) -> torch.Tensor:
        """从层输出中提取残差流张量。不同模型输出格式不同。"""
        ...

    @abstractmethod
    def get_num_layers(self, model) -> int:
        """返回层数。"""
        ...

    @abstractmethod
    def get_hidden_size(self, model) -> int:
        """返回 d_model。"""
        ...
```

### 3.2 Qwen3 架构细节

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
  position: RoPE (旋转位置编码)
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

### 3.3 LLaMA 3 架构细节

```yaml
LLaMA-3-8B (Teacher):
  hf_name: meta-llama/Llama-3.1-8B
  layers: model.model.layers[0..31]    # 32 层
  d_model: 4096
  n_heads: 32 (GQA: 32 query heads, 8 KV heads)
  vocab: 128256
  embed: model.model.embed_tokens
  lm_head: model.lm_head
  norm: RMSNorm (Pre-Norm), 无 QK-Norm
  position: RoPE
  activation: SwiGLU
  layer_output: output[0]  # (batch, seq, 4096)

LLaMA-3.2-1B (Student):
  hf_name: meta-llama/Llama-3.2-1B
  layers: model.model.layers[0..15]    # 16 层
  d_model: 2048
  n_heads: 32 (GQA: 32 query heads, 8 KV heads)
  vocab: 128256 (共享词表)
  其余同上
```

### 3.4 跨架构关键差异对比

| 特性 | Qwen3 | LLaMA 3 |
|------|-------|---------|
| GQA 分组 | 32Q/4KV (teacher), 16Q/8KV (student) | 32Q/8KV (统一) |
| QK-Norm | ✅ 有 | ❌ 无 |
| 词表大小 | 151,936 | 128,256 |
| Teacher 层数 | 36 | 32 |
| Student 层数 | 28 | 16 |
| Teacher d_model | 4096 | 4096 |
| Student d_model | 1024 | 2048 |
| HF 路径 | model.model.layers | model.model.layers |
| 层输出格式 | output[0] | output[0] |
| Tokenizer | Qwen tokenizer | LLaMA tokenizer |

**关键：** 两个模型族的 HuggingFace 接口几乎完全一致。Adapter 差异仅在配置参数，代码逻辑可复用 90%+。
但 **tokenizer 不同**，因此 IOI 名字池、Factual Probing 模板等都需要针对每个 tokenizer 分别验证。

### 3.5 不共享名字池

Qwen3 和 LLaMA 3 的 tokenizer 不同，单 token 名字列表可能不同。
每个模型族需要独立运行 tokenizer 验证，建立各自的名字池。
数据集类设计为接受名字池参数，不硬编码名字列表。

---

## 四、数据集

### 4.1 数据集体系

```
验证任务：IOI（对比对质量最高，用于确认方法在两种架构上 work）
主力任务 1：Factual Knowledge Probing（知识检索机制保留）
主力任务 2：WinoGrande（常识推理机制保留）
可选任务 3：Simple Math（数值推理机制保留）
```

### 4.2 IOI（Indirect Object Identification）

```
Clean:   "When Mary and John went to the store, John gave a drink to"  → Mary
Corrupt: "When Mary and John went to the store, Mary gave a drink to"  → ???
区别仅在 S2 位置。
```

名字池需在**对应 tokenizer** 中验证为单 token。拼接时不额外加空格。

每个样本记录：clean_ids, corrupt_ids, io_token_pos, s2_token_pos, end_token_pos, correct_token_id, wrong_token_id, template_index, is_modified。

### 4.3 Factual Knowledge Probing

```
Clean:   "The capital of France is"            → Paris
Corrupt: "The capital of Germany is"           → Berlin
区别仅在国家名。
```

数据源：基于 LAMA (Petroni et al., 2019) 模板批量生成，或 CounterFact (Meng et al., 2022)。
需要确保：**对应 teacher** 准确率 > 90%，对比对仅改变一个实体。

每个样本记录：clean_ids, corrupt_ids, entity_pos (被修改), answer_pos, correct_token_id, wrong_token_id, template_index, is_modified。

### 4.4 WinoGrande

```
Clean:   "The trophy doesn't fit in the suitcase because it is too big."
         → it = trophy
Corrupt: "The trophy doesn't fit in the suitcase because it is too small."
         → it = suitcase
区别仅在关键形容词。
```

数据源：WinoGrande 数据集筛选，确保对比差异为单词替换。
需要：**对应 teacher** 准确率 > 80%。

每个样本记录：clean_ids, corrupt_ids, modified_pos, pronoun_pos, correct_referent_id, wrong_referent_id, is_modified。

### 4.5 Simple Math（可选）

```
Clean:   "If John has 5 apples and gives 2, he has" → 3
Corrupt: "If John has 7 apples and gives 2, he has" → 5
区别仅在一个数字。
```

需先验证 student 通过标准 KD 能否达到合理准确率。如果不行则放弃。

### 4.6 通用接口

所有数据集继承同一基类：

```python
class ContrastiveDataset:
    """对比数据集基类。"""
    clean_ids: torch.Tensor           # (N, seq_len)
    corrupt_ids: torch.Tensor         # (N, seq_len)
    answer_pos: torch.Tensor          # (N,)
    correct_token_id: torch.Tensor    # (N,)
    wrong_token_id: torch.Tensor      # (N,)
    key_positions: dict[str, torch.Tensor]  # 每种关键位置的索引
    is_modified: dict[str, bool]      # 哪些关键位置在 clean/corrupt 间不同
    model_family: str                 # "qwen3" 或 "llama3"，标识数据是哪个 tokenizer 生成的
```

---

## 五、四个实验 + 泛化验证

### 实验 1：现有蒸馏方法保留了 teacher 的机制吗？

**主线 A 的核心证据。**

```
1. 在 teacher 上搜索因果检查点（Read）
2. 用 StandardKD、FitNets、InformedFitNets 各蒸馏一个 student（3 seeds）
3. 对每个 student，在每个检查点做因果干预（Write）：
   - Δ_T = logit_diff(original) - logit_diff(patched)   [teacher]
   - Δ_S = 同上                                          [student, 用自己的 corrupt 值]
   - causal_consistency = Pearson(Δ_T, Δ_S)
4. 同时记录 task accuracy

在 IOI + Factual Probing + WinoGrande 上运行。
预期：student 准确率接近 teacher，但因果一致性低。
```

### 实验 2：RCID 能否改善机制传递？

**主线 B 的核心证据——证明 RCID 优于现有方法。**

4 种方法对比：
- StandardKD（只看输出）
- FitNets（所有层匹配完整表示）
- Informed FitNets（因果位置匹配完整表示）
- RCID（因果位置匹配对比差值）

评估指标：task accuracy、causal consistency、perplexity。

因素拆解：
- FitNets → Informed FitNets = "选对位置"的贡献
- Informed FitNets → RCID = "匹配对比差值"的贡献

在 IOI + Factual Probing + WinoGrande 上运行。

### 实验 3：机制一致性 vs 行为鲁棒性

收集实验 1、2 中所有 student 的 (causal_consistency, ood_degradation)。

OOD 变体：
- IOI: 未见名字、不同句式、更长模板
- Factual: 不同关系类型、不同实体领域
- WinoGrande: 更长句子、多个干扰项

如果正相关 → 保留 teacher 机制 → 更鲁棒的泛化。

### 实验 4：对比差值的信息纯度

对比 h^T（原始表示）和 d^T（对比差值）的信息纯度。

在**未被修改的 token 位置**做测试。
- task label：语义二分类
- control label：模板编号或不相关属性
- sklearn LogisticRegression, 80/20 split

预期：d^T 的 selectivity > h^T 的 selectivity。

### 实验 5（泛化验证）：LLaMA 3 跨架构复现

**主线 C 的核心证据——方法不绑定于特定模型架构。**

```
在 LLaMA-3-8B → LLaMA-3.2-1B 上重复实验 1 和实验 2 的核心部分：
- 任务：IOI + Factual Probing（2 个任务即可，WinoGrande 可选）
- 方法：StandardKD, RCID（2 个方法即可，完整消融可选）
- Seeds：2-3 seeds
- 评估：task accuracy + causal consistency

论文中呈现为：
- 一张跨架构对比表（Table X）
- 核心结论：RCID 在两种架构上均优于 StandardKD
```

资源估算：LLaMA 3 实验约 1.5 天（4 卡并行），因为只跑核心对比。

---

## 六、核心模块实现规范

### 6.1 模型适配器 (`models/adapter.py`)

```python
class Qwen3Adapter(ModelAdapter):
    def get_layers(self, model: nn.Module) -> nn.ModuleList:
        return model.model.layers

    def get_residual_hook_point(self, model: nn.Module, layer_idx: int) -> nn.Module:
        return model.model.layers[layer_idx]

    def parse_layer_output(self, output: tuple) -> torch.Tensor:
        return output[0]  # (batch, seq, d_model)

    def get_embed_tokens(self, model: nn.Module) -> nn.Embedding:
        return model.model.embed_tokens

    def get_lm_head(self, model: nn.Module) -> nn.Linear:
        return model.lm_head

    def get_num_layers(self, model: nn.Module) -> int:
        return len(model.model.layers)

    def get_hidden_size(self, model: nn.Module) -> int:
        return model.config.hidden_size


class LLaMA3Adapter(ModelAdapter):
    """LLaMA 3 与 Qwen3 的 HF 接口几乎完全一致。"""

    def get_layers(self, model: nn.Module) -> nn.ModuleList:
        return model.model.layers

    def get_residual_hook_point(self, model: nn.Module, layer_idx: int) -> nn.Module:
        return model.model.layers[layer_idx]

    def parse_layer_output(self, output: tuple) -> torch.Tensor:
        return output[0]  # (batch, seq, d_model)

    def get_embed_tokens(self, model: nn.Module) -> nn.Embedding:
        return model.model.embed_tokens

    def get_lm_head(self, model: nn.Module) -> nn.Linear:
        return model.lm_head

    def get_num_layers(self, model: nn.Module) -> int:
        return len(model.model.layers)

    def get_hidden_size(self, model: nn.Module) -> int:
        return model.config.hidden_size


def get_adapter(model_name: str) -> ModelAdapter:
    """根据模型名自动选择 adapter。"""
    name_lower = model_name.lower()
    if "qwen" in name_lower:
        return Qwen3Adapter()
    elif "llama" in name_lower or "meta" in name_lower:
        return LLaMA3Adapter()
    else:
        raise ValueError(f"Unknown model family: {model_name}")
```

### 6.2 因果干预 (`circuit/intervention.py`)

```python
def patch_and_run(
    model: nn.Module,
    adapter: ModelAdapter,
    clean_input: torch.Tensor,       # (batch, seq_len)
    patch_value: torch.Tensor,       # (batch, d_model)
    layer: int,
    token_pos: int,
) -> torch.Tensor:                   # (batch, vocab_size)
    """Pearl do-operator in transformer residual stream."""
    hook_point = adapter.get_residual_hook_point(model, layer)

    def _patch_hook(module, input, output):
        h = adapter.parse_layer_output(output).clone()
        h[:, token_pos, :] = patch_value
        # 重构 output tuple
        return (h,) + output[1:]

    handle = hook_point.register_forward_hook(_patch_hook)
    try:
        with torch.no_grad():
            patched_logits = model(clean_input).logits
    finally:
        handle.remove()
    return patched_logits


def compute_causal_effect(
    model: nn.Module,
    adapter: ModelAdapter,
    clean_input: torch.Tensor,
    corrupt_input: torch.Tensor,
    layer: int,
    token_pos: int,
    answer_pos: int,
    correct_token_id: torch.Tensor,
    wrong_token_id: torch.Tensor,
) -> torch.Tensor:  # (batch,)
    """Δ = logit_diff(original) - logit_diff(patched)"""
    model.eval()

    with torch.no_grad():
        orig_logits = model(clean_input).logits[:, answer_pos, :]
    orig_diff = (
        orig_logits.gather(1, correct_token_id.unsqueeze(1))
        - orig_logits.gather(1, wrong_token_id.unsqueeze(1))
    ).squeeze(1)  # (batch,)

    # 获取 corrupt 值
    corrupt_cache = {}
    hook_point = adapter.get_residual_hook_point(model, layer)
    handle = hook_point.register_forward_hook(
        lambda mod, inp, out, s=corrupt_cache:
            s.update({"h": adapter.parse_layer_output(out)[:, token_pos, :].clone()})
    )
    with torch.no_grad():
        model(corrupt_input)
    handle.remove()

    patched_logits = patch_and_run(
        model, adapter, clean_input, corrupt_cache["h"], layer, token_pos
    )[:, answer_pos, :]
    patched_diff = (
        patched_logits.gather(1, correct_token_id.unsqueeze(1))
        - patched_logits.gather(1, wrong_token_id.unsqueeze(1))
    ).squeeze(1)

    return orig_diff - patched_diff
```

### 6.3 因果一致性评估 (`eval/causal_consistency.py`)

```python
class CausalConsistencyEvaluator:
    def evaluate(
        self,
        teacher: nn.Module,
        student: nn.Module,
        teacher_adapter: ModelAdapter,
        student_adapter: ModelAdapter,
        dataset: ContrastiveDataset,
        checkpoints: list[tuple[int, int]],
        layer_mapping: dict[int, int],
    ) -> dict:
        results = {}
        for (t_layer, t_pos) in checkpoints:
            s_layer = layer_mapping[t_layer]
            delta_T = compute_causal_effect(
                teacher, teacher_adapter, ...)
            delta_S = compute_causal_effect(
                student, student_adapter, ..., layer=s_layer)
            r, p = scipy.stats.pearsonr(
                delta_T.cpu().numpy(), delta_S.cpu().numpy())
            results[(t_layer, t_pos)] = {"correlation": r, "p_value": p}
        return results
```

### 6.4 Procrustes 对齐 (`alignment/procrustes.py`)

```python
def procrustes_align(
    source: torch.Tensor,   # (N, d_S)  — student dim
    target: torch.Tensor,   # (N, d_T)  — teacher dim
) -> torch.Tensor:          # (d_T, d_S)
    """W* = argmin ||target - source @ W^T||_F"""
    eps = 1e-8
    source_c = source - source.mean(dim=0, keepdim=True)
    target_c = target - target.mean(dim=0, keepdim=True)
    source_c = source_c / source_c.norm(dim=-1, keepdim=True).clamp(min=eps)
    target_c = target_c / target_c.norm(dim=-1, keepdim=True).clamp(min=eps)

    M = target_c.T @ source_c  # (d_T, d_S)
    U, S, Vh = torch.linalg.svd(M, full_matrices=False)
    W = U @ Vh  # (d_T, d_S)
    return W
```

**Procrustes 维度适配**：
- Qwen3: W shape = (4096, 1024)
- LLaMA 3: W shape = (4096, 2048)

### 6.5 RCID 损失 (`distillation/rcid_loss.py`)

```python
class RCIDLoss(nn.Module):
    """在因果检查点匹配 teacher 和 student 的对比差值方向。

    W: buffer, 冻结 Procrustes 矩阵 (d_T, d_S)
    教师痕迹: 预计算, detached
    学生前向: 保留梯度
    """
    def __init__(self, checkpoints, layer_mapping, W_matrices):
        super().__init__()
        self.checkpoints = checkpoints
        self.layer_mapping = layer_mapping
        for t_layer, W in W_matrices.items():
            self.register_buffer(f"W_{t_layer}", W)

    def forward(
        self,
        teacher_imprints: dict[tuple[int, int], torch.Tensor],
        student_clean_residuals: dict[int, torch.Tensor],
        student_corrupt_residuals: dict[int, torch.Tensor],
    ) -> torch.Tensor:
        eps = 1e-8
        total = torch.tensor(0.0, device=next(iter(teacher_imprints.values())).device)

        for (t_layer, t_pos) in self.checkpoints:
            s_layer = self.layer_mapping[t_layer]
            W = getattr(self, f"W_{t_layer}")

            d_T = teacher_imprints[(t_layer, t_pos)]      # (batch, d_T), no grad
            d_S = (student_clean_residuals[s_layer][:, t_pos, :]
                   - student_corrupt_residuals[s_layer][:, t_pos, :])  # (batch, d_S), has grad

            aligned = d_S @ W.T                             # (batch, d_T)
            aligned_n = aligned / aligned.norm(dim=-1, keepdim=True).clamp(min=eps)
            d_T_n = d_T / d_T.norm(dim=-1, keepdim=True).clamp(min=eps)

            total = total + (aligned_n - d_T_n).pow(2).sum(dim=-1).mean()

        total = total / len(self.checkpoints)
        assert total.isfinite(), f"RCID loss is {total.item()}"
        return total
```

### 6.6 Informed FitNets (`distillation/baselines.py`)

```python
class InformedFitNetsLoss(nn.Module):
    """与 RCID 共享 checkpoints 和 W，匹配 h^T_clean 而非 d^T。

    消融基线：拆解 RCID 的优势来自"选对位置"还是"匹配对比差值"。
    """
    ...
```

---

## 七、项目结构

```
rcid/
├── CLAUDE.md
├── pyproject.toml
├── configs/
│   ├── master.yaml
│   ├── qwen3.yaml                      # Qwen3 模型特定配置
│   ├── llama3.yaml                     # LLaMA 3 模型特定配置
│   ├── exp1_mechanism_preservation.yaml
│   ├── exp2_rcid_comparison.yaml
│   ├── exp3_robustness_correlation.yaml
│   ├── exp4_information_purity.yaml
│   └── exp5_cross_architecture.yaml    # 新增：跨架构泛化
├── src/
│   └── rcid/
│       ├── __init__.py
│       ├── models/
│       │   ├── __init__.py
│       │   ├── adapter.py              # ModelAdapter 基类 + Qwen3Adapter + LLaMA3Adapter
│       │   ├── teacher.py              # 加载 teacher（自动识别模型族）
│       │   └── student.py              # 加载 student（自动识别模型族）
│       ├── circuit/
│       │   ├── __init__.py
│       │   ├── patching.py             # Read: extract_contrastive_differences
│       │   ├── intervention.py         # Write: patch_and_run, compute_causal_effect
│       │   ├── checkpoint_selection.py # 多样性约束检查点选择
│       │   └── contrastive.py          # ContrastiveDataset 基类
│       ├── alignment/
│       │   ├── __init__.py
│       │   ├── procrustes.py
│       │   ├── layer_matching.py
│       │   └── cka.py
│       ├── distillation/
│       │   ├── __init__.py
│       │   ├── rcid_loss.py
│       │   ├── trainer.py
│       │   └── baselines.py            # StandardKD, FitNets, InformedFitNets
│       ├── data/
│       │   ├── __init__.py
│       │   ├── ioi.py
│       │   ├── factual_probing.py
│       │   ├── winogrande.py
│       │   └── simple_math.py          # 可选
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
│   ├── pilot_validation.py             # 快速验证（Qwen3 + LLaMA 3 各自独立验证）
│   ├── run_exp1.py
│   ├── run_exp2.py
│   ├── run_exp3.py
│   ├── run_exp4.py
│   ├── run_exp5_cross_arch.py          # 新增：LLaMA 3 泛化实验
│   └── run_all.py
├── tests/
│   ├── test_models.py
│   ├── test_data.py
│   ├── test_circuit.py
│   ├── test_alignment.py
│   ├── test_distillation.py
│   ├── test_eval.py
│   └── test_integration.py
└── outputs/
    └── results/
        ├── exp1/
        ├── exp2/
        ├── exp3/
        ├── exp4/
        └── exp5_cross_arch/            # 新增
```

---

## 八、实验配置

### 8.1 模型

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

### 8.2 实验矩阵

```yaml
# 主实验 (Qwen3)
methods: [standard_kd, fitnets, informed_fitnets, rcid]
tasks:
  validation: ioi
  primary: [factual_probing, winogrande]
seeds: [42, 123, 456]

# 核心: 4 methods × 3 tasks × 3 seeds = 36 runs
# 每 run 约 4-6 小时 (A100)
# 4 卡并行 → 约 9 轮 → 2-3 天

# 泛化验证 (LLaMA 3)
llama3_methods: [standard_kd, rcid]                # 只需对比核心方法
llama3_tasks: [ioi, factual_probing]               # 2 个任务
llama3_seeds: [42, 123, 456]

# 泛化: 2 methods × 2 tasks × 3 seeds = 12 runs
# 4 卡并行 → 约 3 轮 → 1-1.5 天
```

### 8.3 训练超参数

```yaml
training:
  epochs: 20
  batch_size: 16        # A100 80GB 足够
  lr: 5e-5
  optimizer: adamw
  scheduler: cosine
  grad_clip: 1.0
  lambda_kl: 1.0
  lambda_rcid: 1.0
  fp16: true            # 减少显存，加速训练
```

### 8.4 多 GPU 策略

```yaml
# 不做分布式训练（模型不需要），而是做实验级并行
# Qwen3 主实验
gpu_0: seed=42, method=standard_kd → method=fitnets → ...
gpu_1: seed=42, method=informed_fitnets → method=rcid → ...
gpu_2: seed=123, ...
gpu_3: seed=456, ...

# LLaMA 3 泛化实验（主实验完成后运行）
gpu_0: seed=42, method=standard_kd
gpu_1: seed=42, method=rcid
gpu_2: seed=123, method=standard_kd + rcid
gpu_3: seed=456, method=standard_kd + rcid
```

---

## 九、Pilot Validation（最先运行！）

在写完整代码之前，必须**分别**对两种模型族跑验证：

```python
# scripts/pilot_validation.py
# 参数: --model_family qwen3 | llama3
# 目标：每种模型 30 分钟内完成

# 1. 加载 teacher，在 IOI 上测试准确率
#    预期：> 95%。如果不行 → IOI 模板需要适配

# 2. 用对应 tokenizer 检查名字池，输出单 token 名字列表
#    预期：至少 20 个可用名字

# 3. 构造 10 个 IOI 对比对，提取各层因果差值范数
#    预期：存在明显的层间差异，高层范数 > 底层
#    如果全部接近 0 → 方法不适用于该架构

# 4. 在一个检查点做 activation patching，计算因果效应
#    预期：某些位置 Δ > 0 且显著
#    如果所有位置 Δ ≈ 0 → patching 在该架构上无效

# 5. 加载 student（未蒸馏），在 IOI 上测试 baseline 准确率
#    评估 student 容量
```

**如果任何一种架构在第 2-4 步失败，该架构不能用于实验。**
**Qwen3 必须全部通过（主实验）。LLaMA 3 如果失败，退回单架构方案。**

---

## 十、编码规范

### 10.1 类型标注（强制）
所有函数签名完整标注。

### 10.2 Tensor Shape 注释（强制）
```python
residual = adapter.parse_layer_output(output)[:, token_pos, :]  # (batch, d_model)
```

### 10.3 断言检查
```python
assert teacher_imprint.dim() == 2, f"Expected 2D, got {teacher_imprint.dim()}D"
assert loss.isfinite(), f"Loss is {loss.item()}"
```

### 10.4 Hook 生命周期管理
```python
handle = hook_point.register_forward_hook(hook_fn)
try:
    model(input_ids)
finally:
    handle.remove()
```

### 10.5 梯度流管理
- 教师模型：始终 eval + no_grad
- 教师痕迹：预计算后 detach
- W 矩阵：注册为 buffer
- 学生前向传播：保留梯度

### 10.6 数值稳定性
```python
eps = 1e-8
norm = x.norm(dim=-1, keepdim=True).clamp(min=eps)
```

### 10.7 不硬编码架构
所有模型交互必须通过 ModelAdapter，禁止直接写 `model.transformer.h[l]` 或 `model.model.layers[l]`（除 adapter 内部实现外）。

### 10.8 文件长度
每个 `.py` 不超过 300 行。

### 10.9 模型族参数化
任何涉及模型名、层数、维度的地方都应从 config 读取或通过 adapter 查询，不硬编码。
数据集类接受 tokenizer 参数，不假设特定词表。

---

## 十一、相关工作定位

**vs Wu et al. (2022) Causal Distillation (DIITO)**：
DIITO 用 interchange intervention（替换为另一个输入的值）+ 预定义因果模型。
RCID 用 activation patching（替换为 corrupt 版本的值）+ 自动检查点搜索。
RCID 不需要预先知道 teacher 的因果结构。

**vs Prakash et al. (2025) Circuit Distillation**：
他们用 CKA 匹配电路组件的完整激活。
我们用 Procrustes 匹配对比差值（过滤了任务无关信息）。

**vs Dunefsky et al. (2025) Distilled Circuits**：
纯分析工作。我们既诊断（实验 1）又治疗（RCID）。

---

## 十二、实施优先级

```
P0（阻塞一切）:
  Pilot validation（Qwen3 + LLaMA 3 分别运行）
  ModelAdapter 抽象层（含 Qwen3Adapter + LLaMA3Adapter）
  IOI 数据集 + 双 tokenizer 验证

P1（核心实验 — Qwen3）:
  intervention.py + causal_consistency.py
  检查点选择（多样性约束）
  所有蒸馏损失 + trainer
  运行实验 1 + 2（IOI 先行，然后扩展到真实任务）

P2（完善论文 — Qwen3）:
  Factual Probing + WinoGrande 数据集
  OOD Robustness 评估
  Information Purity Test
  运行实验 3 + 4

P3（跨架构泛化 — LLaMA 3）:
  LLaMA 3 数据集适配（名字池、模板验证）
  运行实验 5（exp1 + exp2 核心部分的 LLaMA 3 复现）
  跨架构对比表

P4（论文材料）:
  论文图表
  结果分析
  跨架构讨论章节
```

---

## 十三、符号速查

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

---

## 十四、依赖

```toml
[project]
name = "rcid"
requires-python = ">=3.10"
dependencies = [
    "torch>=2.1",
    "transformers>=4.45",    # Qwen3 + LLaMA 3 支持
    "accelerate>=0.27",      # 模型加载
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
```

---

## 十五、论文结构映射

```
Section 1: Introduction
  → 主线 A + B + C 叙事

Section 2: Mechanistic Consistency Metric
  → 2.1-2.2（Read/Write + 因果一致性）

Section 3: Method (RCID)
  → 2.3-2.4 + 6.1-6.5（损失、检查点选择、对齐）

Section 4: Experiments
  4.1 Setup: 两个模型族、三个任务、四个方法
  4.2 Exp1: 现有方法机制不一致（Qwen3 主表）
  4.3 Exp2: RCID 改善 + 因素拆解（Qwen3 主表）
  4.4 Exp3: 一致性 vs 鲁棒性散点图
  4.5 Exp4: 信息纯度
  4.6 Exp5: 跨架构泛化（LLaMA 3 验证表）

Section 5: Discussion
  → 跨架构讨论、MoE 未来方向、局限性
```
