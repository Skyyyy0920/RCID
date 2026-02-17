# RCID Project — CLAUDE.md

> **项目名称**: RCID (Residual Causal Imprint Distillation)
> **论文定位**: 提出因果一致性指标揭示蒸馏中的机制保留问题，并提出 RCID 方法实现更好的蒸馏
> **核心问题**: 蒸馏后的 student 内部推理机制是否与 teacher 一致？如何让它一致？
> **目标会议**: ICLR 2027 / NeurIPS 2026

---

## 一、论文的两条主线

### 主线 A：发现问题

现有蒸馏方法（StandardKD、FitNets 等）产出的 student 即使任务准确率接近 teacher，
其内部推理机制可能完全不同。我们提出**因果一致性**指标来量化这个现象。

### 主线 B：解决问题

我们提出 RCID，一种因果引导的蒸馏方法。通过在因果关键位置匹配 teacher 和 student
的因果差值（而非原始表示），RCID 在任务准确率、因果一致性、OOD 鲁棒性上均优于现有方法。

两条线缺一不可：没有主线 A，RCID 只是"又一个蒸馏方法"；没有主线 B，论文只有诊断没有解药。

---

## 二、核心概念

### 2.1 两种残差流操作

**Read（提取因果痕迹）**：读取残差流值，不改变模型行为。用于检查点搜索和蒸馏训练。
```python
h_clean = hook_read(model, clean_input, layer, pos)
h_corrupt = hook_read(model, corrupt_input, layer, pos)
d = h_clean - h_corrupt  # 因果痕迹
```

**Write（因果干预 / activation patching）**：替换残差流值并继续前向传播。用于因果一致性评估。
```python
logits_patched = patch_and_run(model, clean_input, corrupt_value, layer, pos)
delta = logit_diff(original) - logit_diff(patched)  # 因果效应
```

Read 回答"这个位置在 clean/corrupt 之间差多少"；
Write 回答"这个位置对模型输出有多大因果影响"。

### 2.2 因果一致性指标

对 teacher 和 student 施加相同的因果干预（Write），比较行为变化：

$$\text{CausalConsistency} = \text{Pearson}(\Delta_T, \Delta_S)$$

- $\Delta_T$：对 teacher 在检查点做 patching 后的输出变化
- $\Delta_S$：对 student 在对应位置做同样 patching 后的输出变化
- student 的 patching 值来自 student 自身的 corrupt 前向传播（不是 teacher 的）

### 2.3 RCID 损失

$$\mathcal{L} = \mathcal{L}_{\text{KL}} + \lambda \cdot \frac{1}{|\mathcal{C}|} \sum_{(l,t) \in \mathcal{C}} \left\| \frac{W^* d_{\hat{l},t}^{S}}{\|W^* d_{\hat{l},t}^{S}\|} - \frac{d_{l,t}^{T}}{\|d_{l,t}^{T}\|} \right\|^2$$

其中 $d = h_{\text{clean}} - h_{\text{corrupt}}$ 是因果差值，$W^*$ 是冻结的 Procrustes 对齐矩阵。

### 2.4 因果检查点选择

搜索 teacher 内部所有 (层, token位置) 组合，按残差流差值范数排序选 top-k。

**关键约束——区分两类位置**：

- **被修改位置**：clean 和 corrupt 在该位置放了不同 token（如 IOI 的 S2 位置）。
  底层差异是 trivial 的（embedding 不同），只有高层差异反映模型的加工。
- **未被修改位置**：clean 和 corrupt 在该位置 token 完全相同（如 IOI 的句末位置）。
  任何差异都来自模型内部的信息传播，是干净的因果信号。

检查点选择需要多样性约束：确保两类位置都有代表，不要被被修改位置的 trivial 高范数淹没。

---

## 三、项目结构

```
rcid/
├── CLAUDE.md
├── pyproject.toml
├── configs/
│   ├── master.yaml
│   ├── exp1_circuit_preservation.yaml
│   ├── exp2_causal_guidance.yaml
│   ├── exp3_robustness_correlation.yaml
│   ├── exp4_component_analysis.yaml
│   └── exp5_information_purity.yaml
├── src/
│   └── rcid/
│       ├── __init__.py
│       ├── circuit/                # 因果分析
│       │   ├── patching.py             # extract_causal_imprints (Read)
│       │   ├── intervention.py         # patch_and_run (Write)
│       │   ├── contrastive.py          # 对比数据集基类
│       │   ├── checkpoint_selection.py # 因果检查点选择（含多样性约束）
│       │   └── component_analysis.py   # IOI 电路组件级分析
│       ├── alignment/
│       │   ├── procrustes.py
│       │   ├── layer_matching.py
│       │   └── cka.py
│       ├── distillation/
│       │   ├── rcid_loss.py
│       │   ├── trainer.py
│       │   └── baselines.py           # StandardKD, FitNets, InformedFitNets
│       ├── models/
│       │   ├── teacher.py
│       │   └── student.py
│       ├── data/
│       │   ├── ioi.py                  # IOI 任务（主力）
│       │   ├── greater_than.py         # Greater-Than（泛化验证）
│       │   └── general.py              # WikiText 等通用数据
│       ├── eval/
│       │   ├── causal_consistency.py   # 核心指标
│       │   ├── task_accuracy.py
│       │   ├── ood_robustness.py
│       │   ├── information_purity.py   # 因果差值 vs 原始表示的信息纯度对比
│       │   └── perplexity.py
│       └── visualization/
│           └── paper_figures.py
├── scripts/
│   ├── run_exp1_circuit_preservation.py
│   ├── run_exp2_causal_guidance.py
│   ├── run_exp3_robustness_correlation.py
│   ├── run_exp4_component_analysis.py
│   ├── run_exp5_information_purity.py
│   ├── run_all_experiments.py
│   └── pretrain_students.py
├── tests/
└── outputs/
```

---

## 四、五个核心实验

### 实验 1：现有蒸馏方法保留了 teacher 的电路吗？

**最重要的发现性实验（主线 A 的核心证据）。**

```
步骤：
1. 在 teacher 上搜索因果检查点（Read）
2. 用 StandardKD、FitNets、第三种方法各蒸馏一个 student
3. 对每个 student，在每个检查点做因果干预（Write）：
   - teacher: ΔT = logit_diff(original) - logit_diff(patched)
   - student: ΔS = 同上（用 student 自己的 corrupt 值）
   - causal_consistency = Pearson(ΔT, ΔS)

预期发现：student 任务准确率接近 teacher，但因果一致性很低。
```

### 实验 2：RCID 能否改善机制传递？

**主线 B 的核心证据——证明 RCID 优于现有方法。**

4 种方法对比，同时评估 task accuracy、causal consistency、perplexity：
- StandardKD（只看输出）
- FitNets（所有层匹配完整表示）
- Informed FitNets（因果位置匹配完整表示）
- RCID（因果位置匹配因果差值）

同时做因素拆解：
- FitNets → Informed FitNets 的提升 = "选对位置"的贡献
- Informed FitNets → RCID 的提升 = "匹配因果差值"的贡献

### 实验 3：电路保留度 vs 行为鲁棒性

收集实验 1、2 中所有 student 的 (causal_consistency, ood_robustness) 数据对。
如果正相关 → 建立因果链：保留 teacher 推理机制 → 更鲁棒的泛化 → RCID 的鲁棒性优势有因果解释。

### 实验 4：逐组件电路分析

以 IOI 为案例。IOI 电路有 7 类组件（Wang et al., 2023）：
Duplicate Token Heads、Previous Token Heads、S-Inhibition Heads、
Name Mover Heads、Backup Name Movers、Induction Heads、Negative Name Movers。

对每种蒸馏方法的 student，用 head-level knockout 检查每类组件是否被保留。
生成 7 类组件 × 4 种方法 的保留度矩阵。

预期：RCID 的 student 保留了更多组件。

### 实验 5：因果差值的信息纯度

对比两种表示的信息内容：
- $h^T$（原始残差流，FitNets 匹配的目标）
- $d^T = h^T_{\text{clean}} - h^T_{\text{corrupt}}$（因果差值，RCID 匹配的目标）

用线性探针分别预测任务标签和控制标签：
- 任务标签：IO name 属于预定义的 A 组还是 B 组
- 控制标签：用了哪个句子模板（与因果逻辑无关但存在于输入中）

在**未被修改的 token 位置**做测试（如句末）。

预期：$d^T$ 的 selectivity > $h^T$ 的 selectivity。
因果差值更"纯净"——只编码任务信息，过滤了通用噪声。

---

## 五、核心模块实现规范

### 5.1 因果干预 (`circuit/intervention.py`)

```python
def patch_and_run(
    model: nn.Module,
    clean_input: torch.Tensor,       # (batch, seq_len)
    patch_value: torch.Tensor,       # (batch, d_model)
    layer: int,
    token_pos: int,
) -> torch.Tensor:                   # (batch, vocab_size) — patched logits
    """在第 layer 层第 token_pos 位置注入 patch_value，继续前向传播。
    
    Pearl 的 do-operator 在 transformer 中的实现。
    """
    def _patch_hook(module, input, output):
        h = output[0].clone()                    # (batch, seq_len, d_model)
        h[:, token_pos, :] = patch_value          # 注入
        return (h,) + output[1:]
    
    handle = model.transformer.h[layer].register_forward_hook(_patch_hook)
    try:
        with torch.no_grad():
            patched_logits = model(clean_input).logits
    finally:
        handle.remove()
    
    return patched_logits


def compute_causal_effect(
    model: nn.Module,
    clean_input: torch.Tensor,       # (batch, seq_len)
    corrupt_input: torch.Tensor,     # (batch, seq_len)
    layer: int,
    token_pos: int,
    answer_pos: int,
    correct_token_id: torch.Tensor,  # (batch,)
    wrong_token_id: torch.Tensor,    # (batch,)
) -> torch.Tensor:                   # (batch,) 因果效应 Δ
    """Δ = logit_diff(original) - logit_diff(patched)
    
    logit_diff = logit(correct) - logit(wrong)
    Δ > 0 表示该位置对正确输出有正向因果贡献。
    """
    model.eval()
    
    # 原始输出
    with torch.no_grad():
        orig_logits = model(clean_input).logits[:, answer_pos, :]  # (batch, vocab)
    orig_diff = (
        orig_logits.gather(1, correct_token_id.unsqueeze(1))
        - orig_logits.gather(1, wrong_token_id.unsqueeze(1))
    ).squeeze(1)  # (batch,)
    
    # 获取 corrupt 时该位置的残差流值
    corrupt_cache = {}
    handle = model.transformer.h[layer].register_forward_hook(
        lambda mod, inp, out, s=corrupt_cache: s.update({"h": out[0][:, token_pos, :].clone()})
    )
    with torch.no_grad():
        model(corrupt_input)
    handle.remove()
    corrupt_value = corrupt_cache["h"]  # (batch, d_model)
    
    # Patched 输出
    patched_logits = patch_and_run(model, clean_input, corrupt_value, layer, token_pos)
    patched_logits = patched_logits[:, answer_pos, :]  # (batch, vocab)
    patched_diff = (
        patched_logits.gather(1, correct_token_id.unsqueeze(1))
        - patched_logits.gather(1, wrong_token_id.unsqueeze(1))
    ).squeeze(1)  # (batch,)
    
    return orig_diff - patched_diff  # (batch,)
```

### 5.2 因果一致性评估 (`eval/causal_consistency.py`)

```python
class CausalConsistencyEvaluator:
    """对 teacher 和 student 施加相同因果干预，比较行为变化的相关性。"""
    
    def evaluate(
        self,
        teacher: nn.Module,
        student: nn.Module,
        dataset: ContrastiveDataset,
        checkpoints: list[tuple[int, int]],
        layer_mapping: dict[int, int],
    ) -> dict:
        """
        Returns:
            per_checkpoint: {(l,t): {"correlation": float, "p_value": float}}
            mean_correlation: float
            teacher_deltas: {(l,t): Tensor(n_samples,)}
            student_deltas: {(l,t): Tensor(n_samples,)}
        """
        results = {}
        for (t_layer, t_pos) in checkpoints:
            s_layer = layer_mapping[t_layer]
            
            delta_T = compute_causal_effect(
                teacher, dataset.clean_ids, dataset.corrupt_ids,
                layer=t_layer, token_pos=t_pos,
                answer_pos=dataset.answer_pos,
                correct_token_id=dataset.correct_ids,
                wrong_token_id=dataset.wrong_ids,
            )
            delta_S = compute_causal_effect(
                student, dataset.clean_ids, dataset.corrupt_ids,
                layer=s_layer, token_pos=t_pos,
                answer_pos=dataset.answer_pos,
                correct_token_id=dataset.correct_ids,
                wrong_token_id=dataset.wrong_ids,
            )
            
            r, p = scipy.stats.pearsonr(delta_T.cpu().numpy(), delta_S.cpu().numpy())
            results[(t_layer, t_pos)] = {"correlation": r, "p_value": p}
        
        mean_corr = np.mean([v["correlation"] for v in results.values()])
        return {"per_checkpoint": results, "mean_correlation": mean_corr}
```

### 5.3 Procrustes 对齐 (`alignment/procrustes.py`)

```python
def procrustes_align(
    source: torch.Tensor,   # (N, d_S)
    target: torch.Tensor,   # (N, d_T)
) -> torch.Tensor:          # (d_T, d_S)
    """W* = argmin ||target - source @ W^T||_F,  s.t. W^T W = cI
    解析解：M = target^T @ source, SVD → W = U @ Vh
    """
    assert source.shape[0] == target.shape[0]
    
    source_c = source - source.mean(dim=0, keepdim=True)  # (N, d_S)
    target_c = target - target.mean(dim=0, keepdim=True)  # (N, d_T)
    
    eps = 1e-8
    source_c = source_c / source_c.norm(dim=-1, keepdim=True).clamp(min=eps)
    target_c = target_c / target_c.norm(dim=-1, keepdim=True).clamp(min=eps)
    
    M = target_c.T @ source_c  # (d_T, d_S)
    try:
        U, S, Vh = torch.linalg.svd(M, full_matrices=False)
    except torch.linalg.LinAlgError:
        M_reg = M + 1e-6 * torch.eye(min(M.shape), device=M.device)
        U, S, Vh = torch.linalg.svd(M_reg, full_matrices=False)
    
    W = U @ Vh  # (d_T, d_S)
    return W
```

### 5.4 RCID 损失 (`distillation/rcid_loss.py`)

```python
class RCIDLoss(nn.Module):
    """残差因果印记蒸馏损失。
    
    在因果检查点位置匹配 teacher 和 student 的因果差值（d = h_clean - h_corrupt）。
    W 是冻结 Procrustes 矩阵（buffer）。教师痕迹预计算（detached）。
    学生的两次前向传播（clean + corrupt）保留梯度。
    """
    
    def __init__(
        self,
        checkpoints: list[tuple[int, int]],
        layer_mapping: dict[int, int],
        W_matrices: dict[int, torch.Tensor],
    ):
        super().__init__()
        self.checkpoints = checkpoints
        self.layer_mapping = layer_mapping
        for t_layer, W in W_matrices.items():
            self.register_buffer(f"W_{t_layer}", W)
    
    def forward(
        self,
        teacher_imprints: dict[tuple[int, int], torch.Tensor],  # 预计算, detached
        student: nn.Module,
        clean_input: torch.Tensor,
        corrupt_input: torch.Tensor,
    ) -> torch.Tensor:
        eps = 1e-8
        total_loss = torch.tensor(0.0, device=clean_input.device)
        
        # 学生前向（保留梯度，hooks 中不 detach）
        student_clean = self._extract_residuals(student, clean_input)
        student_corrupt = self._extract_residuals(student, corrupt_input)
        
        for (t_layer, t_pos) in self.checkpoints:
            s_layer = self.layer_mapping[t_layer]
            W = getattr(self, f"W_{t_layer}")  # (d_T, d_S)
            
            d_T = teacher_imprints[(t_layer, t_pos)]  # (batch, d_T), no grad
            d_S = (
                student_clean[s_layer][:, t_pos, :]
                - student_corrupt[s_layer][:, t_pos, :]
            )  # (batch, d_S), has grad
            
            aligned = d_S @ W.T  # (batch, d_T)
            aligned_n = aligned / aligned.norm(dim=-1, keepdim=True).clamp(min=eps)
            d_T_n = d_T / d_T.norm(dim=-1, keepdim=True).clamp(min=eps)
            
            total_loss = total_loss + (aligned_n - d_T_n).pow(2).sum(dim=-1).mean()
        
        total_loss = total_loss / len(self.checkpoints)
        assert total_loss.isfinite(), f"RCID loss is {total_loss.item()}"
        return total_loss
```

### 5.5 Informed FitNets (`distillation/baselines.py`)

```python
class InformedFitNetsLoss(nn.Module):
    """在因果检查点位置匹配完整表示（非因果差值）。
    
    与 RCID 共享 checkpoints 和 W，唯一区别：
    - RCID:           匹配 d^T = h^T_clean - h^T_corrupt
    - InformedFitNets: 匹配 h^T_clean
    
    关键消融基线：拆解 RCID 的优势来自"选对位置"还是"匹配因果差值"。
    """
    
    def forward(
        self,
        teacher_representations: dict[tuple[int, int], torch.Tensor],  # h^T_clean
        student: nn.Module,
        clean_input: torch.Tensor,
        # 不需要 corrupt_input
    ) -> torch.Tensor:
        ...
```

---

## 六、IOI 数据集规范

### 6.1 对比对构造

```
Clean:   "When Mary and John went to the store, John gave a drink to"  → Mary
Corrupt: "When Mary and John went to the store, Mary gave a drink to"  → ???

区别仅在 S2 位置（后半句的主语）：clean 放 S_name，corrupt 放 IO_name。
```

### 6.2 模板拼接

```python
# 名字带前导空格（GPT-2 BPE 的句中 token 形式）
NAMES = [" Dan", " Bob", " Tom", " James", " Mark", " Luke", " Adam", " Paul", " Jack", " Ben",
         " Mary", " Kate", " Alice", " Emma", " Jane", " Anna", " Sara", " Lisa", " Amy", " Rose"]

TEMPLATES = [
    {"setup": "When{IO} and{S} went to the store,", "action": "{S} gave a drink to"},
    ...
]

# 拼接时不要额外加空格，名字的前导空格充当分隔
clean_text = template["setup"].format(IO=io_name, S=s_name) + template["action"].format(S=s_name)
```

### 6.3 每个样本必须记录

```python
@dataclass
class IOISample:
    clean_ids: torch.Tensor         # token ids
    corrupt_ids: torch.Tensor
    io_token_pos: int               # IO name 位置（未被修改）
    s2_token_pos: int               # S name 第二次出现位置（被修改）
    end_token_pos: int              # 句末位置（未被修改，输出答案处）
    correct_token_id: int           # IO name 的 token id
    wrong_token_id: int             # S name 的 token id
    template_index: int             # 模板编号（用于 control label）
    is_modified: dict[str, bool]    # {"io": False, "s2": True, "end": False}
```

---

## 七、编码规范

### 7.1 类型标注（强制）

所有函数签名完整标注。

### 7.2 Tensor Shape 注释（强制）

```python
residual = storage["h"][:, token_pos, :]  # (batch, d_model)
```

### 7.3 断言检查

```python
assert teacher_imprint.dim() == 2, f"Expected 2D, got {teacher_imprint.dim()}D"
assert loss.isfinite(), f"Loss is {loss.item()}"
```

### 7.4 Hook 生命周期管理

```python
handle = model.transformer.h[layer].register_forward_hook(hook_fn)
try:
    model(input_ids)
finally:
    handle.remove()
```

### 7.5 梯度流管理

- 教师模型：始终 eval + no_grad
- 教师痕迹：预计算后 detach
- W 矩阵：注册为 buffer（不参与梯度更新）
- 学生前向传播：保留梯度（hooks 中不 detach）

### 7.6 数值稳定性

```python
eps = 1e-8
norm = x.norm(dim=-1, keepdim=True).clamp(min=eps)
```

### 7.7 文件长度

每个 `.py` 不超过 300 行。

---

## 八、GPT-2 残差流细节

```python
# GPT-2 block 输出：output = (hidden_states, present_key_values, ...)
# output[0] = 残差流（含 attention + MLP + 残差连接）
# GPT-2 用 pre-LN 架构
# model.transformer.h[0] 是最底层，h[11] 是最顶层（GPT-2 Small）
```

---

## 九、实验配置

### 9.1 模型

```yaml
teacher:
  name: gpt2
  n_layers: 12
  d_model: 768

student:
  n_layers: 4
  d_model: 384
  n_heads: 6
```

### 9.2 实验矩阵

```yaml
methods: [standard_kd, fitnets, informed_fitnets, rcid]
primary_task: ioi
generalization_task: greater_than
seeds: [42, 123, 456]
# 核心: 4 methods × 1 task × 3 seeds = 12 runs
# 泛化: 4 methods × 1 task × 1 seed = 4 runs
```

### 9.3 训练超参数

```yaml
training:
  epochs: 20
  batch_size: 32
  lr: 5e-5
  optimizer: adamw
  scheduler: cosine
  grad_clip: 1.0
  lambda_kl: 1.0
  lambda_rcid: 1.0
```

---

## 十、实施优先级

```
P0（阻塞一切）:
  实现 circuit/intervention.py (patch_and_run, compute_causal_effect)
  实现 eval/causal_consistency.py
  修复 IOI 数据集双空格 bug
  修复 rcid_loss.py batch indexing
  修复 procrustes.py L2 normalization

P1（核心实验）:
  改进检查点选择策略（多样性约束）
  实现 Informed FitNets 基线
  运行实验 1（蒸馏 + 因果一致性评估）
  运行实验 2（4 方法对比，证明 RCID 优于其他方法）

P2（完善论文）:
  实现 OOD Robustness 评估
  实现 Information Purity Test
  运行实验 3（一致性 vs 鲁棒性相关性）
  运行实验 4（组件级分析）
  运行实验 5（信息纯度）

P3（论文材料）:
  论文图表生成
  结果分析脚本
```

---

## 十一、符号速查

| 符号 | 含义 | 代码 |
|------|------|------|
| $h_{l,t}$ | 第 $l$ 层第 $t$ 位置的残差流 | `residual[layer][:, pos, :]` |
| $d_{l,t}^T$ | 教师因果痕迹 $= h_{\text{clean}} - h_{\text{corrupt}}$ | `teacher_imprint` |
| $d_{l,t}^S$ | 学生因果痕迹 | `student_imprint` |
| $W^*$ | 冻结 Procrustes 矩阵 | `W` (buffer) |
| $\hat{l}$ | 学生中与教师层 $l$ 匹配的层 | `layer_mapping[l]` |
| $\Delta_T$ | teacher 做 patching 后的行为变化 | `delta_T` |
| $\Delta_S$ | student 做 patching 后的行为变化 | `delta_S` |
| CC | 因果一致性 $= \text{Pearson}(\Delta_T, \Delta_S)$ | `causal_consistency` |

---

## 十二、依赖

```toml
[project]
name = "rcid"
requires-python = ">=3.10"
dependencies = [
    "torch>=2.1",
    "transformers>=4.36",
    "datasets>=2.16",
    "omegaconf>=2.3",
    "wandb>=0.16",
    "einops>=0.7",
    "scipy>=1.11",
    "scikit-learn>=1.3",
    "matplotlib>=3.8",
    "seaborn>=0.13",
    "pytest>=7.4",
    "tqdm>=4.66",
]
```
