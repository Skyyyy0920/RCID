# RCID Project — Claude Code 编码指南

> **项目名称**: RCID (Residual Causal Imprint Distillation)
> **论文标题**: Beyond Component Mapping: Distilling Circuit Knowledge via Residual Stream Causal Imprints
> **目标会议**: NeurIPS 2026

---

## 一、项目结构规范

```
rcid/
├── CLAUDE.md                   # ← 本文件，Claude Code 自动读取
├── pyproject.toml              # 项目元数据与依赖（用 uv/pip 管理）
├── configs/                    # 实验配置（YAML）
│   ├── base.yaml
│   ├── exp1_imprint_extraction.yaml
│   ├── exp2_distillation_comparison.yaml
│   └── exp3_multitask.yaml
├── src/
│   └── rcid/
│       ├── __init__.py
│       ├── circuit/            # 回路分析模块
│       │   ├── patching.py         # activation patching / path patching
│       │   ├── contrastive.py      # 对比数据集构建
│       │   └── checkpoint_selection.py  # 因果检查点选择
│       ├── alignment/          # 跨模型对齐模块
│       │   ├── procrustes.py       # Procrustes 对齐 + 冻结 W
│       │   ├── layer_matching.py   # CKA 层匹配搜索
│       │   └── cka.py              # CKA 计算
│       ├── distillation/       # 蒸馏训练模块
│       │   ├── rcid_loss.py        # L_RCID 损失函数
│       │   ├── trainer.py          # 训练循环
│       │   └── baselines.py        # 标准KD / FitNets / Prakash 基线
│       ├── models/             # 模型定义与加载
│       │   ├── teacher.py
│       │   └── student.py
│       ├── data/               # 数据加载
│       │   ├── ioi.py              # IOI 任务数据
│       │   ├── greater_than.py     # Greater-than 任务数据
│       │   └── general.py          # 通用评估数据 (WikiText等)
│       └── eval/               # 评估模块
│           ├── task_accuracy.py
│           ├── causal_consistency.py   # 因果干预一致性
│           └── perplexity.py
├── scripts/                    # 入口脚本
│   ├── extract_imprints.py     # Step 1: 提取因果痕迹
│   ├── align_spaces.py         # Step 2: Procrustes对齐 + 层匹配
│   ├── train.py                # Step 3: 蒸馏训练
│   └── evaluate.py             # Step 4: 评估
├── tests/                      # 单元测试
│   ├── test_patching.py
│   ├── test_procrustes.py
│   ├── test_cka.py
│   └── test_rcid_loss.py
├── notebooks/                  # 分析与可视化（不放核心逻辑）
│   ├── visualize_imprints.ipynb
│   └── analyze_results.ipynb
└── outputs/                    # 实验输出（gitignore）
    ├── imprints/
    ├── checkpoints/
    └── results/
```

### 关键原则

- `src/rcid/` 下的每个 `.py` 文件不超过 300 行。超过则拆分。
- `scripts/` 是入口，只做参数解析和调用 `src/rcid/` 中的函数，不含实质逻辑。
- `configs/` 控制所有超参数，代码中不硬编码任何数值。
- `tests/` 中的测试**必须**在每次修改核心模块后运行。

---

## 二、编码规范

### 2.1 类型标注（强制）

所有函数签名必须完整标注。这是 NeurIPS 级别代码的基本要求，也帮助 Claude Code 理解意图。

```python
# ✅ 正确
def extract_causal_imprint(
    model: nn.Module,
    clean_input: torch.Tensor,       # shape: (batch, seq_len)
    corrupt_input: torch.Tensor,     # shape: (batch, seq_len)
    layer: int,
    token_pos: int,
) -> torch.Tensor:                   # shape: (batch, d_model)
    """提取指定层和token位置的因果痕迹向量。
    
    因果痕迹定义为: d = r(x_clean) - r(x_corrupt)
    其中 r 是残差流在指定层和位置的值。
    """
    ...

# ❌ 错误
def extract(model, x1, x2, l, t):
    ...
```

### 2.2 Tensor Shape 注释（强制）

每个 tensor 变量旁必须注释 shape。这是调试的生命线。

```python
residual_clean = get_residual(model, clean_input, layer, token_pos)  # (B, d_model)
residual_corrupt = get_residual(model, corrupt_input, layer, token_pos)  # (B, d_model)
imprint = residual_clean - residual_corrupt  # (B, d_model)

# 归一化
imprint_norm = imprint / imprint.norm(dim=-1, keepdim=True)  # (B, d_model)
```

### 2.3 断言检查（关键数学操作处强制）

在所有涉及维度变换、矩阵乘法、损失计算的地方加入 assert：

```python
def rcid_loss(
    teacher_imprint: torch.Tensor,  # (B, d_T)
    student_imprint: torch.Tensor,  # (B, d_S)
    W: torch.Tensor,                # (d_T, d_S)
) -> torch.Tensor:
    assert teacher_imprint.dim() == 2, f"Expected 2D, got {teacher_imprint.dim()}D"
    assert student_imprint.shape[0] == teacher_imprint.shape[0], "Batch size mismatch"
    assert W.shape == (teacher_imprint.shape[1], student_imprint.shape[1]), \
        f"W shape {W.shape} incompatible with d_T={teacher_imprint.shape[1]}, d_S={student_imprint.shape[1]}"
    
    aligned_student = student_imprint @ W.T  # (B, d_T)
    
    # 归一化（避免除以零）
    t_norm = teacher_imprint / (teacher_imprint.norm(dim=-1, keepdim=True) + 1e-8)  # (B, d_T)
    s_norm = aligned_student / (aligned_student.norm(dim=-1, keepdim=True) + 1e-8)  # (B, d_T)
    
    loss = (t_norm - s_norm).pow(2).sum(dim=-1).mean()  # scalar, 等价于 2(1 - cos θ)
    
    assert loss.isfinite(), f"RCID loss is {loss.item()}"
    return loss
```

### 2.4 Hook 管理（易错重灾区）

activation patching 大量使用 PyTorch hooks。必须严格管理生命周期：

```python
# ✅ 正确：用 context manager 管理 hooks
@contextmanager
def residual_hook(
    model: nn.Module,
    layer: int,
    storage: dict[str, torch.Tensor],
):
    """临时注册 hook 提取指定层的残差流输出。"""
    handle = model.transformer.h[layer].register_forward_hook(
        lambda module, input, output: storage.update(
            {"residual": output[0]}  # GPT-2: output[0] 是残差流
        )
    )
    try:
        yield storage
    finally:
        handle.remove()  # 绝对不能泄漏

# 使用
storage = {}
with residual_hook(model, layer=6, storage=storage):
    model(input_ids)
residual = storage["residual"][:, token_pos, :]  # (B, d_model)

# ❌ 错误：hooks 不 remove 会累积，导致内存泄漏和计算错误
handle = model.transformer.h[6].register_forward_hook(...)
model(input_ids)
# 忘了 handle.remove() → 每次调用都会多一个 hook
```

### 2.5 数值稳定性

```python
# 余弦相似度 / 归一化时，始终加 epsilon
eps = 1e-8
norm = x.norm(dim=-1, keepdim=True).clamp(min=eps)
x_normalized = x / norm

# Procrustes 对齐的 SVD 可能不收敛
try:
    U, S, Vh = torch.linalg.svd(M, full_matrices=False)
except torch.linalg.LinAlgError:
    logger.warning("SVD did not converge, falling back to regularized version")
    M_reg = M + 1e-6 * torch.eye(M.shape[0], device=M.device)
    U, S, Vh = torch.linalg.svd(M_reg, full_matrices=False)

W = U @ Vh  # 正交对齐矩阵
```

---

## 三、核心模块实现指南

### 3.1 因果痕迹提取 (`circuit/patching.py`)

```python
def extract_imprints_at_checkpoints(
    model: nn.Module,
    clean_inputs: torch.Tensor,      # (N, seq_len)
    corrupt_inputs: torch.Tensor,    # (N, seq_len)  
    checkpoints: list[tuple[int, int]],  # [(layer, token_pos), ...]
) -> dict[tuple[int, int], torch.Tensor]:  # {(l, t): (N, d_model)}
    """在所有检查点提取因果痕迹。
    
    对每个检查点 (l, t):
        d_{l,t} = residual_l(x_clean)[:, t, :] - residual_l(x_corrupt)[:, t, :]
    
    注意：两次前向传播使用完全相同的模型状态。
    模型始终处于 eval 模式且 no_grad。
    """
    model.eval()
    imprints = {}
    
    with torch.no_grad():
        # 按层分组，减少 hook 注册次数
        layers_needed = sorted(set(l for l, t in checkpoints))
        
        for layer in layers_needed:
            positions = [t for l, t in checkpoints if l == layer]
            
            storage_clean, storage_corrupt = {}, {}
            with residual_hook(model, layer, storage_clean):
                model(clean_inputs)
            with residual_hook(model, layer, storage_corrupt):
                model(corrupt_inputs)
            
            for t in positions:
                d = storage_clean["residual"][:, t, :] - storage_corrupt["residual"][:, t, :]
                imprints[(layer, t)] = d  # (N, d_model)
    
    return imprints
```

**测试要求**：
- 在 GPT-2 上验证：对 IOI 任务，Name Mover heads 所在层的痕迹范数应该最大
- Sanity check：clean == corrupt 时，所有痕迹应为零向量

### 3.2 Procrustes 对齐 (`alignment/procrustes.py`)

```python
def procrustes_align(
    source: torch.Tensor,   # (N, d_S) — 学生侧的对比差值
    target: torch.Tensor,   # (N, d_T) — 教师侧的对比差值
) -> torch.Tensor:          # (d_T, d_S) — 正交对齐矩阵
    """求解 W* = argmin_{W: W^T W = cI} ||W @ source.T - target.T||_F
    
    即最优正交+缩放矩阵，将学生差值空间对齐到教师差值空间。
    解析解：W = U @ Vh，其中 M = target.T @ source = U S Vh (SVD)
    """
    assert source.shape[0] == target.shape[0], "Sample count mismatch"
    
    # 中心化
    source_centered = source - source.mean(dim=0, keepdim=True)  # (N, d_S)
    target_centered = target - target.mean(dim=0, keepdim=True)  # (N, d_T)
    
    M = target_centered.T @ source_centered  # (d_T, d_S)
    U, S, Vh = torch.linalg.svd(M, full_matrices=False)  # U: (d_T, k), Vh: (k, d_S)
    
    W = U @ Vh  # (d_T, d_S)
    
    # 验证正交性
    if W.shape[0] == W.shape[1]:
        orthogonality_error = (W @ W.T - torch.eye(W.shape[0], device=W.device)).norm()
        assert orthogonality_error < 1e-4, f"W not orthogonal: error = {orthogonality_error:.6f}"
    
    return W
```

**测试要求**：
- 构造已知旋转：`target = source @ R.T + noise`，验证恢复的 W ≈ R
- d_T ≠ d_S 时验证 W 的维度正确

### 3.3 CKA 层匹配 (`alignment/layer_matching.py`)

```python
def find_best_layer_mapping(
    teacher_imprints_by_layer: dict[int, torch.Tensor],  # {l: (N, d_T)}
    student_imprints_by_layer: dict[int, torch.Tensor],  # {l': (N, d_S)}
) -> dict[int, int]:  # {teacher_layer: best_student_layer}
    """对每个教师检查点层，找到 CKA 最高的学生层。"""
    mapping = {}
    
    for t_layer, t_data in teacher_imprints_by_layer.items():
        best_score, best_layer = -1.0, -1
        for s_layer, s_data in student_imprints_by_layer.items():
            score = linear_cka(t_data, s_data)
            if score > best_score:
                best_score = score
                best_layer = s_layer
        mapping[t_layer] = best_layer
        logger.info(f"Teacher L{t_layer} → Student L{best_layer} (CKA={best_score:.4f})")
    
    return mapping
```

### 3.4 RCID 损失 (`distillation/rcid_loss.py`)

```python
class RCIDLoss(nn.Module):
    """残差因果印记蒸馏损失。
    
    核心公式:
        L_RCID = (1/|C|) Σ_{(l,t)∈C} || normalize(W* @ d^S_{l,t}) - normalize(d^T_{l,t}) ||²
    
    其中:
        d^T = r^T(x_clean) - r^T(x_corrupt)   (教师因果痕迹，预计算)
        d^S = r^S(x_clean) - r^S(x_corrupt)   (学生因果痕迹，每步计算)
        W*: 冻结的 Procrustes 对齐矩阵
    """
    
    def __init__(
        self,
        W: torch.Tensor,                        # (d_T, d_S), 冻结
        checkpoints: list[tuple[int, int]],      # [(teacher_layer, token_pos), ...]
        layer_mapping: dict[int, int],           # {teacher_layer: student_layer}
        eps: float = 1e-8,
    ):
        super().__init__()
        self.register_buffer("W", W)             # 不参与梯度更新
        self.checkpoints = checkpoints
        self.layer_mapping = layer_mapping
        self.eps = eps
    
    def forward(
        self,
        teacher_imprints: dict[tuple[int, int], torch.Tensor],  # 预计算，{(l,t): (B, d_T)}
        student_model: nn.Module,
        clean_input: torch.Tensor,   # (B, seq_len)
        corrupt_input: torch.Tensor, # (B, seq_len)
    ) -> torch.Tensor:
        # 提取学生的对比差值
        student_layers = sorted(set(self.layer_mapping[l] for l, t in self.checkpoints))
        student_residuals_clean = self._get_residuals(student_model, clean_input, student_layers)
        student_residuals_corrupt = self._get_residuals(student_model, corrupt_input, student_layers)
        
        total_loss = torch.tensor(0.0, device=clean_input.device)
        
        for l_t, t_pos in self.checkpoints:
            l_s = self.layer_mapping[l_t]
            
            # 教师痕迹（预计算）
            d_T = teacher_imprints[(l_t, t_pos)]  # (B, d_T)
            
            # 学生痕迹（在线计算）
            d_S = (student_residuals_clean[l_s][:, t_pos, :]    # (B, d_S)
                   - student_residuals_corrupt[l_s][:, t_pos, :])
            
            # Procrustes 对齐
            d_S_aligned = d_S @ self.W.T  # (B, d_T)
            
            # 归一化后的 MSE（等价于 2(1 - cos θ)）
            d_T_norm = d_T / (d_T.norm(dim=-1, keepdim=True) + self.eps)      # (B, d_T)
            d_S_norm = d_S_aligned / (d_S_aligned.norm(dim=-1, keepdim=True) + self.eps)  # (B, d_T)
            
            checkpoint_loss = (d_T_norm - d_S_norm).pow(2).sum(dim=-1).mean()  # scalar
            total_loss = total_loss + checkpoint_loss
        
        return total_loss / len(self.checkpoints)
    
    def _get_residuals(
        self,
        model: nn.Module,
        input_ids: torch.Tensor,
        layers: list[int],
    ) -> dict[int, torch.Tensor]:  # {layer: (B, seq_len, d_model)}
        """一次前向传播提取多层残差流。"""
        residuals = {}
        handles = []
        
        for layer in layers:
            storage = {}
            handle = model.transformer.h[layer].register_forward_hook(
                lambda mod, inp, out, s=storage: s.update({"r": out[0].detach()})
            )
            handles.append((handle, storage, layer))
        
        with torch.no_grad():
            model(input_ids)
        
        for handle, storage, layer in handles:
            residuals[layer] = storage["r"]
            handle.remove()
        
        return residuals
```

**测试要求**：
- 教师=学生，W=I 时，损失应为零
- 随机 W 时损失应 > 0
- 梯度应只流过学生模型，不流过教师痕迹和 W

---

## 四、实验配置规范

### 4.1 配置文件模板 (`configs/base.yaml`)

```yaml
# 模型
teacher:
  name: "gpt2"                 # HuggingFace model ID
  n_layers: 12
  d_model: 768
  
student:
  name: "custom_4layer"        # 或 "distilgpt2"
  n_layers: 4
  d_model: 384                 # 故意与教师不同，测试跨维度

# 回路分析
circuit:
  task: "ioi"                  # ioi / greater_than / acronym
  n_contrastive_pairs: 500     # 对比数据集大小
  top_k_checkpoints: 5         # 选择因果效应最大的 k 个检查点
  
# 对齐
alignment:
  calibration_size: 200        # Procrustes 校准集大小
  layer_matching: "cka"        # cka / linear_probe / manual
  
# 蒸馏训练
training:
  epochs: 20
  batch_size: 32
  lr: 5e-5
  lambda_rcid: 1.0             # L_RCID 权重
  lambda_kl: 1.0               # L_KL 权重
  optimizer: "adamw"
  scheduler: "cosine"
  seed: 42

# 评估
eval:
  metrics:
    - "task_accuracy"           # IOI 准确率
    - "causal_consistency"      # 因果干预一致性
    - "perplexity"              # WikiText 困惑度（通用能力）
```

### 4.2 实验矩阵

```yaml
# exp2_distillation_comparison.yaml
# 继承 base.yaml，覆盖蒸馏方法

methods:
  - name: "standard_kd"
    lambda_rcid: 0.0
    lambda_kl: 1.0
    
  - name: "fitnets"
    intermediate_matching: "all_layers"
    
  - name: "prakash"
    component_mapping: "ablation_similarity"
    alignment_loss: "cka"
    
  - name: "rcid"
    lambda_rcid: 1.0
    lambda_kl: 1.0

seeds: [42, 123, 456]  # 每个方法跑 3 个种子
```

---

## 五、Claude Code 交互规范

### 5.1 Prompt 模板

在使用 Claude Code 时，遵循以下 prompt 结构：

```
## 任务
[一句话描述要做什么]

## 上下文
- 当前文件: [路径]
- 依赖: [相关模块]
- 数学公式: [如果涉及]

## 约束
- 类型标注完整
- tensor shape 注释
- 关键位置加 assert
- 不超过 300 行

## 验证
完成后运行: pytest tests/test_xxx.py -v
```

### 5.2 分步实现策略

**不要一次性让 Claude Code 写完整个项目。** 按以下顺序逐模块实现和验证：

```
阶段1: 基础设施
  ├── 1a. 项目结构搭建（pyproject.toml, 目录）
  ├── 1b. 模型加载与残差流提取 → test_model_loading.py
  └── 1c. 对比数据集构建（IOI） → test_contrastive_data.py

阶段2: 因果痕迹
  ├── 2a. activation patching → test_patching.py
  ├── 2b. 检查点选择 → test_checkpoint_selection.py
  └── 2c. 痕迹可视化（notebook）→ 目视确认语义合理性

阶段3: 跨模型对齐
  ├── 3a. CKA 实现 → test_cka.py
  ├── 3b. 层匹配搜索 → test_layer_matching.py
  └── 3c. Procrustes 对齐 → test_procrustes.py

阶段4: 蒸馏训练
  ├── 4a. RCID 损失函数 → test_rcid_loss.py
  ├── 4b. 训练循环 → 小规模 sanity check（overfit 10 个样本）
  └── 4c. 基线方法实现 → test_baselines.py

阶段5: 评估与实验
  ├── 5a. 评估指标 → test_eval.py
  ├── 5b. 完整实验运行
  └── 5c. 结果分析与作图
```

### 5.3 代码审查 Checklist

每次 Claude Code 生成代码后，检查以下项目：

```
□ 函数签名有完整类型标注
□ 所有 tensor 变量旁有 shape 注释
□ 维度变换处有 assert
□ hooks 在 context manager 中管理
□ 数值计算有 eps 保护
□ 没有硬编码的超参数（全在 config 中）
□ W 矩阵被注册为 buffer（不参与梯度更新）
□ 教师模型始终在 eval + no_grad 下
□ 损失值有 isfinite 检查
□ 有对应的单元测试
```

---

## 六、常见陷阱与防范

### 6.1 梯度流动错误

```python
# ❌ 危险：教师痕迹参与了学生的计算图
teacher_imprint = get_imprint(teacher, x_c, x_p, l, t)  # 有梯度！
loss = rcid_loss(teacher_imprint, student_imprint)
loss.backward()  # 梯度意外流入教师模型

# ✅ 正确：教师痕迹必须 detach 或在 no_grad 下计算
with torch.no_grad():
    teacher_imprint = get_imprint(teacher, x_c, x_p, l, t)
# 或者
teacher_imprint = get_imprint(teacher, x_c, x_p, l, t).detach()
```

### 6.2 学生前向传播的梯度

```python
# ❌ 错误：学生的对比差值也在 no_grad 下计算了
with torch.no_grad():
    d_S = get_student_residual(student, x_c, l_s, t) - get_student_residual(student, x_p, l_s, t)
# d_S 没有梯度，loss.backward() 无法更新学生

# ✅ 正确：学生的前向传播需要保留梯度
# 但注意：hooks 中不要 detach
handle = model.transformer.h[layer].register_forward_hook(
    lambda mod, inp, out, s=storage: s.update({"r": out[0]})  # 不加 .detach()
)
student(clean_input)     # 这次前向保留梯度
d_S_clean = storage["r"][:, t, :]  # 有梯度

student(corrupt_input)   # 这次也保留梯度
d_S_corrupt = storage["r"][:, t, :]

d_S = d_S_clean - d_S_corrupt  # 有梯度，可以 backward
```

**但这意味着每步训练需要两次前向传播（clean 和 corrupt），是主要的额外开销。**

### 6.3 GPT-2 残差流提取的具体细节

```python
# GPT-2 的 transformer block 输出格式：
# output = (hidden_states, present_key_values, attentions)
# hidden_states 就是残差流（包含了 attention + MLP 的贡献 + 残差连接）

# 但要注意：
# - model.transformer.h[i] 输出的是该 block 处理后的残差流
# - 如果要获取 block 输入端的残差流，需要 hook 到 h[i] 的输入
# - 对于 RCID，我们需要的是 block 输出端（包含了该层的贡献）

# 层归一化的位置也重要：
# GPT-2 用 pre-LN，所以 block 输出 = block 输入 + Attn(LN(input)) + MLP(LN(...))
# 残差流 = h[i] 的输出 = 已经包含了残差连接
```

### 6.4 对比数据集质量

```python
# IOI 对比对的构造必须严格遵循 "最小修改" 原则
# ✅ 正确：只替换关键信息
clean   = "When Mary and John went to the store, John gave a drink to"
corrupt = "When Mary and John went to the store, Mary gave a drink to"
# 仅替换第二个名字：John → Mary

# ❌ 错误：改变了太多内容
corrupt = "When Alice and Bob went to the park, Bob gave a drink to"
# 改了名字 + 地点，差值会混入地点信息
```

---

## 七、可复现性要求

```python
# 在每个 script 入口处
def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# 记录实验环境
def log_environment():
    logger.info(f"Python: {sys.version}")
    logger.info(f"PyTorch: {torch.__version__}")
    logger.info(f"Transformers: {transformers.__version__}")
    logger.info(f"CUDA: {torch.version.cuda}")
    logger.info(f"GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A'}")
    logger.info(f"Config: {OmegaConf.to_yaml(cfg)}")
```

---

## 八、依赖清单

```toml
# pyproject.toml
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
    "scipy>=1.11",          # Procrustes SVD
    "matplotlib>=3.8",
    "seaborn>=0.13",
    "pytest>=7.4",
    "tqdm>=4.66",
]

[project.optional-dependencies]
dev = ["ruff", "mypy", "ipykernel"]
```

---

## 九、关键数学公式速查

供 Claude Code 理解代码意图时参考：

| 符号 | 含义 | 代码变量名 |
|------|------|-----------|
| $\mathbf{d}_{l,t}^T$ | 教师在层 $l$、位置 $t$ 的因果痕迹 | `teacher_imprint` |
| $\mathbf{d}_{\hat{l},t}^S$ | 学生在对应层的因果痕迹 | `student_imprint` |
| $W^*$ | 冻结的 Procrustes 对齐矩阵 | `W` (registered buffer) |
| $\hat{l}$ | 学生中与教师层 $l$ 最匹配的层 | `layer_mapping[l]` |
| $\mathcal{L}_{\text{RCID}}$ | 因果痕迹对齐损失 | `rcid_loss` |
| $\mathcal{L}_{\text{KL}}$ | 输出分布对齐损失 | `kl_loss` |
| $\lambda$ | RCID 损失权重 | `config.training.lambda_rcid` |
| $\mathcal{C}$ | 因果检查点集合 | `checkpoints: list[tuple[int, int]]` |

### 损失函数完整形式

$$\mathcal{L} = \mathcal{L}_{\text{KL}}(y_T, y_S) + \lambda \cdot \frac{1}{|\mathcal{C}|} \sum_{(l,t) \in \mathcal{C}} \left\| \frac{W^* \mathbf{d}_{\hat{l},t}^{S}}{\|W^* \mathbf{d}_{\hat{l},t}^{S}\|} - \frac{\mathbf{d}_{l,t}^{T}}{\|\mathbf{d}_{l,t}^{T}\|} \right\|^2$$
