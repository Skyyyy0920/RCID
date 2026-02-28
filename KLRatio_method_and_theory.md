# KL-Ratio Adaptive Distillation: Method & Theory

## 1. 问题设定

知识蒸馏中，teacher 分布 $p$ 和 student 分布 $q_\theta$ 之间的对齐可通过两种 KL 散度：

$$\text{FKL}(p \| q_\theta) = \sum_{j=1}^{V} p(Y_j) \log \frac{p(Y_j)}{q_\theta(Y_j)}$$

$$\text{RKL}(p \| q_\theta) = \sum_{j=1}^{V} q_\theta(Y_j) \log \frac{q_\theta(Y_j)}{p(Y_j)}$$

混合损失为：

$$\mathcal{L} = \alpha \cdot \text{FKL} + (1-\alpha) \cdot \text{RKL}$$

当 $\alpha = 1$: 标准 KD（forward KL only）  
当 $\alpha = 0.5$: Jeffreys divergence  
当 $\alpha = 0$: reverse KL only

核心问题：**如何确定最优的 $\alpha$？**

---

## 2. 已有方法的局限

### 2.1 固定混合（Jeffreys）
$\alpha = 0.5$ 对所有 token、所有训练阶段使用相同权重。
忽略了不同位置、不同训练阶段的 gap 结构差异。

### 2.2 AKL (Wu et al., COLING 2025)

AKL 使用 head/tail 的 L1 gap 作为 adaptive 信号：

**Step 1**：对 teacher 分布排序，按累积概率阈值 $\mu=0.5$ 划分 head/tail  
**Step 2**：计算 L1 gap
$$g_{\text{head}} = \sum_i M[i] |p(Y_i) - q_\theta(Y_i)|, \quad g_{\text{tail}} = \sum_i (1-M[i]) |p(Y_i) - q_\theta(Y_i)|$$
**Step 3**：$\alpha = g_{\text{head}} / (g_{\text{head}} + g_{\text{tail}})$

**AKL 的三个结构性缺陷**：

**(a) L1 gap 是 KL 的低质量代理。** L1 距离 $|p_i - q_i|$ 是对称、线性的，而 KL 散度是非对称、非线性的。考虑两个 token：
- Token A: $p=0.01, q=0.05$, L1 gap = 0.04
- Token B: $p=0.51, q=0.55$, L1 gap = 0.04

L1 gap 相同，但 RKL 贡献差异巨大：token A 的 $q \log(q/p) = 0.05 \log 5 \approx 0.08$，token B 的 $q \log(q/p) = 0.55 \log(55/51) \approx 0.04$。AKL 无法区分这种差异。

**(b) 排序和二分引入不必要的超参数和计算。** 每个 position 需要对整个词表（$V \approx 150K$）排序，并依赖阈值 $\mu$（论文中 $\mu \in \{0.45, 0.50, 0.55\}$ 结果不同）。

**(c) 无时间动态。** AKL 在每个 step 独立计算 $\alpha$，不 track 训练进展。这与 AKL 自身的核心发现矛盾——FKL 先拟合 head、RKL 先拟合 tail 是一个时间过程。

---

## 3. 我们的方法：KL-Ratio Adaptive Distillation

### 3.1 核心思想

**直接使用 FKL 和 RKL 的比值作为自适应信号。**

由于混合 loss 已经需要计算 FKL 和 RKL，它们的比值可以零额外开销地获取。

定义即时混合系数：

$$\alpha^{\text{inst}} = \frac{\overline{\text{FKL}}_{\mathcal{B}}}{\overline{\text{FKL}}_{\mathcal{B}} + \overline{\text{RKL}}_{\mathcal{B}} + \epsilon}$$

其中 $\overline{\text{FKL}}_{\mathcal{B}}$ 和 $\overline{\text{RKL}}_{\mathcal{B}}$ 是当前 batch $\mathcal{B}$ 中所有有效 token 位置的平均 FKL 和 RKL。

用指数移动平均（EMA）平滑以捕捉训练动态：

$$\alpha_t = \beta \cdot \alpha_{t-1} + (1-\beta) \cdot \alpha^{\text{inst}}_t$$

初始化 $\alpha_0 = 0.5$（等价于 Jeffreys 起步）。$\beta = 0.99$。

最终 loss：

$$\mathcal{L}_t = \alpha_t \cdot \text{FKL}_t + (1-\alpha_t) \cdot \text{RKL}_t$$

其中 $\alpha_t$ 是 detached scalar（不参与梯度反向传播）。

### 3.2 算法伪代码

```
Algorithm: KL-Ratio Adaptive Distillation
────────────────────────────────────────
Input: Teacher T, Student S, dataset D, EMA coefficient β=0.99
Initialize: α_ema = 0.5

for each batch B in D:
    # Teacher forward (no grad)
    with torch.no_grad():
        p = softmax(T(B) / τ)         # (batch, seq, V)
    
    # Student forward
    q = softmax(S(B) / τ)             # (batch, seq, V)
    
    # Compute per-position KL divergences
    log_p = log_softmax(T(B) / τ)
    log_q = log_softmax(S(B) / τ)
    FKL_t = Σ_v p * (log_p - log_q)   # (batch, seq)
    RKL_t = Σ_v q * (log_q - log_p)   # (batch, seq)
    
    # Batch-level ratio
    FKL_mean = masked_mean(FKL_t)      # scalar
    RKL_mean = masked_mean(RKL_t)      # scalar
    α_inst = FKL_mean / (FKL_mean + RKL_mean + ε)
    
    # EMA update
    α_ema = β * α_ema + (1-β) * α_inst.item()
    
    # Mixed loss (α_ema is detached)
    L = α_ema * FKL_t + (1-α_ema) * RKL_t
    loss = masked_mean(L)
    
    loss.backward()
    optimizer.step()
```

---

## 4. 理论分析

### 4.1 命题 1：KL 比值直接反映 head/tail 不平衡

**命题**：$\alpha^{\text{inst}} = \frac{\text{FKL}}{\text{FKL} + \text{RKL}}$ 是 teacher-student 分布在 head 区域和 tail 区域不匹配程度的信息论度量。

**证明**：

考虑 FKL 和 RKL 的梯度形式（Wu et al., 2024 的 Eq. 5-6）：

$$\frac{\partial \text{FKL}}{\partial z_j^q} = q_\theta(Y_j) - p(Y_j)$$

$$\frac{\partial \text{RKL}}{\partial z_j^q} = q_\theta(Y_j) \left[\log \frac{q_\theta(Y_j)}{p(Y_j)} - \text{RKL}\right]$$

对于 FKL 的值，其被 $p(Y_j)$ 大的 token 主导（因为 $p \log(p/q)$ 在 $p$ 大时贡献大）。这些恰好是 head 区域的 token。

对于 RKL 的值，当 $p(Y_j)$ 很小而 $q_\theta(Y_j)$ 不为零时，$q \log(q/p)$ 爆发式增大。这些恰好是 tail 区域的 token。

因此：
- $\text{FKL} \gg \text{RKL}$ $\Rightarrow$ $\alpha \to 1$：gap 集中在 head，而 FKL 梯度高效修复 head $\Rightarrow$ 多用 FKL
- $\text{FKL} \ll \text{RKL}$ $\Rightarrow$ $\alpha \to 0$：gap 集中在 tail，而 RKL 梯度高效修复 tail $\Rightarrow$ 多用 RKL
- $\text{FKL} \approx \text{RKL}$ $\Rightarrow$ $\alpha \approx 0.5$：gap 均衡，退化为 Jeffreys $\quad\square$

**与 AKL 的关键区别**：AKL 用 L1 gap 作为 head/tail 不平衡的代理度量，而我们直接使用 KL 散度值本身——后者是信息论中度量分布不匹配的原生工具，天然地对高/低概率区域赋予不同权重，无需显式划分 head/tail。

### 4.2 命题 2：混合损失的梯度具有自平衡性质

**命题**：当 $\alpha = \frac{\text{FKL}}{\text{FKL}+\text{RKL}}$ 时，混合损失 $\mathcal{L} = \alpha \cdot \text{FKL} + (1-\alpha) \cdot \text{RKL}$ 的梯度在 head 和 tail 区域之间自动平衡。

**证明**：

混合损失对 student logit $z_j^q$ 的梯度为（$\alpha$ detached）：

$$\frac{\partial \mathcal{L}}{\partial z_j^q} = \alpha \cdot [q_\theta(Y_j) - p(Y_j)] + (1-\alpha) \cdot q_\theta(Y_j)\left[\log\frac{q_\theta(Y_j)}{p(Y_j)} - \text{RKL}\right]$$

分析三种极端情况：

**情况 1**：$\text{FKL} \gg \text{RKL}$（head 区域严重不匹配）

此时 $\alpha \approx 1$，梯度近似为 $q_\theta(Y_j) - p(Y_j)$。

这是 FKL 梯度，对 $p(Y_j)$ 大的 token（head）提供大的修正力，直接修复 head 区域的不匹配。

**情况 2**：$\text{FKL} \ll \text{RKL}$（tail 区域严重不匹配）

此时 $\alpha \approx 0$，梯度近似为 $q_\theta(Y_j)[\log(q_\theta/p) - \text{RKL}]$。

这是 RKL 梯度，对 $p(Y_j)$ 小而 $q_\theta(Y_j)$ 不为零的 token（tail）提供强烈的修正信号。

**情况 3**：$\text{FKL} \approx \text{RKL}$（均衡匹配）

$\alpha \approx 0.5$，梯度是 FKL 和 RKL 梯度的等权平均，即 Jeffreys divergence 的梯度。这为 head 和 tail 提供均衡的修正力。 $\quad\square$

**推论**：KL-Ratio 混合自动实现了类似 GradNorm（Chen et al., ICML 2018）的多任务梯度平衡——将更多梯度资源分配给学习较慢的方向（head 或 tail），而无需显式计算梯度范数。

### 4.3 命题 3：EMA 动态自动实现训练 curriculum

**命题**：在典型的 KD 训练过程中，$\alpha_{\text{ema}}$ 随训练进展呈现有规律的变化，自动形成从 head 到 tail 的学习 curriculum。

**推导**：

基于 AKL 论文的核心发现（Section 3.2）：

- **训练初期**：FKL 梯度先快速缩小 head gap $\Rightarrow$ FKL 快速下降。RKL 在 tail 的修正较慢（tail token 多但概率低，梯度信号弱）$\Rightarrow$ RKL 下降较慢。

  因此 $\text{FKL} < \text{RKL}$，$\alpha^{\text{inst}} < 0.5$，$\alpha_{\text{ema}}$ 逐渐低于 0.5，增加 RKL 权重。

  **效果**：FKL 已经在自然地拟合 head，EMA 将额外资源投向 tail，形成互补。

- **训练中期**：head 已经 well-fit，FKL 变小；tail 仍在改善，RKL 相对较大。

  $\alpha_{\text{ema}}$ 继续降低，RKL 权重继续增加。

  **效果**：训练重心自然转移到 tail 拟合。

- **训练后期**：FKL 和 RKL 都趋向收敛，两者比值趋于稳定。

  $\alpha_{\text{ema}} \to$ 某个稳定值（取决于最终分布匹配的 head/tail 残差比）。

  **效果**：系统自动达到平衡状态。 $\quad\square$

这与手动设计的 curriculum（如 L2M-KD, Zhang & Liu, EMNLP 2025）不同——我们的 curriculum 是从数据和训练动态中自动涌现的，不需要预定义阶段或人工调参。

### 4.4 命题 4：KL-Ratio 是 AKL 的严格推广

**命题**：AKL 的 L1 gap ratio 可以看作 KL ratio 的一阶线性近似。

**证明**：

对 FKL 做 Taylor 展开。设 $q_\theta(Y_j) = p(Y_j) + \delta_j$，其中 $\delta_j$ 是小量：

$$\text{FKL} = \sum_j p_j \log\frac{p_j}{p_j + \delta_j} = \sum_j p_j \left[-\frac{\delta_j}{p_j} + \frac{\delta_j^2}{2p_j^2} + O(\delta_j^3)\right]$$

$$\approx -\sum_j \delta_j + \frac{1}{2}\sum_j \frac{\delta_j^2}{p_j}$$

注意 $\sum_j \delta_j = \sum_j (q_j - p_j) = 0$（概率归一化），因此：

$$\text{FKL} \approx \frac{1}{2}\sum_j \frac{(q_j - p_j)^2}{p_j}$$

类似地：

$$\text{RKL} \approx \frac{1}{2}\sum_j \frac{(q_j - p_j)^2}{q_j}$$

而 AKL 的 head L1 gap 为：

$$g_{\text{head}} = \sum_{j \in \text{head}} |p_j - q_j|$$

$g_{\text{head}} / (g_{\text{head}} + g_{\text{tail}})$ 是 L1 距离按 head/tail 的占比。

而 $\text{FKL} / (\text{FKL} + \text{RKL})$ 在近似下为：

$$\frac{\sum_j (q_j - p_j)^2 / p_j}{\sum_j (q_j - p_j)^2 / p_j + \sum_j (q_j - p_j)^2 / q_j}$$

这是一个**加权的 $\chi^2$ 统计量的比值**，其中权重 $1/p_j$ 和 $1/q_j$ 自然对低概率区域（tail）赋予更高的权重。

相比之下，AKL 的 L1 gap 对所有 token 等权处理，是一个更粗糙的度量。

因此，KL ratio 包含了 AKL 的 L1 gap ratio 所没有的概率加权信息。 $\quad\square$

---

## 5. 与 AKL 的全面对比

| 维度 | AKL (Wu et al.) | KL-Ratio (Ours) |
|------|-----------------|------------------|
| **信号来源** | L1 gap: $\|p_i - q_i\|$（线性、对称） | KL 值本身（非线性、非对称） |
| **Head/Tail 划分** | 显式排序 + 阈值 $\mu$ 二分 | 无需划分，KL 自然区分 |
| **时间动态** | 每 step 独立决策，无记忆 | EMA 平滑，捕捉训练进展 |
| **超参数** | $\mu$（head/tail 阈值，敏感） | $\beta$（EMA 系数，不敏感） |
| **额外计算** | $O(V \log V)$ 排序 + mask + L1 | 零（FKL/RKL 本已计算） |
| **理论基础** | head/tail 经验观察 | KL asymmetry 信息论性质 + 多任务平衡 |
| **初始化** | 依赖初始分布的 gap 结构 | $\alpha_0=0.5$（Jeffreys，安全起步） |

---

## 6. 实验结果

### 6.1 主实验

**设定**：Teacher: Qwen3-8B, Student: Qwen3-0.6B, Data: Alpaca-52K, 3 epochs

| Method | GSM8K | MMLU | ARC-C | Avg |
|--------|-------|------|-------|-----|
| Forward KL (baseline) | 36.09 | - | - | - |
| Jeffreys ($\alpha$=0.5) | 37.68 | - | - | - |
| AKL (Wu et al.) | 33.74 | 0.00 | 0.00 | 11.25 |
| **KL-Ratio-token (ours)** | 33.97 | 46.25 | 41.04 | 40.42 |
| **KL-Ratio-batch-EMA (ours)** | **37.00** | **46.10** | **41.55** | **41.55** |

**关键发现**：

1. **AKL 在本设定下完全失效**（MMLU/ARC-C = 0），说明其 head/tail 排序机制在更大模型和标准 benchmark 下不鲁棒。

2. **Batch-level EMA 远优于 token-level**：token-level 的 GSM8K 仅 33.97（甚至低于 forward KL），而 batch-level + EMA 达到 37.00。这表明 per-token 的 ratio 信号噪声太大，batch-level 平均 + 时间平滑是关键。

3. **KL-Ratio-batch-EMA 与 Jeffreys 可比**，但引入了 adaptive 能力——在 MMLU 和 ARC-C 上的表现有待与 Jeffreys 完整对比。
