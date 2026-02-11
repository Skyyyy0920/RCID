"""回路分析模块。

负责对 Transformer 模型进行机械可解释性分析，定位因果关键检查点，
并提取残差流因果痕迹 (causal imprints)。

子模块:
    patching            — activation patching / path patching，提取因果痕迹向量
    contrastive         — 对比数据集构建（minimal-pair 原则）
    checkpoint_selection — 基于因果效应排序，选择 top-k 检查点
"""
