"""RCID — Residual Causal Imprint Distillation.

通过残差流因果痕迹实现知识蒸馏的核心库。
本包提供从因果痕迹提取、跨模型对齐到蒸馏训练的完整流水线。

主要子模块:
    circuit     — 回路分析：activation patching、对比数据集构建、检查点选择
    alignment   — 跨模型对齐：Procrustes 对齐、CKA 层匹配
    distillation — 蒸馏训练：RCID 损失函数、训练循环、基线方法
    models      — 教师与学生模型的定义与加载
    data        — 任务数据集加载（IOI、Greater-than 等）
    eval        — 评估指标：任务准确率、因果一致性、困惑度
"""
