"""评估模块。

提供多维度的模型评估指标，衡量蒸馏后学生模型在任务能力、
因果结构保持和通用语言建模能力方面的表现。

子模块:
    task_accuracy       — 特定任务准确率（IOI、Greater-than 等）
    causal_consistency  — 因果干预一致性（符号一致性 + Pearson 相关评估器）
    perplexity          — WikiText 困惑度，衡量通用语言建模能力
    ood_generators      — 分布外 (OOD) 测试集生成器
    ood_robustness      — OOD 鲁棒性评估（性能保持率）
    leakage_test        — 因果痕迹信息泄露测试（线性探针法）
"""
