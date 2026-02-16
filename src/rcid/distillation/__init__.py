"""蒸馏训练模块。

实现 RCID 方法的核心训练逻辑，包括 L_RCID 损失函数、训练循环，
以及用于对比实验的基线蒸馏方法（标准 KD、FitNets、Prakash）。

子模块:
    rcid_loss  — L_RCID 损失函数，计算归一化因果痕迹的对齐误差
    trainer    — 蒸馏训练循环，协调教师痕迹预计算与学生在线更新
    baselines  — 基线蒸馏方法实现（Standard KD / FitNets / Prakash）
    tinybert   — TinyBERT 风格蒸馏（hidden states + attention matching）
    minilm     — MiniLM 风格蒸馏（Value-Value relation matrix matching）
    informed_fitnets — Informed FitNets 消融基线（RCID 检查点 + 完整表示匹配）
"""
