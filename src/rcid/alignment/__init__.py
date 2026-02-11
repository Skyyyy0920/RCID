"""跨模型对齐模块。

解决教师和学生模型残差流维度不同 (d_T ≠ d_S) 的问题，
通过 Procrustes 正交对齐和 CKA 层匹配，建立两个模型表示空间之间的映射。

子模块:
    procrustes     — Procrustes 正交对齐，求解冻结矩阵 W*
    layer_matching — 基于 CKA 的教师-学生层匹配搜索
    cka            — Centered Kernel Alignment 计算
"""
