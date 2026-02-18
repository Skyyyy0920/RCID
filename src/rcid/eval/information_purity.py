"""Information purity test: selectivity of h vs d representations."""

from __future__ import annotations

import torch


def evaluate_information_purity(
    representations: torch.Tensor,  # (N, d_model)
    task_labels: torch.Tensor,       # (N,) binary
    control_labels: torch.Tensor,    # (N,) — template index or irrelevant attribute
    test_size: float = 0.2,
    seed: int = 42,
) -> dict[str, float]:
    """Compare task-relevant vs task-irrelevant information in representations.

    Selectivity = task_accuracy - control_accuracy.
    Higher selectivity means the representation is more task-specific.

    Args:
        representations: Feature vectors to evaluate.
        task_labels: Binary task labels (e.g., which answer is correct).
        control_labels: Control labels (e.g., template index).
        test_size: Fraction for test split.
        seed: Random seed for reproducibility.

    Returns:
        Dict with task_accuracy, control_accuracy, selectivity.
    """
    import numpy as np
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import train_test_split

    X = representations.detach().cpu().numpy()
    y_task = task_labels.detach().cpu().numpy()
    y_ctrl = control_labels.detach().cpu().numpy()

    # Task probe
    X_train, X_test, yt_train, yt_test = train_test_split(
        X, y_task, test_size=test_size, random_state=seed,
    )
    clf_task = LogisticRegression(max_iter=1000, random_state=seed)
    clf_task.fit(X_train, yt_train)
    task_acc = clf_task.score(X_test, yt_test)

    # Control probe
    X_train_c, X_test_c, yc_train, yc_test = train_test_split(
        X, y_ctrl, test_size=test_size, random_state=seed,
    )
    n_classes_ctrl = len(np.unique(y_ctrl))
    if n_classes_ctrl < 2:
        control_acc = 1.0  # degenerate
    else:
        clf_ctrl = LogisticRegression(max_iter=1000, random_state=seed)
        clf_ctrl.fit(X_train_c, yc_train)
        control_acc = clf_ctrl.score(X_test_c, yc_test)

    selectivity = task_acc - control_acc

    return {
        "task_accuracy": task_acc,
        "control_accuracy": control_acc,
        "selectivity": selectivity,
    }
