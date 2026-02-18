"""Task accuracy evaluation for contrastive datasets."""

from __future__ import annotations

import torch
import torch.nn as nn

from rcid.circuit.contrastive import ContrastiveDataset
from rcid.models.adapter import ModelAdapter


def evaluate_task_accuracy(
    model: nn.Module,
    adapter: ModelAdapter,
    dataset: ContrastiveDataset,
    batch_size: int = 32,
) -> dict[str, float]:
    """Evaluate task accuracy on a contrastive dataset.

    Checks if logit(correct) > logit(wrong) at the answer position.

    Returns:
        Dict with accuracy, logit_diff_mean, logit_diff_std.
    """
    device = next(model.parameters()).device
    model.eval()

    all_correct = 0
    all_diffs: list[float] = []
    n = len(dataset)

    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        clean = dataset.clean_ids[start:end].to(device)  # (bs, seq)
        answer_pos = dataset.answer_pos[start:end].to(device)  # (bs,)
        correct_id = dataset.correct_token_id[start:end].to(device)  # (bs,)
        wrong_id = dataset.wrong_token_id[start:end].to(device)  # (bs,)

        with torch.no_grad():
            logits = model(clean).logits  # (bs, seq, vocab)

        bs = clean.shape[0]
        batch_idx = torch.arange(bs, device=device)
        logits_at_ans = logits[batch_idx, answer_pos, :]  # (bs, vocab)

        correct_logits = logits_at_ans.gather(1, correct_id.unsqueeze(1)).squeeze(1)
        wrong_logits = logits_at_ans.gather(1, wrong_id.unsqueeze(1)).squeeze(1)
        diffs = correct_logits - wrong_logits  # (bs,)

        all_correct += (diffs > 0).sum().item()
        all_diffs.extend(diffs.cpu().tolist())

    diffs_tensor = torch.tensor(all_diffs)
    return {
        "accuracy": all_correct / max(n, 1),
        "logit_diff_mean": diffs_tensor.mean().item(),
        "logit_diff_std": diffs_tensor.std().item(),
    }
