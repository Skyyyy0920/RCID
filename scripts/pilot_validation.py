"""Pilot validation: go/no-go checks before full experiments.

Usage:
    python scripts/pilot_validation.py --model_family qwen3 --device cuda:0
    python scripts/pilot_validation.py --model_family llama3 --device cuda:0
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
import torch.nn as nn

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from rcid.models.adapter import ModelAdapter, get_adapter
from rcid.models.teacher import load_teacher
from rcid.models.student import load_student
from rcid.data.ioi import IOIDataset, build_single_token_names

MODEL_CONFIGS = {
    "qwen3": {"teacher": "Qwen/Qwen3-8B", "student": "Qwen/Qwen3-0.6B"},
    "llama3": {"teacher": "meta-llama/Llama-3.1-8B", "student": "meta-llama/Llama-3.2-1B"},
}


def _extract_residual_norms(
    model: nn.Module,
    adapter: ModelAdapter,
    clean_ids: torch.Tensor,
    corrupt_ids: torch.Tensor,
) -> dict[int, torch.Tensor]:
    """Extract per-layer contrastive difference norms (inline for pilot)."""
    n_layers = adapter.get_num_layers(model)
    device = clean_ids.device

    def _run_and_capture(input_ids):
        cache = {}
        handles = []
        for l in range(n_layers):
            hp = adapter.get_residual_hook_point(model, l)
            def _hook(mod, inp, out, idx=l):
                cache[idx] = adapter.parse_layer_output(out).detach().clone()
            handles.append(hp.register_forward_hook(_hook))
        try:
            with torch.no_grad():
                model(input_ids)
        finally:
            for h in handles:
                h.remove()
        return cache

    h_clean = _run_and_capture(clean_ids)
    h_corrupt = _run_and_capture(corrupt_ids)

    norms = {}
    for l in range(n_layers):
        diff = h_clean[l] - h_corrupt[l]  # (batch, seq, d_model)
        norms[l] = diff.norm(dim=-1).mean(dim=0)  # (seq_len,)
    return norms


def _patch_and_measure(
    model: nn.Module,
    adapter: ModelAdapter,
    clean_ids: torch.Tensor,
    corrupt_ids: torch.Tensor,
    layer: int,
    token_pos: int,
    answer_pos: torch.Tensor,
    correct_id: torch.Tensor,
    wrong_id: torch.Tensor,
) -> float:
    """Inline activation patching for pilot (single checkpoint)."""
    device = clean_ids.device
    with torch.no_grad():
        orig_logits = model(clean_ids).logits
    batch_idx = torch.arange(clean_ids.shape[0], device=device)
    orig_diff = (
        orig_logits[batch_idx, answer_pos].gather(1, correct_id.unsqueeze(1)).squeeze(1)
        - orig_logits[batch_idx, answer_pos].gather(1, wrong_id.unsqueeze(1)).squeeze(1)
    )

    # Get corrupt value
    cache = {}
    hp = adapter.get_residual_hook_point(model, layer)
    handle = hp.register_forward_hook(
        lambda mod, inp, out: cache.update({
            "h": adapter.parse_layer_output(out)[:, token_pos, :].detach().clone()
        })
    )
    try:
        with torch.no_grad():
            model(corrupt_ids)
    finally:
        handle.remove()

    # Patch
    def _patch_hook(mod, inp, out):
        h = adapter.parse_layer_output(out).clone()
        h[:, token_pos, :] = cache["h"]
        return (h,) + out[1:]

    handle2 = hp.register_forward_hook(_patch_hook)
    try:
        with torch.no_grad():
            patched_logits = model(clean_ids).logits
    finally:
        handle2.remove()

    patched_diff = (
        patched_logits[batch_idx, answer_pos].gather(1, correct_id.unsqueeze(1)).squeeze(1)
        - patched_logits[batch_idx, answer_pos].gather(1, wrong_id.unsqueeze(1)).squeeze(1)
    )
    delta = (orig_diff - patched_diff).mean().item()
    return delta


def main() -> None:
    parser = argparse.ArgumentParser(description="RCID Pilot Validation")
    parser.add_argument("--model_family", choices=["qwen3", "llama3"], required=True)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()

    cfg = MODEL_CONFIGS[args.model_family]
    print(f"\n{'='*60}")
    print(f"  PILOT VALIDATION: {args.model_family}")
    print(f"{'='*60}\n")

    # Step 1: Load teacher and test IOI accuracy
    print("[Step 1] Loading teacher model...")
    teacher, adapter, tokenizer = load_teacher(cfg["teacher"], device=args.device)
    print(f"  Model loaded: {cfg['teacher']}")
    print(f"  Layers: {adapter.get_num_layers(teacher)}, d_model: {adapter.get_hidden_size(teacher)}")

    # Step 2: Validate single-token name pool
    print("\n[Step 2] Checking single-token name pool...")
    name_pool = build_single_token_names(tokenizer)
    print(f"  Valid single-token names: {len(name_pool)}")
    if len(name_pool) < 20:
        print(f"  FAIL: Need >= 20 names, got {len(name_pool)}")
        print(f"  Available: {name_pool}")
        sys.exit(1)
    else:
        print(f"  PASS: {len(name_pool)} names available")
        print(f"  Sample: {name_pool[:10]}")

    # Step 3: Build IOI dataset and test teacher accuracy
    print("\n[Step 3] Testing teacher IOI accuracy...")
    ioi = IOIDataset(tokenizer=tokenizer, n_samples=50, name_pool=name_pool, seed=42)
    ds = ioi.dataset
    device = args.device

    with torch.no_grad():
        logits = teacher(ds.clean_ids.to(device)).logits
    batch_idx = torch.arange(len(ds), device=device)
    ans_logits = logits[batch_idx, ds.answer_pos.to(device)]
    correct = ans_logits.gather(1, ds.correct_token_id.to(device).unsqueeze(1)).squeeze(1)
    wrong = ans_logits.gather(1, ds.wrong_token_id.to(device).unsqueeze(1)).squeeze(1)
    accuracy = (correct > wrong).float().mean().item()
    print(f"  Teacher IOI accuracy: {accuracy:.2%}")
    if accuracy < 0.95:
        print(f"  WARNING: accuracy < 95%, IOI templates may need adaptation")

    # Step 4: Extract contrastive difference norms
    print("\n[Step 4] Extracting contrastive difference norms...")
    subset = 10
    norms = _extract_residual_norms(
        teacher, adapter,
        ds.clean_ids[:subset].to(device),
        ds.corrupt_ids[:subset].to(device),
    )
    print("  Layer-wise mean norms (averaged over positions):")
    low_layers_norm = 0.0
    high_layers_norm = 0.0
    n_layers = len(norms)
    for l in sorted(norms.keys()):
        mean_norm = norms[l].mean().item()
        bar = "█" * int(mean_norm * 2)
        print(f"    Layer {l:2d}: {mean_norm:.4f} {bar}")
        if l < n_layers // 2:
            low_layers_norm += mean_norm
        else:
            high_layers_norm += mean_norm

    if high_layers_norm <= low_layers_norm:
        print("  WARNING: High layers not showing larger norms than low layers")
    else:
        print(f"  PASS: High-layer mean ({high_layers_norm/(n_layers//2):.4f}) > "
              f"Low-layer mean ({low_layers_norm/(n_layers//2):.4f})")

    # Step 5: Activation patching
    print("\n[Step 5] Activation patching at best checkpoint...")
    best_layer = max(norms.keys(), key=lambda l: norms[l].max().item())
    best_pos = norms[best_layer].argmax().item()
    print(f"  Best checkpoint: layer={best_layer}, pos={best_pos}")

    delta = _patch_and_measure(
        teacher, adapter,
        ds.clean_ids[:subset].to(device),
        ds.corrupt_ids[:subset].to(device),
        layer=best_layer, token_pos=best_pos,
        answer_pos=ds.answer_pos[:subset].to(device),
        correct_id=ds.correct_token_id[:subset].to(device),
        wrong_id=ds.wrong_token_id[:subset].to(device),
    )
    print(f"  Causal effect (delta): {delta:.4f}")
    if abs(delta) < 0.01:
        print("  WARNING: Causal effect near zero at best checkpoint")
    else:
        print(f"  PASS: Significant causal effect")

    # Step 6: Student baseline
    print("\n[Step 6] Loading student and testing baseline...")
    student, s_adapter, _ = load_student(cfg["student"], device=args.device)
    student.eval()
    with torch.no_grad():
        s_logits = student(ds.clean_ids.to(device)).logits
    s_ans = s_logits[batch_idx, ds.answer_pos.to(device)]
    s_correct = s_ans.gather(1, ds.correct_token_id.to(device).unsqueeze(1)).squeeze(1)
    s_wrong = s_ans.gather(1, ds.wrong_token_id.to(device).unsqueeze(1)).squeeze(1)
    s_accuracy = (s_correct > s_wrong).float().mean().item()
    print(f"  Student IOI baseline accuracy: {s_accuracy:.2%}")

    print(f"\n{'='*60}")
    print(f"  PILOT VALIDATION COMPLETE: {args.model_family}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
