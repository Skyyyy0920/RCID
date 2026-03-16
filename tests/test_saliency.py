"""Tests for SaGD (Saliency-Guided Knowledge Distillation).

Validates SaliencyComputer: shape correctness, response zeroing,
distribution normalisation, JSD properties, and no-parameter-update
guarantee.
"""

from __future__ import annotations

import torch

from conftest import TinyTransformerModel, TinyAdapter
from rcid.distillation.saliency import SaliencyComputer


def _make_batch(
    model: TinyTransformerModel,
    batch_size: int = 2,
    seq_len: int = 8,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Create synthetic input_ids, attention_mask, and labels_mask."""
    vocab = model.config.hidden_size  # use hidden_size as proxy (100 in fixture)
    # Workaround: actual vocab is in lm_head.out_features
    vocab = model.lm_head.out_features

    input_ids = torch.randint(0, vocab, (batch_size, seq_len))
    attention_mask = torch.ones(batch_size, seq_len, dtype=torch.long)
    # First 4 positions = prompt, rest = response
    labels_mask = torch.zeros(batch_size, seq_len, dtype=torch.long)
    labels_mask[:, 4:] = 1
    return input_ids, attention_mask, labels_mask


# ── Test 1: saliency shape ────────────────────────────────────────────


def test_saliency_shape() -> None:
    """Saliency output must have shape (B, L)."""
    torch.manual_seed(42)
    model = TinyTransformerModel(n_layers=2, d_model=16, vocab_size=50)
    model.eval()

    computer = SaliencyComputer(temperature=2.0)
    B, L = 3, 10
    input_ids = torch.randint(0, 50, (B, L))
    attn_mask = torch.ones(B, L, dtype=torch.long)
    labels_mask = torch.zeros(B, L, dtype=torch.long)
    labels_mask[:, 5:] = 1

    saliency = computer.compute(model, input_ids, attn_mask, labels_mask)

    assert saliency.shape == (B, L), f"Expected ({B}, {L}), got {saliency.shape}"
    assert saliency.dtype == torch.float32 or saliency.dtype == torch.float16


# ── Test 2: response positions are zeroed ─────────────────────────────


def test_response_positions_zeroed() -> None:
    """Saliency at response positions must be exactly zero."""
    torch.manual_seed(42)
    model = TinyTransformerModel(n_layers=2, d_model=16, vocab_size=50)

    computer = SaliencyComputer()
    B, L = 2, 8
    input_ids = torch.randint(0, 50, (B, L))
    attn_mask = torch.ones(B, L, dtype=torch.long)
    labels_mask = torch.zeros(B, L, dtype=torch.long)
    labels_mask[:, 4:] = 1  # positions 4-7 are response

    saliency = computer.compute(model, input_ids, attn_mask, labels_mask)

    # Response positions must be zero
    response_saliency = saliency[:, 4:]
    assert response_saliency.abs().max() == 0.0, (
        f"Response saliency not zero: {response_saliency}"
    )

    # Prompt positions should generally be nonzero
    prompt_saliency = saliency[:, :4]
    assert prompt_saliency.abs().sum() > 0, "All prompt saliency is zero"


# ── Test 3: distribution sums to 1 ───────────────────────────────────


def test_distribution_sums_to_one() -> None:
    """to_distribution() must return a valid probability distribution."""
    torch.manual_seed(42)
    model = TinyTransformerModel(n_layers=2, d_model=16, vocab_size=50)

    computer = SaliencyComputer(temperature=2.0)
    B, L = 2, 8
    input_ids = torch.randint(0, 50, (B, L))
    attn_mask = torch.ones(B, L, dtype=torch.long)
    labels_mask = torch.zeros(B, L, dtype=torch.long)
    labels_mask[:, 4:] = 1

    saliency = computer.compute(model, input_ids, attn_mask, labels_mask)
    dist = computer.to_distribution(saliency, labels_mask, attention_mask=attn_mask)

    # Distribution should sum to 1 over prompt positions for each sample
    for b in range(B):
        prompt_sum = dist[b, :4].sum().item()
        assert abs(prompt_sum - 1.0) < 1e-5, (
            f"Sample {b}: prompt distribution sums to {prompt_sum}"
        )

    # Response positions should have 0 probability
    assert dist[:, 4:].abs().max() < 1e-6, "Response positions have nonzero prob"


# ── Test 3b: padding positions must not leak probability ─────────────


def test_distribution_no_padding_leakage() -> None:
    """Padding positions must get 0 probability, not exp(0)/Z leakage."""
    computer = SaliencyComputer(temperature=2.0)
    B, L = 2, 12

    # Simulate: positions 0-3 are prompt, 4-7 are response, 8-11 are padding
    saliency = torch.zeros(B, L)
    saliency[:, :4] = torch.rand(B, 4) + 0.1  # non-zero prompt saliency

    labels_mask = torch.zeros(B, L, dtype=torch.long)
    labels_mask[:, 4:8] = 1  # response positions

    attention_mask = torch.ones(B, L, dtype=torch.long)
    attention_mask[:, 8:] = 0  # padding positions

    dist = computer.to_distribution(saliency, labels_mask, attention_mask=attention_mask)

    # Only prompt positions (0-3) should have non-zero probability
    for b in range(B):
        prompt_sum = dist[b, :4].sum().item()
        assert abs(prompt_sum - 1.0) < 1e-5, (
            f"Sample {b}: prompt distribution sums to {prompt_sum}, expected 1.0"
        )
    # Response positions (4-7) must be 0
    assert dist[:, 4:8].abs().max() < 1e-6, "Response positions have nonzero prob"
    # Padding positions (8-11) must be 0
    assert dist[:, 8:].abs().max() < 1e-6, (
        f"Padding leakage! Padding positions have prob: {dist[:, 8:]}"
    )


# ── Test 4: JSD = 0 for identical distributions ──────────────────────


def test_jsd_zero_for_identical() -> None:
    """JSD(T, T) must be zero (or very close)."""
    computer = SaliencyComputer()
    B, L = 3, 6
    labels_mask = torch.zeros(B, L, dtype=torch.long)
    labels_mask[:, 3:] = 1

    # Create a valid distribution
    saliency = torch.rand(B, L)
    saliency[:, 3:] = 0  # zero response positions
    dist = computer.to_distribution(saliency, labels_mask)

    jsd = computer.divergence(dist, dist, labels_mask)  # (B,)

    assert jsd.shape == (B,)
    assert jsd.abs().max() < 1e-5, f"JSD not zero for identical: {jsd}"


# ── Test 5: JSD > 0 for different distributions ──────────────────────


def test_jsd_positive_for_different() -> None:
    """JSD(T, S) must be positive when distributions differ."""
    computer = SaliencyComputer()
    B, L = 2, 8
    labels_mask = torch.zeros(B, L, dtype=torch.long)
    labels_mask[:, 4:] = 1

    # Create two different distributions
    saliency_T = torch.tensor([[1.0, 0.0, 0.0, 0.0, 0, 0, 0, 0],
                                [1.0, 0.0, 0.0, 0.0, 0, 0, 0, 0]])
    saliency_S = torch.tensor([[0.0, 0.0, 0.0, 1.0, 0, 0, 0, 0],
                                [0.0, 0.0, 0.0, 1.0, 0, 0, 0, 0]])

    dist_T = computer.to_distribution(saliency_T, labels_mask)
    dist_S = computer.to_distribution(saliency_S, labels_mask)
    jsd = computer.divergence(dist_T, dist_S, labels_mask)  # (B,)

    assert jsd.shape == (B,)
    assert (jsd > 0.01).all(), f"JSD not positive for different dists: {jsd}"

    # JSD should be bounded by log(2) ≈ 0.693
    import math
    assert (jsd <= math.log(2) + 1e-5).all(), f"JSD exceeds log(2): {jsd}"


# ── Test 6: no model parameters updated ──────────────────────────────


def test_no_model_params_updated() -> None:
    """compute() must not modify any model parameter or accumulate gradients."""
    torch.manual_seed(42)
    model = TinyTransformerModel(n_layers=2, d_model=16, vocab_size=50)

    # Store original parameter values and grad state
    original_params: dict[str, torch.Tensor] = {}
    original_grads: dict[str, torch.Tensor | None] = {}
    for name, p in model.named_parameters():
        original_params[name] = p.data.clone()
        original_grads[name] = p.grad.clone() if p.grad is not None else None

    computer = SaliencyComputer()
    B, L = 2, 6
    input_ids = torch.randint(0, 50, (B, L))
    attn_mask = torch.ones(B, L, dtype=torch.long)
    labels_mask = torch.zeros(B, L, dtype=torch.long)
    labels_mask[:, 3:] = 1

    _ = computer.compute(model, input_ids, attn_mask, labels_mask)

    # Verify no parameter values changed
    for name, p in model.named_parameters():
        assert torch.equal(p.data, original_params[name]), (
            f"Parameter {name} was modified during saliency computation"
        )

    # Verify no new gradients accumulated
    for name, p in model.named_parameters():
        if original_grads[name] is None:
            assert p.grad is None, (
                f"Parameter {name} gained a gradient during saliency computation"
            )
        else:
            assert torch.equal(p.grad, original_grads[name]), (
                f"Parameter {name} gradient was modified"
            )


# ── Test 7: create_graph=True enables gradient flow ──────────────────


def test_create_graph_gradient_flow() -> None:
    """With create_graph=True, L_sal gradients must flow to model params."""
    torch.manual_seed(42)
    model = TinyTransformerModel(n_layers=2, d_model=16, vocab_size=50)
    model.train()

    computer = SaliencyComputer()
    B, L = 2, 6
    input_ids = torch.randint(0, 50, (B, L))
    attn_mask = torch.ones(B, L, dtype=torch.long)
    labels_mask = torch.zeros(B, L, dtype=torch.long)
    labels_mask[:, 3:] = 1

    saliency = computer.compute(
        model, input_ids, attn_mask, labels_mask, create_graph=True,
    )

    # Saliency must have grad_fn (differentiable w.r.t. model params)
    assert saliency.grad_fn is not None, (
        "Saliency has no grad_fn with create_graph=True"
    )

    # Build a dummy saliency alignment loss and backprop
    loss = saliency.sum()
    loss.backward()

    # At least some model parameters must have received gradients
    has_grad = any(
        p.grad is not None and p.grad.abs().sum() > 0
        for p in model.parameters()
    )
    assert has_grad, "No model parameter received gradient from saliency loss"


def test_create_graph_false_no_grad_fn() -> None:
    """With create_graph=False (default), saliency must be detached."""
    torch.manual_seed(42)
    model = TinyTransformerModel(n_layers=2, d_model=16, vocab_size=50)

    computer = SaliencyComputer()
    B, L = 2, 6
    input_ids = torch.randint(0, 50, (B, L))
    attn_mask = torch.ones(B, L, dtype=torch.long)
    labels_mask = torch.zeros(B, L, dtype=torch.long)
    labels_mask[:, 3:] = 1

    saliency = computer.compute(model, input_ids, attn_mask, labels_mask)

    assert saliency.grad_fn is None, (
        "Saliency should be detached with create_graph=False"
    )


# ── Test 9: per-sample KL reweighting logic ──────────────────────────


def test_sagd_weight_normalisation() -> None:
    """SaGD weights (softmax(jsd/tau) * B) should have mean ≈ 1."""
    import torch.nn.functional as F

    B = 4
    # Simulate different JSD values
    jsd = torch.tensor([0.1, 0.3, 0.5, 0.2])
    tau_w = 1.0

    weights = F.softmax(jsd / tau_w, dim=0) * B  # (B,)

    assert weights.shape == (B,)
    assert abs(weights.mean().item() - 1.0) < 1e-5, (
        f"Weights mean not 1.0: {weights.mean()}"
    )
    # Higher JSD should get higher weight
    assert weights[2] > weights[0], "Highest JSD should get highest weight"
