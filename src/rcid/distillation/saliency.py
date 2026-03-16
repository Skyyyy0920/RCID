"""Saliency-Guided Knowledge Distillation (SaGD).

Computes input saliency (gradient of response log-likelihood w.r.t. input
embeddings) and uses saliency divergence between teacher and student to
reweight per-sample KL loss.

Unlike TSD (Transfer Saliency Distillation) which adds saliency as an extra
loss term, SaGD uses it as a sample importance signal: samples where teacher
and student attend to very different parts of the prompt receive higher KL
weight.

Usage::

    computer = SaliencyComputer(temperature=2.0)
    saliency = computer.compute(model, input_ids, attention_mask, labels_mask)
    dist = computer.to_distribution(saliency, labels_mask)
    jsd = computer.divergence(dist_T, dist_S, labels_mask)
"""

from __future__ import annotations

import logging
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


class SaliencyComputer:
    """Compute input saliency via gradient of response log-prob w.r.t. embeddings.

    Saliency at position t = ||d(LL_response) / d(e_t)||_2
    where LL_response = sum_{t in response} log p(x_t | x_{<t}).

    This measures how much each prompt token's embedding affects the model's
    response generation probability.

    Args:
        temperature: Temperature for softmax when converting saliency to
            a distribution.
        eps: Small constant for numerical stability.
    """

    def __init__(self, temperature: float = 2.0, eps: float = 1e-8) -> None:
        self.temperature = temperature
        self.eps = eps

    @torch.enable_grad()
    def compute(
        self,
        model: nn.Module,
        input_ids: torch.Tensor,       # (B, L)
        attention_mask: torch.Tensor,   # (B, L)
        labels_mask: torch.Tensor,      # (B, L) — 1=response, 0=prompt/pad
    ) -> torch.Tensor:
        """Compute per-position saliency for each sample.

        Returns:
            saliency: (B, L) — L2 norm of embedding gradient at each position.
                Response positions are zeroed out (only prompt saliency matters).
        """
        was_training = model.training
        model.eval()

        # Get embedding layer
        if hasattr(model, "get_input_embeddings"):
            embed_layer = model.get_input_embeddings()
        elif hasattr(model, "model") and hasattr(model.model, "embed_tokens"):
            embed_layer = model.model.embed_tokens
        else:
            raise AttributeError("Cannot find embedding layer on model")

        # Temporarily freeze model parameters so backward only flows to embed
        param_states: list[tuple[nn.Parameter, bool]] = []
        for p in model.parameters():
            param_states.append((p, p.requires_grad))
            p.requires_grad_(False)

        try:
            # Create detached embedding as leaf tensor for gradient
            with torch.no_grad():
                embed = embed_layer(input_ids)  # (B, L, d_model)
            embed = embed.detach().requires_grad_(True)

            # Forward pass using inputs_embeds
            outputs = model(
                inputs_embeds=embed, attention_mask=attention_mask,
            )
            logits = outputs.logits  # (B, L, V)

            # Compute response log-likelihood (next-token prediction)
            shift_logits = logits[:, :-1, :].float()  # (B, L-1, V)
            shift_labels = input_ids[:, 1:]             # (B, L-1)
            shift_resp = labels_mask[:, 1:].float()     # (B, L-1)

            log_probs = F.log_softmax(shift_logits, dim=-1)  # (B, L-1, V)
            token_log_probs = log_probs.gather(
                dim=-1, index=shift_labels.unsqueeze(-1),
            ).squeeze(-1)  # (B, L-1)

            # Sum log-probs over response positions
            response_ll = (token_log_probs * shift_resp).sum()  # scalar

            # Backward to get gradients w.r.t. embeddings
            response_ll.backward()

            assert embed.grad is not None, "Embedding gradient is None"

            # Saliency = L2 norm of gradient at each position
            saliency = embed.grad.norm(dim=-1)  # (B, L)

            # Zero out response positions — only prompt saliency matters
            prompt_mask = (1 - labels_mask).float() * attention_mask.float()
            saliency = saliency * prompt_mask  # (B, L)
            saliency = saliency.detach()

        finally:
            # Restore parameter gradient states
            for p, req in param_states:
                p.requires_grad_(req)

        if was_training:
            model.train()

        return saliency  # (B, L)

    def to_distribution(
        self,
        saliency: torch.Tensor,    # (B, L)
        labels_mask: torch.Tensor,  # (B, L)
    ) -> torch.Tensor:
        """Convert saliency to a probability distribution over prompt positions.

        Applies softmax(saliency / temperature) over prompt positions only.
        Response and padding positions get probability 0.

        Returns:
            dist: (B, L) — distribution summing to 1 over prompt positions.
        """
        prompt_mask = (1 - labels_mask).float()  # (B, L) — 1=prompt
        scores = saliency.float() / self.temperature  # (B, L)

        # Non-prompt positions → -inf so softmax gives 0
        scores = scores.masked_fill(prompt_mask < 0.5, float("-inf"))
        dist = F.softmax(scores, dim=-1)  # (B, L)

        # Handle all-masked rows (edge case)
        nan_mask = dist.isnan()
        if nan_mask.any():
            dist = dist.masked_fill(nan_mask, 0.0)

        return dist  # (B, L)

    def divergence(
        self,
        dist_T: torch.Tensor,      # (B, L) — teacher saliency distribution
        dist_S: torch.Tensor,      # (B, L) — student saliency distribution
        labels_mask: torch.Tensor,  # (B, L)
    ) -> torch.Tensor:
        """Compute Jensen-Shannon Divergence between two saliency distributions.

        JSD(T || S) = 0.5 * KL(T || M) + 0.5 * KL(S || M),  M = 0.5*(T+S)

        Returns:
            jsd: (B,) — per-sample JSD, bounded in [0, log(2)].
        """
        eps = self.eps
        M = 0.5 * (dist_T + dist_S)  # (B, L)

        prompt_mask = (1 - labels_mask).float()  # (B, L)

        # KL(T || M) = sum T * log(T / M) over prompt positions
        log_T_over_M = torch.log((dist_T + eps) / (M + eps))  # (B, L)
        log_S_over_M = torch.log((dist_S + eps) / (M + eps))  # (B, L)

        kl_T_M = (dist_T * log_T_over_M * prompt_mask).sum(dim=-1)  # (B,)
        kl_S_M = (dist_S * log_S_over_M * prompt_mask).sum(dim=-1)  # (B,)

        jsd = 0.5 * kl_T_M + 0.5 * kl_S_M  # (B,)

        # Clamp to [0, log(2)] for numerical safety
        log2 = torch.log(torch.tensor(2.0, device=jsd.device)).item()
        jsd = jsd.clamp(min=0.0, max=log2)

        return jsd  # (B,)
