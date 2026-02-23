"""Shared utilities for experiment scripts."""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import torch
import torch.nn as nn

# Ensure src/ is on the path for all scripts
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from rcid.alignment.cka import cka_matrix
from rcid.alignment.layer_matching import match_layers
from rcid.alignment.procrustes import compute_procrustes_matrices
from rcid.circuit.checkpoint_selection import select_checkpoints
from rcid.circuit.contrastive import ContrastiveDataset
from rcid.circuit.patching import extract_contrastive_differences
from rcid.data.factual_probing import FactualProbingDataset
from rcid.data.ioi import IOIDataset
from rcid.data.winogrande import WinoGrandeDataset
from rcid.models.adapter import ModelAdapter

logger = logging.getLogger(__name__)

MODEL_CONFIGS: dict[str, dict[str, str]] = {
    "qwen3": {"teacher": "Qwen/Qwen3-8B", "student": "Qwen/Qwen3-0.6B"},
    "llama3": {"teacher": "meta-llama/Llama-3.1-8B", "student": "meta-llama/Llama-3.2-1B"},
}

DEFAULT_TRAIN_CONFIG: dict = {
    "epochs": 20, "batch_size": 16, "lr": 5e-5, "weight_decay": 0.01,
    "lambda_kl": 1.0, "lambda_rcid": 1.0, "temperature": 2.0,
    "grad_clip": 1.0, "fp16": True, "scheduler": "cosine", "warmup_ratio": 0.05,
}


def build_dataset(
    task: str, tokenizer: object, n_samples: int = 500, seed: int = 42,
) -> tuple[object, ContrastiveDataset]:
    """Build contrastive dataset for a task.

    Returns:
        (wrapper_object, inner_ContrastiveDataset).
    """
    if task == "ioi":
        wrapper = IOIDataset(tokenizer=tokenizer, n_samples=n_samples, seed=seed)
    elif task == "factual":
        wrapper = FactualProbingDataset(tokenizer=tokenizer, seed=seed)
    elif task == "winogrande":
        wrapper = WinoGrandeDataset(tokenizer=tokenizer, seed=seed)
    else:
        raise ValueError(f"Unknown task: {task}")
    return wrapper, wrapper.dataset


def prepare_alignment(
    teacher: nn.Module,
    student: nn.Module,
    t_adapter: ModelAdapter,
    s_adapter: ModelAdapter,
    dataset: ContrastiveDataset,
    device: str,
    top_k: int = 20,
    diversity_ratio: float = 0.5,
    all_layer_W: bool = False,
) -> tuple[list[tuple[int, int]], dict[int, int], dict[int, torch.Tensor]]:
    """Extract diffs, select checkpoints, compute CKA mapping + Procrustes.

    CKA uses flattened (batch*seq, d) representations for maximum information
    retention (not mean-pooled).

    Args:
        all_layer_W: If True, compute Procrustes W for every mapped layer pair
            (needed by FitNets). If False (default), compute only for checkpoint
            layers (sufficient for InformedFitNets / RCID).

    Returns:
        (checkpoints, layer_mapping, W_matrices)
    """
    t_layers = list(range(t_adapter.get_num_layers(teacher)))
    s_layers = list(range(s_adapter.get_num_layers(student)))
    clean = dataset.clean_ids.to(device)
    corrupt = dataset.corrupt_ids.to(device)

    logger.info("Extracting teacher contrastive differences...")
    t_diffs = extract_contrastive_differences(teacher, t_adapter, clean, corrupt, t_layers)
    logger.info("Selecting causal checkpoints...")
    checkpoints = select_checkpoints(t_diffs, dataset, top_k=top_k,
                                     diversity_ratio=diversity_ratio)
    logger.info(f"Selected {len(checkpoints)} checkpoints")

    logger.info("Extracting student contrastive differences...")
    s_diffs = extract_contrastive_differences(student, s_adapter, clean, corrupt, s_layers)

    # CKA with flattened (batch*seq, d) — retains positional information
    logger.info("Computing CKA matrix and layer mapping...")
    cka_scores = cka_matrix(t_diffs, s_diffs)  # auto-flattens 3D → 2D
    layer_mapping = match_layers(cka_scores=cka_scores, strategy="greedy")

    if all_layer_W:
        # FitNets: need W for every mapped layer pair
        logger.info("Computing Procrustes alignment matrices for ALL mapped layers...")
        W_matrices = compute_procrustes_matrices(t_diffs, s_diffs, layer_mapping)
    else:
        # RCID / InformedFitNets: only need W for checkpoint layers
        cp_t = sorted({cp[0] for cp in checkpoints})
        cp_s = sorted({layer_mapping[tl] for tl in cp_t if tl in layer_mapping})
        t_cp_diffs = {l: t_diffs[l] for l in cp_t if l in t_diffs}
        s_cp_diffs = {l: s_diffs[l] for l in cp_s if l in s_diffs}
        logger.info("Computing Procrustes alignment matrices for checkpoint layers...")
        W_matrices = compute_procrustes_matrices(t_cp_diffs, s_cp_diffs, layer_mapping)

    return checkpoints, layer_mapping, W_matrices


def save_student_checkpoint(
    student: nn.Module,
    output_dir: Path,
    method: str,
    task: str,
    seed: int,
) -> str:
    """Save student state_dict and return the checkpoint path."""
    ckpt_path = output_dir / f"{method}_{task}_seed{seed}_student.pt"
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(student.state_dict(), ckpt_path)
    logger.info("Student checkpoint saved: %s", ckpt_path)
    return str(ckpt_path)


def load_perplexity_texts() -> list[str]:
    """Load evaluation texts for perplexity. Uses WikiText-2 if available."""
    try:
        from datasets import load_dataset
        wikitext = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
        texts = [t for t in wikitext["text"] if len(t.strip()) > 50][:100]
        if texts:
            logger.info("Loaded %d WikiText-2 texts for perplexity", len(texts))
            return texts
    except Exception as exc:
        logger.warning("Could not load WikiText-2: %s. Using fallback texts.", exc)

    return [
        "The quick brown fox jumps over the lazy dog.",
        "Natural language processing has made remarkable progress in recent years.",
        "Transformers use self-attention mechanisms to model long-range dependencies.",
        "Knowledge distillation transfers information from a large model to a small one.",
        "The residual stream carries information across transformer layers.",
        "Causal interventions reveal the internal mechanisms of neural networks.",
        "Contrastive learning captures the difference between similar inputs.",
        "Mechanistic interpretability aims to understand how models compute.",
        "The alignment of representations across models requires careful calibration.",
        "Evaluation of distilled models should go beyond task accuracy alone.",
    ]
