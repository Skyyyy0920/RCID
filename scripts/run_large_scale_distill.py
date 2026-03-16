"""Large-scale distillation with adaptive KL + RCID/SaGD methods on Dolly-15K.

Trains a student on instruction data with sequence-level KL, using one of
several adaptive divergence methods, RCID, or SaGD regularisation.  Evaluation
uses ROUGE-L on the Dolly test split.

Methods
-------
KL methods (single data stream):
  standard_kd       — forward KL baseline
  reverse_kl        — reverse KL baseline
  standard_kd_akl   — AKL (Wu et al., COLING 2025)
  standard_kd_klr   — KL-Ratio adaptive (ours)

RCID methods (dual data stream: KL + contrastive regulariser):
  standard_kd_rcid             — KL + RCID (contrastive diff matching)
  standard_kd_fitnets          — KL + FitNets (all-layer repr. matching)
  standard_kd_informed_fitnets — KL + InformedFitNets (checkpoint matching)

Usage::

    python scripts/run_large_scale_distill.py --model_family qwen3 \
        --method standard_kd --device cuda:0

    python scripts/run_large_scale_distill.py --model_family qwen3 \
        --method standard_kd_rcid --contrastive_pairs_path data/contrastive_pairs/

Output directory structure::

    outputs/large_scale/{model_family}/{method}/seed_{seed}/
        ├── student_final.pt
        ├── training_log.json
        ├── training_stats.jsonl
        ├── eval_results.json        (ROUGE-L)
        └── test_generations.json    (generated texts for inspection)
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import Any

import torch

# Ensure src/ is on the path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from rcid import set_all_seeds
from rcid.models.teacher import load_teacher
from rcid.models.student import load_student

logger = logging.getLogger(__name__)

MODEL_CONFIGS: dict[str, dict[str, str]] = {
    "qwen3": {"teacher": "Qwen/Qwen3-8B", "student": "Qwen/Qwen3-0.6B"},
    "llama3": {
        "teacher": "meta-llama/Llama-3.1-8B",
        "student": "meta-llama/Llama-3.2-1B",
    },
}

METHODS = [
    # KL methods
    "standard_kd", "reverse_kl", "standard_kd_akl", "standard_kd_klr",
    # RCID methods
    "standard_kd_rcid", "standard_kd_fitnets", "standard_kd_informed_fitnets",
    # SaGD methods
    "standard_kd_sagd",
]

_RCID_METHODS = {"standard_kd_rcid", "standard_kd_fitnets", "standard_kd_informed_fitnets"}
_SAGD_METHODS = {"standard_kd_sagd"}

DEFAULT_DATASET = "databricks/databricks-dolly-15k"


# ------------------------------------------------------------------
# RCID alignment helpers
# ------------------------------------------------------------------

def _compute_alignment(
    teacher: torch.nn.Module,
    student: torch.nn.Module,
    t_adapter: Any,
    s_adapter: Any,
    contrastive_ds: Any,
    device: str,
    top_k: int = 10,
) -> tuple[dict[int, int], dict[int, torch.Tensor], list[tuple[int, int]]]:
    """Compute CKA -> layer matching -> Procrustes -> checkpoints.

    Returns (layer_mapping, W_matrices, checkpoints).
    """
    from rcid.alignment.cka import cka_matrix
    from rcid.alignment.layer_matching import match_layers
    from rcid.alignment.procrustes import compute_procrustes_matrices
    from rcid.circuit.patching import extract_contrastive_differences
    from rcid.circuit.checkpoint_selection import select_checkpoints

    n_t = t_adapter.get_num_layers(teacher)
    n_s = s_adapter.get_num_layers(student)

    # Use a subset for alignment (100 pairs is plenty)
    n_align = min(100, len(contrastive_ds))
    if n_align == 0:
        raise ValueError(
            "Contrastive dataset is empty — cannot compute alignment. "
            "Check --contrastive_pairs_path or the JSON content."
        )
    clean_ids = contrastive_ds.clean_ids[:n_align].to(device)
    corrupt_ids = contrastive_ds.corrupt_ids[:n_align].to(device)
    align_bs = max(1, min(32, n_align))

    t_layers = list(range(n_t))
    s_layers = list(range(n_s))

    logger.info("Extracting teacher contrastive diffs (%d layers)...", n_t)
    t_diffs_pooled = extract_contrastive_differences(
        teacher, t_adapter, clean_ids, corrupt_ids, t_layers,
        batch_size=align_bs, pool_seq=True,
    )
    logger.info("Extracting student contrastive diffs (%d layers)...", n_s)
    s_diffs_pooled = extract_contrastive_differences(
        student, s_adapter, clean_ids, corrupt_ids, s_layers,
        batch_size=align_bs, pool_seq=True,
    )

    # CKA -> layer matching
    logger.info("Computing CKA matrix...")
    cka = cka_matrix(t_diffs_pooled, s_diffs_pooled)
    layer_mapping = match_layers(cka, n_t, n_s)
    logger.info("Layer mapping: %s", layer_mapping)

    # Procrustes matrices
    logger.info("Computing Procrustes alignment matrices...")
    W_matrices = compute_procrustes_matrices(t_diffs_pooled, s_diffs_pooled, layer_mapping)

    # Checkpoint selection (needs full-sequence diffs)
    logger.info("Selecting top-%d causal checkpoints...", top_k)
    t_diffs_full = extract_contrastive_differences(
        teacher, t_adapter, clean_ids, corrupt_ids, t_layers,
        batch_size=align_bs, pool_seq=False,
    )
    checkpoints = select_checkpoints(t_diffs_full, contrastive_ds, top_k=top_k)
    logger.info("Selected checkpoints: %s", checkpoints)

    return layer_mapping, W_matrices, checkpoints


def _build_rcid_loss_fn(
    method: str,
    layer_mapping: dict[int, int],
    W_matrices: dict[int, torch.Tensor],
    checkpoints: list[tuple[int, int]],
) -> torch.nn.Module:
    """Build the appropriate regulariser loss module for the RCID method."""
    if method == "standard_kd_rcid":
        from rcid.distillation.rcid_loss import RCIDLoss
        return RCIDLoss(checkpoints, layer_mapping, W_matrices)
    elif method == "standard_kd_fitnets":
        from rcid.distillation.baselines import FitNetsLoss
        return FitNetsLoss(layer_mapping, W_matrices)
    elif method == "standard_kd_informed_fitnets":
        from rcid.distillation.baselines import InformedFitNetsLoss
        return InformedFitNetsLoss(checkpoints, layer_mapping, W_matrices)
    else:
        raise ValueError(f"Unknown RCID method: {method}")


# ------------------------------------------------------------------
# ROUGE-L evaluation
# ------------------------------------------------------------------

def _evaluate_rouge(
    student: torch.nn.Module,
    tokenizer: Any,
    save_dir: Path,
    device: str,
) -> dict[str, Any]:
    """Evaluate on Dolly test split with ROUGE-L."""
    from rcid.data.dolly_utils import get_dolly_prompts
    from rcid.eval.rouge_eval import evaluate_rouge, save_generations

    eval_prompts, eval_refs = get_dolly_prompts(split="test")
    logger.info("Evaluating ROUGE-L on %d Dolly test samples...", len(eval_prompts))

    results = evaluate_rouge(
        student, tokenizer, eval_prompts, eval_refs,
        max_new_tokens=256, batch_size=8,
    )
    logger.info("Test ROUGE-L: %.4f", results["rouge_l_f"])

    with open(save_dir / "eval_results.json", "w", encoding="utf-8") as f:
        json.dump(
            {k: v for k, v in results.items() if k != "generations"},
            f, indent=2,
        )
    save_generations(
        eval_prompts, results["generations"], eval_refs,
        filepath=str(save_dir / "test_generations.json"),
        rouge_scores=results.get("per_sample_rouge_l_f"),
    )
    return results


# ------------------------------------------------------------------
# Main run function
# ------------------------------------------------------------------

def run_single(
    method: str,
    model_family: str,
    seed: int,
    device: str,
    data_source: str,
    max_train_samples: int | None,
    output_dir: str,
    *,
    epochs: int = 3,
    batch_size: int = 8,
    gradient_accumulation: int = 4,
    lr: float = 2e-5,
    max_seq_len: int = 512,
    fp16: bool = True,
    temperature: float = 2.0,
    # Adaptive method options
    klr_granularity: str = "token",
    klr_beta: float = 0.99,
    klr_fixed_alpha: float | None = None,
    akl_mu: float = 0.5,
    # RCID options
    lambda_rcid: float = 0.1,
    rcid_every_n_steps: int = 5,
    contrastive_pairs_path: str | None = None,
    contrastive_task_types: list[str] | None = None,
    top_k_checkpoints: int = 10,
    # SaGD options
    teacher_saliency_path: str | None = None,
    sagd_every_n_steps: int = 1,
    sagd_tau_w: float = 1.0,
    saliency_temperature: float = 2.0,
    # Eval
    skip_eval: bool = False,
) -> dict[str, Any]:
    """Full pipeline for one run: load -> align -> train -> eval -> save."""
    set_all_seeds(seed)
    mcfg = MODEL_CONFIGS[model_family]

    run_dir = Path(output_dir) / model_family / method / f"seed_{seed}"
    run_dir.mkdir(parents=True, exist_ok=True)

    # ── Save config.json ────────────────────────────────────────────
    run_config: dict[str, Any] = {
        "method": method, "model_family": model_family,
        "teacher": mcfg["teacher"], "student": mcfg["student"],
        "seed": seed, "data_source": data_source,
        "max_train_samples": max_train_samples,
        "epochs": epochs, "batch_size": batch_size,
        "gradient_accumulation": gradient_accumulation,
        "effective_batch_size": batch_size * gradient_accumulation,
        "lr": lr, "temperature": temperature, "max_seq_len": max_seq_len,
        "fp16": fp16, "device": device,
        "klr_granularity": klr_granularity, "klr_beta": klr_beta,
        "klr_fixed_alpha": klr_fixed_alpha, "akl_mu": akl_mu,
        "lambda_rcid": lambda_rcid, "rcid_every_n_steps": rcid_every_n_steps,
        "contrastive_pairs_path": contrastive_pairs_path,
        "teacher_saliency_path": teacher_saliency_path,
        "sagd_every_n_steps": sagd_every_n_steps,
        "sagd_tau_w": sagd_tau_w,
    }
    with open(run_dir / "config.json", "w", encoding="utf-8") as f:
        json.dump(run_config, f, indent=2)

    # ── Load models ─────────────────────────────────────────────────
    logger.info("Loading teacher: %s", mcfg["teacher"])
    teacher, t_adp, tokenizer = load_teacher(mcfg["teacher"], device=device)
    logger.info("Loading student: %s", mcfg["student"])
    student, s_adp, _ = load_student(mcfg["student"], device=device)

    # ── Instruction dataset ─────────────────────────────────────────
    from rcid.data.instruction_dataset import InstructionDataset

    logger.info("Loading instruction dataset: %s (max=%s)", data_source, max_train_samples)
    main_ds = InstructionDataset(
        dataset_name=data_source, tokenizer=tokenizer,
        max_seq_len=max_seq_len, max_samples=max_train_samples, split="train",
    )
    logger.info("Instruction dataset ready: %d samples", len(main_ds))

    # ── RCID alignment (only for RCID methods) ──────────────────────
    contrastive_ds = None
    rcid_loss_fn = None
    layer_mapping: dict[int, int] = {}
    checkpoints: list[tuple[int, int]] = []

    if method in _RCID_METHODS:
        assert contrastive_pairs_path is not None, (
            f"Method {method!r} requires --contrastive_pairs_path"
        )
        from rcid.data.generated_contrastive import GeneratedContrastiveDataset

        pairs_path = Path(contrastive_pairs_path)
        if pairs_path.is_dir():
            contrastive_ds = GeneratedContrastiveDataset.from_directory(
                pairs_path, tokenizer, max_seq_len=max_seq_len,
                task_types=contrastive_task_types,
            )
        else:
            contrastive_ds = GeneratedContrastiveDataset(
                pairs_path, tokenizer, max_seq_len=max_seq_len,
            )
        logger.info("Loaded %d contrastive pairs", len(contrastive_ds))

        layer_mapping, W_matrices, checkpoints = _compute_alignment(
            teacher, student, t_adp, s_adp, contrastive_ds,
            device=device, top_k=top_k_checkpoints,
        )
        rcid_loss_fn = _build_rcid_loss_fn(
            method, layer_mapping, W_matrices, checkpoints,
        )

    # ── SaGD (single data stream — no contrastive pairs needed) ─────
    if method in _SAGD_METHODS:
        assert teacher_saliency_path is not None, (
            f"Method {method!r} requires --teacher_saliency_path. "
            "Run scripts/precompute_teacher_saliency.py first."
        )
        logger.info("SaGD: teacher saliency at %s", teacher_saliency_path)

    # ── Build trainer ───────────────────────────────────────────────
    trainer_cfg: dict[str, Any] = {
        "method": method, "epochs": epochs,
        "batch_size": batch_size, "gradient_accumulation": gradient_accumulation,
        "lr": lr, "weight_decay": 0.01, "max_grad_norm": 1.0,
        "warmup_ratio": 0.03, "fp16": fp16, "temperature": temperature,
        "save_every_n_epochs": 1, "use_wandb": False,
        "log_every": 50, "jsonl_every": 100,
        "klr_granularity": klr_granularity, "klr_beta": klr_beta,
        "klr_fixed_alpha": klr_fixed_alpha, "akl_mu": akl_mu,
        "teacher_saliency_path": teacher_saliency_path,
        "sagd_every_n_steps": sagd_every_n_steps,
        "sagd_tau_w": sagd_tau_w,
        "saliency_temperature": saliency_temperature,
    }

    from rcid.distillation.scalable_trainer import ScalableDistillationTrainer

    trainer = ScalableDistillationTrainer(
        teacher=teacher, student=student,
        teacher_adapter=t_adp, student_adapter=s_adp,
        tokenizer=tokenizer, main_dataset=main_ds, config=trainer_cfg,
        contrastive_dataset=contrastive_ds,
        rcid_loss_fn=rcid_loss_fn,
        lambda_rcid=lambda_rcid,
        rcid_every_n_steps=rcid_every_n_steps,
        layer_mapping=layer_mapping,
        checkpoints=checkpoints,
    )

    # ── Train ───────────────────────────────────────────────────────
    logger.info("Training: method=%s  epochs=%d  batch=%dx%d  lr=%.2e",
                method, epochs, batch_size, gradient_accumulation, lr)
    t0 = time.time()
    history = trainer.train(save_dir=str(run_dir))
    train_secs = time.time() - t0
    logger.info("Training complete in %.1f s", train_secs)

    eval_kl = trainer.evaluate()

    final_ckpt = run_dir / "student_final.pt"
    torch.save(student.state_dict(), final_ckpt)
    logger.info("Final checkpoint: %s", final_ckpt)

    # ── ROUGE-L evaluation ──────────────────────────────────────────
    rouge_results: dict[str, Any] = {}
    if not skip_eval:
        try:
            rouge_results = _evaluate_rouge(student, tokenizer, run_dir, device)
        except Exception as e:
            logger.error("ROUGE-L evaluation failed: %s", e)
            rouge_results = {"error": str(e)}

    # ── Save training_log.json ──────────────────────────────────────
    training_log: dict[str, Any] = {
        "method": method, "model_family": model_family, "seed": seed,
        "train_time_sec": round(train_secs, 1),
        "n_main_samples": len(main_ds),
        "n_contrastive_pairs": len(contrastive_ds) if contrastive_ds else 0,
        "history": history,
        "eval_kl_loss": eval_kl.get("kl_loss"),
        "final_loss": history["loss"][-1] if history.get("loss") else None,
        "rouge_l_f": rouge_results.get("rouge_l_f"),
        "checkpoint_path": str(final_ckpt),
    }
    with open(run_dir / "training_log.json", "w", encoding="utf-8") as f:
        json.dump(training_log, f, indent=2)
    logger.info("Training log: %s", run_dir / "training_log.json")

    return training_log


# ------------------------------------------------------------------
# CLI
# ------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(
        description="Large-scale distillation with adaptive KL + RCID methods",
    )
    ap.add_argument("--model_family", choices=["qwen3", "llama3"], default="qwen3")
    ap.add_argument("--method", choices=METHODS, default="standard_kd",
                    help="Distillation method")
    ap.add_argument("--data_source", default=DEFAULT_DATASET,
                    help="HuggingFace dataset for instruction KL (default: Dolly-15K)")
    ap.add_argument("--max_train_samples", type=int, default=None)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--gradient_accumulation", type=int, default=4)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--max_seq_len", type=int, default=512)
    ap.add_argument("--temperature", type=float, default=2.0)
    ap.add_argument("--fp16", action="store_true", default=True)
    ap.add_argument("--no_fp16", dest="fp16", action="store_false")
    # Adaptive KL options
    ap.add_argument("--klr_granularity", default="token", choices=["token", "batch"])
    ap.add_argument("--klr_beta", type=float, default=0.99)
    ap.add_argument("--klr_fixed_alpha", type=float, default=None)
    ap.add_argument("--akl_mu", type=float, default=0.5)
    # RCID options
    ap.add_argument("--lambda_rcid", type=float, default=0.1,
                    help="Weight for RCID regulariser (default: 0.1)")
    ap.add_argument("--rcid_every_n_steps", type=int, default=5,
                    help="Compute RCID loss every N main-stream steps")
    ap.add_argument("--contrastive_pairs_path", type=str, default=None,
                    help="Path to contrastive pairs JSON file or directory")
    ap.add_argument("--contrastive_task_types", type=str, nargs="*", default=None,
                    help="Task types to load (e.g. entity_swap number_perturb)")
    ap.add_argument("--top_k_checkpoints", type=int, default=10,
                    help="Number of causal checkpoints to select")
    # SaGD options
    ap.add_argument("--teacher_saliency_path", type=str, default=None,
                    help="Path to precomputed teacher saliency .pt cache")
    ap.add_argument("--sagd_every_n_steps", type=int, default=1,
                    help="Apply SaGD reweighting every N steps (default: every step)")
    ap.add_argument("--sagd_tau_w", type=float, default=1.0,
                    help="Temperature for SaGD weight softmax")
    ap.add_argument("--saliency_temperature", type=float, default=2.0,
                    help="Temperature for saliency-to-distribution conversion")
    # General
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--output_dir", default="outputs/large_scale")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--skip_eval", action="store_true", help="Skip ROUGE-L eval")
    args = ap.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    logger.info(
        "=== Large-scale distill: method=%s  family=%s  seed=%d  data=%s ===",
        args.method, args.model_family, args.seed, args.data_source,
    )

    result = run_single(
        method=args.method, model_family=args.model_family,
        seed=args.seed, device=args.device,
        data_source=args.data_source,
        max_train_samples=args.max_train_samples,
        output_dir=args.output_dir,
        epochs=args.epochs, batch_size=args.batch_size,
        gradient_accumulation=args.gradient_accumulation,
        lr=args.lr, max_seq_len=args.max_seq_len,
        fp16=args.fp16, temperature=args.temperature,
        klr_granularity=args.klr_granularity,
        klr_beta=args.klr_beta, klr_fixed_alpha=args.klr_fixed_alpha,
        akl_mu=args.akl_mu,
        lambda_rcid=args.lambda_rcid,
        rcid_every_n_steps=args.rcid_every_n_steps,
        contrastive_pairs_path=args.contrastive_pairs_path,
        contrastive_task_types=args.contrastive_task_types,
        top_k_checkpoints=args.top_k_checkpoints,
        teacher_saliency_path=args.teacher_saliency_path,
        sagd_every_n_steps=args.sagd_every_n_steps,
        sagd_tau_w=args.sagd_tau_w,
        saliency_temperature=args.saliency_temperature,
        skip_eval=args.skip_eval,
    )

    logger.info(
        "Done. train_time=%.1fs  final_loss=%s  eval_kl=%s  rouge_l=%.4f",
        result["train_time_sec"], result["final_loss"],
        result["eval_kl_loss"], result.get("rouge_l_f") or 0.0,
    )


if __name__ == "__main__":
    main()
