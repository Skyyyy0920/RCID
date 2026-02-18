"""Experiment 4: Information Purity — d^T (contrastive diff) vs h^T (raw representation).

At UNMODIFIED token positions, compare selectivity of d^T = h_clean - h_corrupt
against h^T = h_clean. Higher selectivity(d^T) means contrastive differences
filter out task-irrelevant information more effectively.
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import torch

from rcid import set_all_seeds
from rcid.models.teacher import load_teacher
from rcid.data.ioi import IOIDataset
from rcid.data.factual_probing import FactualProbingDataset
from rcid.data.winogrande import WinoGrandeDataset
from rcid.circuit.patching import extract_residual_at_layers, extract_contrastive_differences
from rcid.eval.information_purity import evaluate_information_purity

logger = logging.getLogger(__name__)

MODEL_NAMES: dict[str, str] = {
    "qwen3": "Qwen/Qwen3-8B",
    "llama3": "meta-llama/Llama-3.1-8B",
}


def build_dataset(task: str, tokenizer, n_samples: int, seed: int = 42):
    """Return (ContrastiveDataset, control_labels (N,))."""
    if task == "ioi":
        builder = IOIDataset(tokenizer, n_samples=n_samples, seed=seed)
        ds, n_t = builder.dataset, len(builder.templates)
        ctrl = torch.tensor([i % n_t for i in range(len(ds))], dtype=torch.long)
    elif task == "factual":
        builder = FactualProbingDataset(tokenizer, seed=seed)
        ds, n_t = builder.dataset, len(builder.templates)
        ctrl = torch.tensor([i % n_t for i in range(len(ds))], dtype=torch.long)
    elif task == "winogrande":
        builder = WinoGrandeDataset(tokenizer, seed=seed)
        ds = builder.dataset
        ctrl = torch.arange(len(ds), dtype=torch.long) % max(len(ds) // 3, 2)
    else:
        raise ValueError(f"Unknown task: {task}")
    return ds, ctrl


def select_layers(n_layers: int, stride: int = 4) -> list[int]:
    """Every `stride`-th layer, always including the last."""
    layers = list(range(0, n_layers, stride))
    if (n_layers - 1) not in layers:
        layers.append(n_layers - 1)
    return layers


def gather_at_positions(
    acts: dict[int, torch.Tensor],  # {layer: (N, seq, d)}
    positions: torch.Tensor,         # (N,)
) -> dict[int, torch.Tensor]:       # {layer: (N, d)}
    """Index into each layer's activation at per-sample token positions."""
    out: dict[int, torch.Tensor] = {}
    for l, t in acts.items():
        idx = positions.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, t.shape[-1])  # (N,1,d)
        out[l] = t.gather(1, idx).squeeze(1)  # (N, d)
    return out


def run_purity_experiment(args: argparse.Namespace) -> dict:
    """Core logic: extract h^T and d^T, probe each layer."""
    set_all_seeds(42)
    model_name = MODEL_NAMES[args.model_family]
    logger.info("Loading teacher: %s", model_name)
    teacher, adapter, tokenizer = load_teacher(model_name, device=args.device)

    logger.info("Building dataset: task=%s n_samples=%d", args.task, args.n_samples)
    dataset, control_labels = build_dataset(args.task, tokenizer, args.n_samples)
    n = len(dataset)
    logger.info("Dataset size: %d", n)

    # Identify unmodified positions
    unmod_names = dataset.get_unmodified_positions()
    if unmod_names:
        pos_name = unmod_names[0]
        positions = dataset.key_positions[pos_name]  # (N,)
        logger.info("Unmodified position: '%s' (sample-0 pos=%d)", pos_name, positions[0].item())
    else:
        logger.warning("No unmodified key positions; falling back to position 0")
        pos_name = "fallback_0"
        positions = torch.zeros(n, dtype=torch.long)

    # Binary task labels
    task_labels = (dataset.correct_token_id > dataset.wrong_token_id).long()  # (N,)
    logger.info("Task label balance: %d / %d positive", task_labels.sum().item(), n)

    # Select layers
    n_layers = adapter.get_num_layers(teacher)
    layers = select_layers(n_layers, stride=4)
    logger.info("Layers to analyse (%d): %s", len(layers), layers)

    # Extract representations in batches
    clean_ids = dataset.clean_ids.to(args.device)      # (N, seq)
    corrupt_ids = dataset.corrupt_ids.to(args.device)   # (N, seq)
    h_acc: dict[int, list[torch.Tensor]] = {l: [] for l in layers}
    d_acc: dict[int, list[torch.Tensor]] = {l: [] for l in layers}
    bs = 32

    for s in range(0, n, bs):
        e = min(s + bs, n)
        pos_b = positions[s:e].to(args.device)
        h_clean = extract_residual_at_layers(teacher, adapter, clean_ids[s:e], layers)
        d_batch = extract_contrastive_differences(
            teacher, adapter, clean_ids[s:e], corrupt_ids[s:e], layers,
        )
        h_pos = gather_at_positions(h_clean, pos_b)
        d_pos = gather_at_positions(d_batch, pos_b)
        for l in layers:
            h_acc[l].append(h_pos[l].cpu())
            d_acc[l].append(d_pos[l].cpu())
        logger.info("  Batch %d-%d / %d", s, e, n)

    h_T = {l: torch.cat(h_acc[l]) for l in layers}  # {layer: (N, d_model)}
    d_T = {l: torch.cat(d_acc[l]) for l in layers}   # {layer: (N, d_model)}

    # Probe each layer
    results: dict[str, object] = {
        "meta": {
            "model_family": args.model_family,
            "model_name": model_name,
            "task": args.task,
            "n_samples": n,
            "layers_analysed": layers,
            "unmodified_position": pos_name,
        },
        "layers": {},
    }

    for l in layers:
        logger.info("Probing layer %d ...", l)
        pur_h = evaluate_information_purity(h_T[l], task_labels, control_labels)
        pur_d = evaluate_information_purity(d_T[l], task_labels, control_labels)
        gain = pur_d["selectivity"] - pur_h["selectivity"]
        results["layers"][str(l)] = {  # type: ignore[index]
            "h_T": pur_h, "d_T": pur_d, "selectivity_gain": gain,
        }
        logger.info(
            "  Layer %2d | h sel=%.4f | d sel=%.4f | gain=%+.4f",
            l, pur_h["selectivity"], pur_d["selectivity"], gain,
        )

    return results


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Exp4: information purity d^T vs h^T")
    p.add_argument("--model_family", type=str, default="qwen3", choices=["qwen3", "llama3"])
    p.add_argument("--task", type=str, default="ioi", choices=["ioi", "factual", "winogrande"])
    p.add_argument("--device", type=str, default="cuda:0")
    p.add_argument("--output_dir", type=str, default=None)
    p.add_argument("--n_samples", type=int, default=500)
    return p.parse_args()


def main() -> None:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    args = parse_args()
    if args.output_dir is None:
        args.output_dir = str(Path("outputs/results/exp4") / args.model_family)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    results = run_purity_experiment(args)

    out_path = out_dir / f"{args.task}_purity.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    logger.info("Results saved to %s", out_path)


if __name__ == "__main__":
    main()
