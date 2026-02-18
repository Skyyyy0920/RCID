"""Experiment 3: Mechanistic consistency vs OOD robustness correlation.

Collects (causal_consistency, ood_degradation) pairs from all trained students
in exp1/exp2, then computes Pearson correlation to test whether preserving
teacher mechanisms leads to more robust OOD generalization.

Usage:
    python scripts/run_exp3.py --model_family qwen3 --device cuda:0
    python scripts/run_exp3.py --model_family llama3 --device cuda:0
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import torch

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from rcid import set_all_seeds
from rcid.data.factual_probing import FactualProbingDataset
from rcid.data.ioi import IOIDataset, build_single_token_names
from rcid.data.winogrande import WinoGrandeDataset
from rcid.eval.ood_robustness import evaluate_ood_robustness
from rcid.eval.task_accuracy import evaluate_task_accuracy
from rcid.models.adapter import get_adapter
from rcid.models.student import load_student_from_checkpoint

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
)
logger = logging.getLogger(__name__)

STUDENT_NAMES: dict[str, str] = {
    "qwen3": "Qwen/Qwen3-0.6B",
    "llama3": "meta-llama/Llama-3.2-1B",
}


def _pearson_r(x: torch.Tensor, y: torch.Tensor) -> float:
    """Compute Pearson correlation using pure torch."""
    x_c = x - x.mean()
    y_c = y - y.mean()
    num = (x_c * y_c).sum()
    den = (x_c.pow(2).sum() * y_c.pow(2).sum()).sqrt().clamp(min=1e-10)
    return (num / den).item()


def _collect_result_files(
    results_dir: Path, model_family: str
) -> list[dict]:
    """Scan exp1/ and exp2/ for result JSON files matching model_family."""
    entries: list[dict] = []
    for exp_dir in ["exp1", "exp2"]:
        search_root = results_dir / exp_dir / model_family
        if not search_root.exists():
            logger.warning("Directory not found: %s", search_root)
            continue
        for json_path in search_root.glob("*.json"):
            try:
                data = json.loads(json_path.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError) as exc:
                logger.warning("Skipping %s: %s", json_path, exc)
                continue
            if data.get("model_family") != model_family:
                continue
            if "checkpoint_path" not in data or "causal_consistency" not in data:
                logger.warning("Missing fields in %s, skipping", json_path)
                continue
            entries.append(data)
    logger.info("Found %d result entries for %s", len(entries), model_family)
    return entries


def _build_ood_dataset(
    task: str, tokenizer, name_pool: list[str]
):
    """Build the OOD dataset variant for a given task name."""
    if task == "ioi":
        return IOIDataset.build_ood_dataset(
            tokenizer=tokenizer,
            name_pool=name_pool,
            in_distribution_names=name_pool[:10],
            n_samples=50,
        ).dataset
    elif task == "factual_probing":
        return FactualProbingDataset.build_ood_dataset(
            tokenizer=tokenizer,
        ).dataset
    elif task == "winogrande":
        return WinoGrandeDataset.build_ood_dataset(
            tokenizer=tokenizer,
        ).dataset
    else:
        raise ValueError(f"Unknown task: {task}")


def _build_id_dataset(
    task: str, tokenizer, name_pool: list[str]
):
    """Build the in-distribution dataset for a given task name."""
    if task == "ioi":
        return IOIDataset(tokenizer=tokenizer, n_samples=100,
                          name_pool=name_pool, seed=42).dataset
    elif task == "factual_probing":
        return FactualProbingDataset(tokenizer=tokenizer, seed=42).dataset
    elif task == "winogrande":
        return WinoGrandeDataset(tokenizer=tokenizer, seed=42).dataset
    else:
        raise ValueError(f"Unknown task: {task}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Exp3: CC vs OOD robustness correlation"
    )
    parser.add_argument(
        "--model_family", choices=["qwen3", "llama3"], required=True
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--results_dir", default="outputs/results",
        help="Root results directory containing exp1/ and exp2/",
    )
    parser.add_argument(
        "--output_dir", default=None,
        help="Output directory (default: outputs/results/exp3/{model_family})",
    )
    args = parser.parse_args()

    set_all_seeds(42)
    results_dir = Path(args.results_dir)
    output_dir = (
        Path(args.output_dir)
        if args.output_dir
        else results_dir / "exp3" / args.model_family
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    student_name = STUDENT_NAMES[args.model_family]

    # --- Collect previous experiment results ---
    entries = _collect_result_files(results_dir, args.model_family)
    if not entries:
        logger.error("No result files found. Run exp1/exp2 first.")
        sys.exit(1)

    # --- Preload tokenizer once to build datasets ---
    logger.info("Loading tokenizer from %s ...", student_name)
    _, adapter, tokenizer = load_student_from_checkpoint(
        entries[0]["checkpoint_path"], student_name,
        device="cpu", dtype=torch.float32,
    )
    name_pool = build_single_token_names(tokenizer)

    # Cache datasets to avoid rebuilding per student
    id_datasets: dict[str, object] = {}
    ood_datasets: dict[str, object] = {}

    # --- Evaluate each student ---
    scatter_data: list[dict] = []
    for idx, entry in enumerate(entries):
        method = entry["method"]
        task = entry["task"]
        seed = entry.get("seed", 0)
        cc = entry["causal_consistency"]["mean_correlation"]
        ckpt = entry["checkpoint_path"]

        logger.info(
            "[%d/%d] method=%s task=%s seed=%d cc=%.4f",
            idx + 1, len(entries), method, task, seed, cc,
        )

        # Build datasets (cached)
        if task not in id_datasets:
            id_datasets[task] = _build_id_dataset(task, tokenizer, name_pool)
        if task not in ood_datasets:
            ood_datasets[task] = _build_ood_dataset(task, tokenizer, name_pool)

        # Load student
        model, adapter, _ = load_student_from_checkpoint(
            ckpt, student_name, device=args.device, dtype=torch.float16,
        )
        model.eval()

        # ID accuracy
        id_result = evaluate_task_accuracy(model, adapter, id_datasets[task])
        id_acc = id_result["accuracy"]

        # OOD robustness
        ood_result = evaluate_ood_robustness(
            model, adapter, ood_datasets[task], id_acc
        )
        ood_deg = ood_result["relative_degradation"]

        scatter_data.append({
            "method": method,
            "task": task,
            "seed": seed,
            "causal_consistency": cc,
            "id_accuracy": id_acc,
            "ood_accuracy": ood_result["ood_accuracy"],
            "ood_degradation": ood_deg,
        })
        logger.info(
            "  id_acc=%.4f ood_acc=%.4f degradation=%.4f",
            id_acc, ood_result["ood_accuracy"], ood_deg,
        )

        # Free GPU memory
        del model
        torch.cuda.empty_cache()

    # --- Compute overall correlation ---
    cc_vals = torch.tensor([d["causal_consistency"] for d in scatter_data])
    deg_vals = torch.tensor([d["ood_degradation"] for d in scatter_data])
    n_points = len(scatter_data)

    if n_points < 3:
        logger.warning("Only %d data points; correlation unreliable.", n_points)
        pearson = float("nan")
    else:
        pearson = _pearson_r(cc_vals, -deg_vals)  # higher CC → lower degradation

    logger.info("=" * 60)
    logger.info("Pearson(CC, -OOD_degradation) = %.4f  (n=%d)", pearson, n_points)
    logger.info("=" * 60)

    # --- Per-task breakdown ---
    tasks_seen = sorted({d["task"] for d in scatter_data})
    per_task: dict[str, dict] = {}
    for task in tasks_seen:
        subset = [d for d in scatter_data if d["task"] == task]
        if len(subset) < 3:
            per_task[task] = {"pearson": float("nan"), "n": len(subset)}
            continue
        t_cc = torch.tensor([d["causal_consistency"] for d in subset])
        t_deg = torch.tensor([d["ood_degradation"] for d in subset])
        per_task[task] = {
            "pearson": _pearson_r(t_cc, -t_deg),
            "n": len(subset),
        }
        logger.info("  %s: r=%.4f (n=%d)", task, per_task[task]["pearson"],
                     per_task[task]["n"])

    # --- Save ---
    output = {
        "model_family": args.model_family,
        "n_students": n_points,
        "overall_pearson": pearson,
        "per_task": per_task,
        "scatter_data": scatter_data,
    }
    out_path = output_dir / "correlation_analysis.json"
    out_path.write_text(json.dumps(output, indent=2), encoding="utf-8")
    logger.info("Results saved to %s", out_path)


if __name__ == "__main__":
    main()
