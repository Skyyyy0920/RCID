"""Evaluate a distilled student on standard LLM benchmarks via lm-evaluation-harness.

Requires: ``pip install lm-eval>=0.4``

If a ``--model_path`` checkpoint (.pt state_dict) is provided, the weights are
merged into the base HF model specified by ``--model_name``, saved to a temp
directory, and the temp directory is passed to lm-eval as ``pretrained=``.

Usage::

    # Evaluate a distilled checkpoint
    python scripts/eval_benchmarks.py \
        --model_path outputs/large_scale/qwen3/standard_kd_rcid/seed_42/student_final.pt \
        --model_name Qwen/Qwen3-0.6B \
        --benchmarks mmlu,gsm8k,arc_challenge \
        --device cuda:0

    # Evaluate base model (no checkpoint)
    python scripts/eval_benchmarks.py \
        --model_name Qwen/Qwen3-0.6B \
        --benchmarks mmlu,gsm8k
"""
from __future__ import annotations

import argparse
import json
import logging
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

import torch

# Ensure src/ is on the path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

logger = logging.getLogger(__name__)

DEFAULT_BENCHMARKS = [
    "mmlu", "gsm8k", "arc_challenge", "hellaswag", "winogrande", "truthfulqa_mc2",
]

# Standard few-shot counts per benchmark
STANDARD_NUM_FEWSHOT: dict[str, int] = {
    "mmlu": 5,
    "gsm8k": 8,
    "arc_challenge": 25,
    "arc_easy": 25,
    "hellaswag": 10,
    "winogrande": 5,
    "truthfulqa_mc2": 0,
    "truthfulqa": 0,
    "piqa": 0,
    "boolq": 0,
}


# ==================================================================
# Core: lm-eval-harness subprocess
# ==================================================================

def _prepare_model_dir(
    model_name: str, model_path: str | None,
) -> tuple[str, str | None]:
    """Prepare the model directory for lm-eval.

    If *model_path* is a ``.pt`` state_dict, loads it into the base model,
    saves via ``save_pretrained`` to a temp directory, and returns
    ``(temp_dir, temp_dir)``.

    If *model_path* is ``None`` (base model), returns ``(model_name, None)``.

    Returns:
        (actual_model_path, tmp_dir_to_cleanup)
        ``tmp_dir_to_cleanup`` is ``None`` when no temp dir was created.
    """
    if model_path is None:
        return model_name, None

    ckpt = Path(model_path)
    if not ckpt.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt}")

    from transformers import AutoModelForCausalLM, AutoTokenizer

    logger.info("Merging checkpoint into base model for lm-eval...")
    logger.info("  Base model : %s", model_name)
    logger.info("  Checkpoint : %s", model_path)

    # Load base model on CPU, apply state_dict, save to temp dir
    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.float16, trust_remote_code=True,
    )
    state_dict = torch.load(ckpt, map_location="cpu", weights_only=True)
    model.load_state_dict(state_dict)
    del state_dict

    tmp_dir = tempfile.mkdtemp(prefix="eval_merged_model_")
    logger.info("  Saving merged model to: %s", tmp_dir)
    model.save_pretrained(tmp_dir)
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # Also save tokenizer so lm-eval can load it from the same dir
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    tokenizer.save_pretrained(tmp_dir)

    logger.info("  Merged model ready.")
    return tmp_dir, tmp_dir


def evaluate_via_lm_eval_cli(
    model_name: str,
    model_path: str | None,
    benchmarks: list[str],
    num_fewshot: dict[str, int],
    batch_size: int,
    device: str,
) -> dict[str, dict[str, Any]]:
    """Run lm-eval-harness CLI and parse results.

    If *model_path* is provided the weights are merged into the HF model
    specified by *model_name* and saved to a temp directory for lm-eval.
    """
    # Prepare the model (merge checkpoint if needed)
    actual_model_path, tmp_dir = _prepare_model_dir(model_name, model_path)

    results: dict[str, dict[str, Any]] = {}

    try:
        for bench in benchmarks:
            n_shot = num_fewshot.get(bench, 0)
            logger.info("lm-eval: %s (%d-shot, batch=%d)...", bench, n_shot, batch_size)

            with tempfile.TemporaryDirectory() as tmpdir:
                model_args = (
                    f"pretrained={actual_model_path},"
                    f"dtype=float16,"
                    f"trust_remote_code=True"
                )

                cmd = [
                    "lm_eval",
                    "--model", "hf",
                    "--model_args", model_args,
                    "--tasks", bench,
                    "--num_fewshot", str(n_shot),
                    "--batch_size", str(batch_size),
                    "--device", device,
                    "--output_path", tmpdir,
                    "--log_samples",
                ]

                t0 = time.time()
                try:
                    proc = subprocess.run(
                        cmd, capture_output=True, text=True, timeout=3600,
                    )
                    elapsed = time.time() - t0

                    if proc.returncode != 0:
                        logger.error(
                            "lm-eval failed for %s:\n%s", bench, proc.stderr[-500:],
                        )
                        results[bench] = {"error": proc.stderr[-300:]}
                        continue

                    # Parse output JSON
                    bench_result = _parse_lm_eval_output(tmpdir, bench)
                    bench_result["time_sec"] = round(elapsed, 1)
                    bench_result["num_fewshot"] = n_shot
                    results[bench] = bench_result

                    score = bench_result.get("accuracy") or bench_result.get("exact_match")
                    logger.info("  %s = %.4f (%.1f s)", bench, score or 0.0, elapsed)

                except subprocess.TimeoutExpired:
                    logger.error("  %s: TIMEOUT", bench)
                    results[bench] = {"error": "timeout"}
                except Exception as exc:
                    logger.error("  %s: %s", bench, exc)
                    results[bench] = {"error": str(exc)}
    finally:
        # Clean up temp model directory
        if tmp_dir is not None:
            logger.info("Cleaning up temp model dir: %s", tmp_dir)
            shutil.rmtree(tmp_dir, ignore_errors=True)

    return results


def _parse_lm_eval_output(output_dir: str, task_name: str) -> dict[str, Any]:
    """Parse lm-eval-harness JSON output directory."""
    out_path = Path(output_dir)

    # lm-eval writes results to a nested directory structure
    # Try to find the results JSON file
    candidates = list(out_path.rglob("results.json"))
    if not candidates:
        # Fallback: any JSON file
        candidates = list(out_path.rglob("*.json"))

    for json_file in candidates:
        try:
            with open(json_file, "r", encoding="utf-8") as f:
                data = json.load(f)
            task_results = data.get("results", {}).get(task_name, {})
            if task_results:
                return _extract_metrics(task_results, task_name)
        except (json.JSONDecodeError, KeyError):
            continue

    return {"error": "could not parse lm-eval output"}


def _extract_metrics(raw: dict[str, Any], task_name: str) -> dict[str, Any]:
    """Extract the primary metric from lm-eval raw results.

    lm-eval keys follow the pattern ``<metric>,<filter>`` where common
    metrics are ``acc_norm``, ``acc``, ``exact_match`` and common filters
    are ``none``, ``flexible-extract``, ``strict-match``, etc.

    Strategy:
      1. Scan *all* keys in ``raw`` and bucket them by metric prefix.
      2. Pick the best metric by priority: acc_norm > acc > exact_match.
      3. Within a metric, pick the best filter by priority:
         ``none`` > ``flexible-extract`` > (first remaining).
    """
    result: dict[str, Any] = {"num_samples": raw.get("alias", task_name)}

    # --- Step 1: bucket keys by (metric_prefix, filter) ---------------
    # metric_values: { metric_prefix: { filter: value } }
    metric_values: dict[str, dict[str, float]] = {}
    stderr_values: dict[str, dict[str, float]] = {}

    for key, value in raw.items():
        if not isinstance(value, (int, float)):
            continue
        # Parse "metric,filter" or bare "metric"
        if "," in key:
            metric_prefix, filt = key.split(",", 1)
        else:
            metric_prefix, filt = key, "none"

        if metric_prefix.endswith("_stderr"):
            base = metric_prefix.removesuffix("_stderr")
            stderr_values.setdefault(base, {})[filt] = value
        else:
            metric_values.setdefault(metric_prefix, {})[filt] = value

    # --- Step 2: pick the best metric by priority ---------------------
    METRIC_PRIORITY = ["acc_norm", "acc", "exact_match"]
    FILTER_PRIORITY = ["none", "flexible-extract"]

    chosen_metric: str | None = None
    chosen_filter: str | None = None
    chosen_value: float | None = None

    for mp in METRIC_PRIORITY:
        if mp not in metric_values:
            continue
        filters = metric_values[mp]
        # Pick filter by priority, fallback to first available
        for fp in FILTER_PRIORITY:
            if fp in filters:
                chosen_metric, chosen_filter, chosen_value = mp, fp, filters[fp]
                break
        if chosen_metric is None:
            # None of the priority filters present — take first available
            first_filt = next(iter(filters))
            chosen_metric, chosen_filter = mp, first_filt
            chosen_value = filters[first_filt]
        break  # found a metric in priority order, stop

    if chosen_metric is not None and chosen_value is not None:
        out_name = "accuracy" if "acc" in chosen_metric else "exact_match"
        result[out_name] = chosen_value
        result["raw_metric_key"] = f"{chosen_metric},{chosen_filter}"
        logger.debug(
            "  %s: picked %s,%s = %.4f",
            task_name, chosen_metric, chosen_filter, chosen_value,
        )

        # Matching stderr
        if chosen_metric in stderr_values:
            stderr_filt = stderr_values[chosen_metric]
            if chosen_filter in stderr_filt:
                result["stderr"] = stderr_filt[chosen_filter]

    # --- Step 3: store all raw metrics for transparency ---------------
    result["all_metrics"] = {
        k: v for k, v in raw.items()
        if isinstance(v, (int, float)) and not k.startswith("alias")
    }

    return result


# ==================================================================
# Summary + output
# ==================================================================

def print_summary(
    results: dict[str, dict[str, Any]], model_path: str | None,
) -> None:
    """Print a compact results table to the logger."""
    label = Path(model_path).stem if model_path else "base"
    print()  # noqa: T201
    print("=" * 62)  # noqa: T201
    print(f"  BENCHMARK RESULTS — {label}")  # noqa: T201
    print("-" * 62)  # noqa: T201
    print(f"  {'Benchmark':<22s}  {'Metric':<16s}  {'Score':>8s}")  # noqa: T201
    print("-" * 62)  # noqa: T201
    for bench, res in results.items():
        if "error" in res:
            print(f"  {bench:<22s}  {'—':<16s}  {'FAILED':>8s}")  # noqa: T201
            continue
        if "accuracy" in res:
            metric, score = "accuracy", res["accuracy"]
        elif "exact_match" in res:
            metric, score = "exact_match", res["exact_match"]
        else:
            metric, score = "?", 0.0
        print(f"  {bench:<22s}  {metric:<16s}  {score:>8.4f}")  # noqa: T201
    print("=" * 62)  # noqa: T201
    print()  # noqa: T201


def save_results(
    results: dict[str, dict[str, Any]],
    model_path: str | None,
    model_name: str,
    output_path: str,
) -> None:
    """Save results JSON."""
    output = {
        "model_path": model_path,
        "model_name": model_name,
        "results": results,
    }
    out_file = Path(output_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)
    logger.info("Results saved: %s", out_file)


# ==================================================================
# CLI
# ==================================================================

def main() -> None:
    ap = argparse.ArgumentParser(
        description="Evaluate distilled student on LLM benchmarks (requires lm-eval>=0.4)",
    )
    ap.add_argument(
        "--model_path", default=None,
        help="Path to student checkpoint (.pt state_dict). Omit to evaluate base model.",
    )
    ap.add_argument(
        "--model_name", required=True,
        help="HuggingFace model name for config/tokenizer (e.g. Qwen/Qwen3-0.6B)",
    )
    ap.add_argument(
        "--benchmarks",
        default="mmlu,gsm8k,arc_challenge,hellaswag,winogrande,truthfulqa_mc2",
        help="Comma-separated benchmark list",
    )
    ap.add_argument(
        "--output_path", default=None,
        help="Output JSON path (default: alongside checkpoint)",
    )
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument(
        "--num_fewshot", type=int, default=None,
        help="Override few-shot count for ALL benchmarks (default: per-benchmark standard)",
    )
    args = ap.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    benchmarks = [b.strip() for b in args.benchmarks.split(",") if b.strip()]
    logger.info("Benchmarks: %s", benchmarks)

    # Build per-benchmark num_fewshot
    num_fewshot = dict(STANDARD_NUM_FEWSHOT)
    if args.num_fewshot is not None:
        num_fewshot = {b: args.num_fewshot for b in benchmarks}

    # Determine output path
    output_path = args.output_path
    if output_path is None:
        if args.model_path:
            output_path = str(Path(args.model_path).parent / "benchmark_results.json")
        else:
            output_path = "benchmark_results.json"

    logger.info("Model: %s", args.model_name)
    logger.info("Checkpoint: %s", args.model_path or "(base)")

    # Run lm-eval-harness
    results = evaluate_via_lm_eval_cli(
        model_name=args.model_name,
        model_path=args.model_path,
        benchmarks=benchmarks,
        num_fewshot=num_fewshot,
        batch_size=args.batch_size,
        device=args.device,
    )

    # Print summary and save
    print_summary(results, args.model_path)
    save_results(results, args.model_path, args.model_name, output_path)


if __name__ == "__main__":
    main()
