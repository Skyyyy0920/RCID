"""Evaluate a distilled student on standard LLM benchmarks.

Two evaluation modes:

  **Mode 1 (recommended)**: lm-evaluation-harness via subprocess.
  Calls ``lm_eval --model hf ...`` and parses the JSON output.
  Requires: ``pip install lm-eval>=0.4``

  **Mode 2 (fallback)**: simple custom few-shot evaluation.
  If lm-eval is not installed, runs lightweight built-in evaluators
  for MMLU (5-shot MC), GSM8K (8-shot exact-match), ARC-Challenge
  (25-shot MC).

Usage::

    # With lm-eval-harness (recommended)
    python scripts/eval_benchmarks.py \
        --model_path outputs/large_scale/qwen3/standard_kd_rcid/seed_42/student_final.pt \
        --model_name Qwen/Qwen3-0.6B \
        --benchmarks mmlu,gsm8k,arc_challenge \
        --device cuda:0

    # Evaluate base model (no checkpoint)
    python scripts/eval_benchmarks.py \
        --model_name Qwen/Qwen3-0.6B \
        --benchmarks mmlu,gsm8k

    # Fallback mode (no lm-eval installed)
    python scripts/eval_benchmarks.py \
        --model_path outputs/.../student_final.pt \
        --model_name Qwen/Qwen3-0.6B \
        --benchmarks mmlu,gsm8k,arc_challenge \
        --fallback
"""
from __future__ import annotations

import argparse
import json
import logging
import re
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

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
# Model loading
# ==================================================================

def _load_model(
    model_name: str,
    model_path: str | None = None,
    device: str = "cuda:0",
) -> tuple[torch.nn.Module, Any]:
    """Load HF model + tokenizer, optionally overriding weights from checkpoint."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    logger.info("Loading base model: %s", model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.float16, trust_remote_code=True,
    )

    if model_path is not None:
        ckpt = Path(model_path)
        if not ckpt.exists():
            raise FileNotFoundError(f"Checkpoint not found: {ckpt}")
        logger.info("Loading checkpoint weights: %s", ckpt)
        state_dict = torch.load(ckpt, map_location="cpu", weights_only=True)
        model.load_state_dict(state_dict, strict=False)
        logger.info("Checkpoint loaded")

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    model = model.to(device).eval()
    return model, tokenizer


# ==================================================================
# Mode 1: lm-eval-harness subprocess
# ==================================================================

def _lm_eval_available() -> bool:
    """Check if lm-eval CLI is on PATH."""
    try:
        r = subprocess.run(
            ["lm_eval", "--help"],
            capture_output=True, timeout=15,
        )
        return r.returncode == 0
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False


def evaluate_via_lm_eval_cli(
    model_name: str,
    model_path: str | None,
    benchmarks: list[str],
    num_fewshot: dict[str, int],
    batch_size: int,
    device: str,
) -> dict[str, dict[str, Any]]:
    """Run lm-eval-harness CLI and parse results.

    If *model_path* is provided the weights are loaded into the HF model
    specified by *model_name*.
    """
    results: dict[str, dict[str, Any]] = {}

    for bench in benchmarks:
        n_shot = num_fewshot.get(bench, 0)
        logger.info("lm-eval: %s (%d-shot, batch=%d)...", bench, n_shot, batch_size)

        with tempfile.TemporaryDirectory() as tmpdir:
            # Build model_args
            model_args = f"pretrained={model_name},dtype=float16,trust_remote_code=True"
            if model_path is not None:
                model_args += f",peft_model_path={model_path}"

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
                    logger.error("lm-eval failed for %s:\n%s", bench, proc.stderr[-500:])
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
    """Extract the primary metric from lm-eval raw results."""
    result: dict[str, Any] = {"num_samples": raw.get("alias", task_name)}

    # Priority: acc_norm > acc > exact_match
    for key in ("acc_norm,none", "acc,none", "exact_match,none",
                "acc_norm", "acc", "exact_match"):
        if key in raw:
            metric_name = "accuracy" if "acc" in key else "exact_match"
            result[metric_name] = raw[key]
            break

    # Also capture stderr if available
    for key in ("acc_norm_stderr,none", "acc_stderr,none"):
        if key in raw:
            result["stderr"] = raw[key]
            break

    return result


# ==================================================================
# Mode 2: fallback custom evaluators
# ==================================================================

def _generate(
    model: torch.nn.Module,
    tokenizer: Any,
    prompt: str,
    max_new_tokens: int = 64,
    device: str = "cuda:0",
) -> str:
    """Generate text from a prompt."""
    enc = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1536)
    input_ids = enc.input_ids.to(device)

    with torch.no_grad():
        out = model.generate(
            input_ids,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
        )
    # Decode only the newly generated tokens
    new_tokens = out[0, input_ids.shape[1]:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True)


def _score_choices(
    model: torch.nn.Module,
    tokenizer: Any,
    prompt: str,
    choices: list[str],
    device: str = "cuda:0",
) -> list[float]:
    """Score each choice by the log-likelihood of the choice tokens given prompt."""
    scores: list[float] = []

    for choice in choices:
        full_text = prompt + choice
        enc = tokenizer(full_text, return_tensors="pt", truncation=True, max_length=2048)
        input_ids = enc.input_ids.to(device)  # (1, L)

        prompt_enc = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048)
        prompt_len = prompt_enc.input_ids.shape[1]

        with torch.no_grad():
            logits = model(input_ids).logits  # (1, L, V)

        # Log-prob of choice tokens: positions [prompt_len-1 .. L-2] predict [prompt_len .. L-1]
        choice_logits = logits[0, prompt_len - 1:-1, :]  # (n_choice_tokens, V)
        choice_ids = input_ids[0, prompt_len:]            # (n_choice_tokens,)

        log_probs = F.log_softmax(choice_logits.float(), dim=-1)
        token_log_probs = log_probs.gather(1, choice_ids.unsqueeze(1)).squeeze(1)
        scores.append(token_log_probs.sum().item())

    return scores


def evaluate_mmlu_fallback(
    model: torch.nn.Module,
    tokenizer: Any,
    device: str = "cuda:0",
    num_fewshot: int = 5,
    max_subjects: int = 57,
) -> dict[str, Any]:
    """Simple MMLU 5-shot MC evaluation (fallback)."""
    try:
        from datasets import load_dataset
        ds = load_dataset("cais/mmlu", "all", split="test")
    except Exception as exc:
        logger.warning("Could not load MMLU: %s", exc)
        return {"error": str(exc)}

    # Load a few validation examples for few-shot
    try:
        val_ds = load_dataset("cais/mmlu", "all", split="validation")
    except Exception:
        val_ds = None

    correct = 0
    total = 0
    labels = ["A", "B", "C", "D"]

    for sample in ds:
        question = sample["question"]
        choices = sample["choices"]
        answer_idx = sample["answer"]  # 0-3

        # Build few-shot prompt
        prompt = ""
        if val_ds is not None and num_fewshot > 0:
            # Use first N validation examples with same subject
            shot_count = 0
            for vs in val_ds:
                if shot_count >= num_fewshot:
                    break
                prompt += _format_mmlu_question(vs["question"], vs["choices"])
                prompt += f" {labels[vs['answer']]}\n\n"
                shot_count += 1

        prompt += _format_mmlu_question(question, choices)

        # Score each answer choice
        scores = _score_choices(model, tokenizer, prompt, labels, device)
        pred = scores.index(max(scores))

        if pred == answer_idx:
            correct += 1
        total += 1

    accuracy = correct / max(total, 1)
    logger.info("MMLU fallback: accuracy=%.4f (%d/%d)", accuracy, correct, total)
    return {"accuracy": accuracy, "num_samples": total, "num_fewshot": num_fewshot}


def _format_mmlu_question(question: str, choices: list[str]) -> str:
    """Format a single MMLU question."""
    labels = ["A", "B", "C", "D"]
    text = f"Question: {question}\n"
    for label, choice in zip(labels, choices):
        text += f"{label}. {choice}\n"
    text += "Answer:"
    return text


def evaluate_gsm8k_fallback(
    model: torch.nn.Module,
    tokenizer: Any,
    device: str = "cuda:0",
    num_fewshot: int = 8,
    max_samples: int = 1319,
) -> dict[str, Any]:
    """Simple GSM8K 8-shot exact-match evaluation (fallback)."""
    try:
        from datasets import load_dataset
        ds = load_dataset("openai/gsm8k", "main", split="test")
    except Exception as exc:
        logger.warning("Could not load GSM8K: %s", exc)
        return {"error": str(exc)}

    # Few-shot examples from train split
    shots: list[dict] = []
    if num_fewshot > 0:
        try:
            train_ds = load_dataset("openai/gsm8k", "main", split="train")
            shots = [train_ds[i] for i in range(num_fewshot)]
        except Exception:
            pass

    correct = 0
    total = 0

    for sample in ds:
        if total >= max_samples:
            break

        # Build prompt
        prompt = ""
        for shot in shots:
            prompt += f"Q: {shot['question']}\nA: {shot['answer']}\n\n"
        prompt += f"Q: {sample['question']}\nA:"

        # Generate answer
        response = _generate(model, tokenizer, prompt, max_new_tokens=256, device=device)

        # Extract final number from both prediction and gold
        pred_num = _extract_final_number(response)
        gold_num = _extract_final_number(sample["answer"])

        if pred_num is not None and gold_num is not None and pred_num == gold_num:
            correct += 1
        total += 1

    em = correct / max(total, 1)
    logger.info("GSM8K fallback: exact_match=%.4f (%d/%d)", em, correct, total)
    return {"exact_match": em, "num_samples": total, "num_fewshot": num_fewshot}


def _extract_final_number(text: str) -> float | None:
    """Extract the last number from text (GSM8K convention: #### <number>)."""
    # First try #### pattern
    match = re.search(r"####\s*([\d,.-]+)", text)
    if match:
        try:
            return float(match.group(1).replace(",", ""))
        except ValueError:
            pass
    # Fallback: last number in text
    numbers = re.findall(r"-?[\d,]+\.?\d*", text)
    if numbers:
        try:
            return float(numbers[-1].replace(",", ""))
        except ValueError:
            pass
    return None


def evaluate_arc_fallback(
    model: torch.nn.Module,
    tokenizer: Any,
    device: str = "cuda:0",
    num_fewshot: int = 25,
    split: str = "ARC-Challenge",
) -> dict[str, Any]:
    """Simple ARC-Challenge 25-shot MC evaluation (fallback)."""
    try:
        from datasets import load_dataset
        ds = load_dataset("allenai/ai2_arc", split, split="test")
    except Exception as exc:
        logger.warning("Could not load ARC: %s", exc)
        return {"error": str(exc)}

    # Few-shot examples from train
    shots: list[dict] = []
    if num_fewshot > 0:
        try:
            train_ds = load_dataset("allenai/ai2_arc", split, split="train")
            shots = [train_ds[i] for i in range(min(num_fewshot, len(train_ds)))]
        except Exception:
            pass

    correct = 0
    total = 0

    for sample in ds:
        choices = sample["choices"]
        labels = choices["label"]    # e.g. ["A", "B", "C", "D"]
        texts = choices["text"]      # e.g. ["choice1", "choice2", ...]
        answer_key = sample["answerKey"]  # e.g. "A"

        # Build prompt
        prompt = ""
        for shot in shots:
            sc = shot["choices"]
            prompt += _format_arc_question(
                shot["question"], sc["label"], sc["text"],
            )
            prompt += f" {shot['answerKey']}\n\n"

        prompt += _format_arc_question(sample["question"], labels, texts)

        # Score each label
        scores = _score_choices(model, tokenizer, prompt, labels, device)
        pred_idx = scores.index(max(scores))

        if labels[pred_idx] == answer_key:
            correct += 1
        total += 1

    accuracy = correct / max(total, 1)
    logger.info("ARC fallback: accuracy=%.4f (%d/%d)", accuracy, correct, total)
    return {"accuracy": accuracy, "num_samples": total, "num_fewshot": num_fewshot}


def _format_arc_question(question: str, labels: list[str], texts: list[str]) -> str:
    """Format a single ARC question."""
    result = f"Question: {question}\n"
    for label, text in zip(labels, texts):
        result += f"{label}. {text}\n"
    result += "Answer:"
    return result


def evaluate_fallback(
    model: torch.nn.Module,
    tokenizer: Any,
    benchmarks: list[str],
    num_fewshot: dict[str, int],
    device: str = "cuda:0",
) -> dict[str, dict[str, Any]]:
    """Run built-in fallback evaluators for supported benchmarks."""
    results: dict[str, dict[str, Any]] = {}

    for bench in benchmarks:
        n_shot = num_fewshot.get(bench, 0)
        logger.info("Fallback eval: %s (%d-shot)...", bench, n_shot)
        t0 = time.time()

        if bench == "mmlu":
            res = evaluate_mmlu_fallback(model, tokenizer, device, n_shot)
        elif bench == "gsm8k":
            res = evaluate_gsm8k_fallback(model, tokenizer, device, n_shot)
        elif bench in ("arc_challenge", "arc_easy"):
            arc_split = "ARC-Challenge" if bench == "arc_challenge" else "ARC-Easy"
            res = evaluate_arc_fallback(model, tokenizer, device, n_shot, arc_split)
        else:
            logger.warning(
                "  %s: no fallback evaluator available, skipping", bench,
            )
            results[bench] = {"error": "no fallback evaluator"}
            continue

        res["time_sec"] = round(time.time() - t0, 1)
        results[bench] = res
        score = res.get("accuracy") or res.get("exact_match")
        logger.info("  %s = %.4f (%.1f s)", bench, score or 0.0, res["time_sec"])

    return results


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
        description="Evaluate distilled student on LLM benchmarks",
    )
    ap.add_argument(
        "--model_path", default=None,
        help="Path to student checkpoint (.pt). Omit to evaluate base model.",
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
    ap.add_argument(
        "--fallback", action="store_true",
        help="Force fallback mode (built-in evaluators, no lm-eval)",
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

    # Decide mode
    use_fallback = args.fallback or not _lm_eval_available()
    if use_fallback and not args.fallback:
        logger.warning(
            "lm-eval-harness not found, falling back to built-in evaluators. "
            "Install lm-eval for full benchmark support: pip install lm-eval>=0.4",
        )

    if use_fallback:
        # Fallback: load model ourselves and run custom evaluators
        model, tokenizer = _load_model(
            args.model_name, args.model_path, args.device,
        )
        results = evaluate_fallback(
            model, tokenizer, benchmarks, num_fewshot, args.device,
        )
    else:
        # Primary: use lm-eval-harness CLI
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
