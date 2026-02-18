"""Orchestrator: run multiple experiments with GPU assignment."""

from __future__ import annotations

import argparse
import logging
import os
import subprocess
import sys
from pathlib import Path

logger = logging.getLogger(__name__)

SCRIPT_DIR = Path(__file__).resolve().parent

EXPERIMENT_SCRIPTS = {
    1: SCRIPT_DIR / "run_exp1.py",
    2: SCRIPT_DIR / "run_exp2.py",
    3: SCRIPT_DIR / "run_exp3.py",
    4: SCRIPT_DIR / "run_exp4.py",
    5: SCRIPT_DIR / "run_exp5_cross_arch.py",
}


def _build_cmd(
    exp: int,
    model_family: str,
    gpu: int,
    extra_args: list[str],
) -> list[str]:
    """Build subprocess command for a single experiment."""
    script = str(EXPERIMENT_SCRIPTS[exp])
    cmd = [sys.executable, script]

    # exp5 is always llama3; others accept --model_family
    if exp != 5:
        cmd += ["--model_family", model_family]

    cmd += ["--device", f"cuda:{gpu}"]
    cmd += extra_args
    return cmd


def run_experiments(
    experiments: list[int],
    model_family: str,
    gpus: list[int],
    extra_args: list[str],
    parallel: bool = False,
) -> None:
    """Run selected experiments, optionally in parallel across GPUs."""
    if parallel and len(gpus) > 1:
        _run_parallel(experiments, model_family, gpus, extra_args)
    else:
        gpu = gpus[0] if gpus else 0
        for exp in experiments:
            cmd = _build_cmd(exp, model_family, gpu, extra_args)
            logger.info("Running: %s", " ".join(cmd))
            env = os.environ.copy()
            env["CUDA_VISIBLE_DEVICES"] = str(gpu)
            result = subprocess.run(cmd, env=env)
            if result.returncode != 0:
                logger.error("Experiment %d failed (exit %d)", exp, result.returncode)
            else:
                logger.info("Experiment %d completed", exp)


def _run_parallel(
    experiments: list[int],
    model_family: str,
    gpus: list[int],
    extra_args: list[str],
) -> None:
    """Run experiments in parallel, assigning round-robin to GPUs."""
    processes: list[tuple[int, subprocess.Popen]] = []  # type: ignore[type-arg]

    for i, exp in enumerate(experiments):
        gpu = gpus[i % len(gpus)]
        cmd = _build_cmd(exp, model_family, gpu, extra_args)
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(gpu)
        logger.info("Launching exp%d on GPU %d: %s", exp, gpu, " ".join(cmd))
        proc = subprocess.Popen(cmd, env=env)
        processes.append((exp, proc))

    # Wait for all
    for exp, proc in processes:
        retcode = proc.wait()
        if retcode != 0:
            logger.error("Experiment %d failed (exit %d)", exp, retcode)
        else:
            logger.info("Experiment %d completed", exp)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run RCID experiments")
    parser.add_argument(
        "--exp", type=str, default="1,2,3,4,5",
        help="Comma-separated experiment numbers (default: 1,2,3,4,5)",
    )
    parser.add_argument(
        "--model_family", type=str, default="qwen3",
        choices=["qwen3", "llama3", "both"],
    )
    parser.add_argument(
        "--gpu", type=str, default="0",
        help="Comma-separated GPU ids (default: 0)",
    )
    parser.add_argument("--parallel", action="store_true")
    parser.add_argument("--skip_existing", action="store_true")

    args, extra = parser.parse_known_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    experiments = [int(x.strip()) for x in args.exp.split(",")]
    gpus = [int(x.strip()) for x in args.gpu.split(",")]
    extra_args = list(extra)
    if args.skip_existing:
        extra_args.append("--skip_existing")

    families = ["qwen3", "llama3"] if args.model_family == "both" else [args.model_family]

    for family in families:
        logger.info("=== Model family: %s ===", family)
        run_experiments(experiments, family, gpus, extra_args, args.parallel)


if __name__ == "__main__":
    main()
