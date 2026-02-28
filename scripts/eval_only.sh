#!/usr/bin/env bash
# =================================================================
# Evaluate existing checkpoints without retraining.
#
# Usage:
#   # Evaluate a single experiment
#   bash scripts/eval_only.sh forward_kl cuda:0
#
#   # Evaluate all experiments in outputs/paper/
#   bash scripts/eval_only.sh --all cuda:0
#
#   # Custom output dir
#   OUTPUT_DIR=outputs/phase2 bash scripts/eval_only.sh klr_batch_ema cuda:0
# =================================================================
set -euo pipefail

OUTPUT_DIR="${OUTPUT_DIR:-outputs/paper}"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

if [ "${1:-}" = "--all" ]; then
    DEVICE="${2:-cuda:0}"
    echo "Evaluating all experiments in ${OUTPUT_DIR}..."
    for dir in "${OUTPUT_DIR}"/*/; do
        exp_name="$(basename "$dir")"
        ckpt="${dir}student_final.pt"
        if [ -f "$ckpt" ]; then
            echo ""
            echo "=== Evaluating: ${exp_name} ==="
            CUDA_VISIBLE_DEVICES="${DEVICE#cuda:}" python "${SCRIPT_DIR}/run_paper_experiments.py" \
                --experiment "$exp_name" \
                --device "cuda:0" \
                --output_dir "$OUTPUT_DIR" \
                --eval_only \
                || echo "FAILED: ${exp_name}"
        else
            echo "Skipping ${exp_name} (no checkpoint)"
        fi
    done
else
    EXP_NAME="${1:?Usage: eval_only.sh <experiment_name|--all> [device]}"
    DEVICE="${2:-cuda:0}"

    echo "Evaluating: ${EXP_NAME} on ${DEVICE}"
    python "${SCRIPT_DIR}/run_paper_experiments.py" \
        --experiment "$EXP_NAME" \
        --device "$DEVICE" \
        --output_dir "$OUTPUT_DIR" \
        --eval_only
fi

echo ""
echo "Done."
