#!/usr/bin/env bash
# =================================================================
# Phase 1: PADD quick validation (single A100, ~2-3 days)
#
# Experiments:
#   1. standard_kd (baseline)               ~3-4 h
#   2. PADD tau=0.5                          ~3-4 h
#   3. PADD tau=1.0                          ~3-4 h
#   4. PADD tau=2.0                          ~3-4 h
#   5. PADD tau=5.0                          ~3-4 h
#   6. pure reverse KL (ablation)            ~3-4 h
#   7. fixed JSD (ablation)                  ~3-4 h
#
# + lm-eval per model (MMLU, GSM8K, ARC-C)  ~30 min each
#
# Go/no-go: any tau beats standard_kd on any benchmark → Phase 2
# =================================================================
set -euo pipefail

DEVICE="${DEVICE:-cuda:0}"
SEED="${SEED:-42}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/padd}"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

run() {
    echo ""
    echo "============================================================"
    echo "  $*"
    echo "============================================================"
    python "${SCRIPT_DIR}/run_padd_distill.py" "$@" \
        --device "$DEVICE" --seed "$SEED" --output_dir "$OUTPUT_DIR"
}

# ── 1. Baseline ────────────────────────────────────────────────────
run --method standard_kd

# ── 2-5. PADD tau sweep ───────────────────────────────────────────
for TAU in 0.5 1.0 2.0 5.0; do
    run --method standard_kd_padd --tau "$TAU"
done

# ── 6. Pure reverse KL (ablation) ─────────────────────────────────
run --method pure_reverse_kl --tau 1.0

# ── 7. Fixed JSD (ablation) ───────────────────────────────────────
run --method fixed_jsd --tau 1.0

echo ""
echo "============================================================"
echo "  Phase 1 complete!  Results in: $OUTPUT_DIR"
echo "============================================================"
