#!/usr/bin/env bash
# =================================================================
# Phase 2: AKL vs KL-Ratio vs Jeffreys
#
# ~3-4 hours per experiment, 5 experiments = 15-20 hours total
# Single A100 80 GB
#
# Experiments:
#   1. forward_kl       — standard forward KL (baseline)
#   2. jeffreys         — fixed 0.5 FKL + 0.5 RKL
#   3. akl_mu0.5        — AKL (Wu et al., COLING 2025)
#   4. klr_token        — KL-Ratio per-token (ours)
#   5. klr_batch_ema    — KL-Ratio batch + EMA (ours)
# =================================================================
set -euo pipefail

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

DEVICE="${1:-cuda:0}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/phase2}"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

echo "=== Phase 2 Experiments ==="
echo "Device: $DEVICE"
echo "Output: $OUTPUT_DIR"
echo ""

for EXP in forward_kl jeffreys akl_mu0.5 klr_token klr_batch_ema; do
    echo ""
    echo "========================================="
    echo "  Running experiment: $EXP"
    echo "========================================="

    python "${SCRIPT_DIR}/run_phase2_experiments.py" \
        --experiments "$EXP" \
        --device "$DEVICE" \
        --output_dir "$OUTPUT_DIR"

    echo "Completed: $EXP"
done

echo ""
echo "=== All experiments complete ==="
echo ""

# ── Summary table ─────────────────────────────────────────────────
python -c "
import json, glob, os

results = {}
for f in sorted(glob.glob('${OUTPUT_DIR}/*/eval_results.json')):
    name = os.path.basename(os.path.dirname(f))
    with open(f) as fh:
        data = json.load(fh)
    results[name] = data.get('results', {})

print('Method              | GSM8K  | MMLU   | ARC-C  | Avg')
print('-' * 60)
for name in ['forward_kl', 'jeffreys', 'akl_mu0.5', 'klr_token', 'klr_batch_ema']:
    r = results.get(name, {})
    gsm = r.get('gsm8k', {}).get('exact_match', r.get('gsm8k', {}).get('accuracy', 0))
    mmlu = r.get('mmlu', {}).get('accuracy', 0)
    arc = r.get('arc_challenge', {}).get('accuracy', 0)
    avg = (gsm + mmlu + arc) / 3 if (gsm + mmlu + arc) > 0 else 0
    print(f'{name:20s}| {gsm*100:6.2f} | {mmlu*100:6.2f} | {arc*100:6.2f} | {avg*100:.2f}')
"
