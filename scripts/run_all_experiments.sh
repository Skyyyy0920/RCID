#!/usr/bin/env bash
# =================================================================
# Paper experiments: 4-GPU parallel execution
#
# Part 1: Main comparison table (6 experiments)
# Part 2: Beta ablation (4 experiments)
# Part 3: Generate figures
#
# Usage:
#   bash scripts/run_all_experiments.sh
#   OUTPUT_DIR=outputs/paper bash scripts/run_all_experiments.sh
# =================================================================
set -euo pipefail
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

OUTPUT_DIR="${OUTPUT_DIR:-outputs/paper}"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
LOG_DIR="${OUTPUT_DIR}/logs"
mkdir -p "$LOG_DIR"

run_exp() {
    local exp_name="$1"
    local gpu_id="$2"
    local log_file="${LOG_DIR}/${exp_name}.log"
    echo "[GPU ${gpu_id}] Starting: ${exp_name}"
    CUDA_VISIBLE_DEVICES="$gpu_id" python "${SCRIPT_DIR}/run_paper_experiments.py" \
        --experiment "$exp_name" \
        --device "cuda:0" \
        --output_dir "$OUTPUT_DIR" \
        > "$log_file" 2>&1
    local status=$?
    if [ $status -eq 0 ]; then
        echo "[GPU ${gpu_id}] Done: ${exp_name}"
    else
        echo "[GPU ${gpu_id}] FAILED: ${exp_name} (see ${log_file})"
    fi
    return $status
}

echo "=== KL-Ratio Paper Experiments ==="
echo "Output: $OUTPUT_DIR"
echo ""

# ============================================================
# Part 1: Main Comparison Table
# ============================================================
echo "===== Part 1: Main Comparison (6 experiments) ====="

# Round 1: 4 in parallel
run_exp forward_kl   0 &
run_exp reverse_kl   1 &
run_exp jeffreys     2 &
run_exp akl          3 &
wait
echo "Round 1 done"

# Round 2: 2 in parallel
run_exp klr_token     0 &
run_exp klr_batch_ema 1 &
wait
echo "Round 2 done"
echo ""

# ============================================================
# Part 2: Beta Ablation (4 experiments)
# ============================================================
echo "===== Part 2: Beta Ablation (4 experiments) ====="

run_exp klr_no_ema     0 &
run_exp klr_beta_0.9   1 &
run_exp klr_beta_0.95  2 &
run_exp klr_beta_0.999 3 &
wait
echo "Beta ablation done"
echo ""

# ============================================================
# Part 3: Summary + Figures
# ============================================================
echo "===== Part 3: Summary + Figures ====="
python "${SCRIPT_DIR}/plot_paper_figures.py" --output_dir "$OUTPUT_DIR" || true

# Print summary table (ROUGE-L on Dolly test split)
echo ""
echo "===== Results Summary (ROUGE-L on Dolly test) ====="
python -c "
import json, glob, os

results = {}
for f in sorted(glob.glob('${OUTPUT_DIR}/*/eval_results.json')):
    name = os.path.basename(os.path.dirname(f))
    with open(f) as fh:
        data = json.load(fh)
    results[name] = data

print('Method              | ROUGE-L F | ROUGE-L P | ROUGE-L R')
print('-' * 60)
order = ['forward_kl', 'reverse_kl', 'jeffreys', 'akl',
         'klr_token', 'klr_batch_ema',
         'klr_no_ema', 'klr_beta_0.9', 'klr_beta_0.95', 'klr_beta_0.999']
for name in order:
    r = results.get(name, {})
    f1 = r.get('rouge_l_f', 0)
    p = r.get('rouge_l_p', 0)
    rec = r.get('rouge_l_r', 0)
    print(f'{name:20s}| {f1*100:9.2f} | {p*100:9.2f} | {rec*100:9.2f}')
" || echo "(summary failed, check individual eval_results.json)"

echo ""
echo "===== ALL DONE ====="
