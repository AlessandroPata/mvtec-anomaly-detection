#!/usr/bin/env bash
# Sprint 4b — targeted PatchCore tuning on weak categories.
# 4 cats × 4 combos × 1 seed (~5 min total); winners run × 3 seeds.
set -euo pipefail

cd /notebooks/storage_project_outputs_datasets/project/ocgan-modernized

python -c "import hydra, pytorch_msssim, lpips" 2>/dev/null || \
  pip install --quiet --no-input hydra-core pytorch_msssim "numpy<2" \
    "opencv-python-headless<4.12" albumentations einops lpips

LOG_DIR="logs/patchcore_tuning"
STATE_DIR="logs/patchcore_tuning_state"
mkdir -p "${LOG_DIR}" "${STATE_DIR}"

export PATCHCORE_CSV="logs/patchcore_tuning.csv"
# Don't reset — allow incremental appending across runs

SEED=43
# Weak cats: pill, zipper, capsule, screw
WEAK_CATS=(pill zipper capsule screw)

# Grid: feature_level × aggregation × topk
# Baseline (already have): layer2+layer3 / topk_mean / k=3
# Variants to try:
#   A: layer3     / topk_mean      / k=3
#   B: layer3     / topk_mean      / k=9
#   C: layer2+layer3 / topk_reweighted / k=9
#   D: layer3     / topk_reweighted / k=9

declare -A COMBOS
COMBOS["A"]="layer3 topk_mean 3"
COMBOS["B"]="layer3 topk_mean 9"
COMBOS["C"]="layer2+layer3 topk_reweighted 9"
COMBOS["D"]="layer3 topk_reweighted 9"

run_one() {
  local cat="$1" combo_id="$2"
  read -r flevel agg k <<< "${COMBOS[$combo_id]}"
  local run="${cat}_tune_${combo_id}_s${SEED}"
  local log="${LOG_DIR}/${run}.log"
  local done="${STATE_DIR}/${run}.done"
  if [[ -f "${done}" ]]; then
    echo "SKIP DONE ${run}"
    return 0
  fi
  echo "RUN ${run} (level=${flevel} agg=${agg} k=${k})"
  python scripts/patchcore_pure.py \
    --config-path ../configs \
    --config-name "experiments/final_per_category/${cat}" \
    project.seed="${SEED}" \
    project.experiment_name="${run}" \
    model.backbone.name=wide_resnet50_2 \
    memory_bank.enabled=true \
    "memory_bank.feature_level=${flevel}" \
    memory_bank.aggregation="${agg}" \
    memory_bank.topk="${k}" \
    memory_bank.max_patches=10000 \
    ++memory_bank.candidate_pool_size=20000 \
    hydra.job.chdir=false \
    > "${log}" 2>&1
  if grep -q "Training finished." "${log}"; then
    touch "${done}"
    grep "\[Test\]" "${log}"
  else
    echo "  FAIL ${run}; tail:"
    tail -10 "${log}"
  fi
}

echo "=== Sprint 4b: PatchCore tuning for weak categories ==="
total=$(( ${#WEAK_CATS[@]} * ${#COMBOS[@]} ))
i=0
for cat in "${WEAK_CATS[@]}"; do
  for combo_id in "${!COMBOS[@]}"; do
    i=$((i+1))
    echo "[${i}/${total}]"
    run_one "${cat}" "${combo_id}"
  done
done

echo
echo "=== TUNING RESULTS (vs baseline layer2+layer3/topk_mean/k=3/s43) ==="
# Baseline AUROC from patchcore_pure.csv for seed 43
python3 - <<'PY'
import csv
from collections import defaultdict

baseline_s43 = {}
for row in csv.DictReader(open("logs/patchcore_pure.csv")):
    if int(row["seed"]) == 43:
        baseline_s43[row["category"]] = float(row["auroc"])

tuning = list(csv.DictReader(open("logs/patchcore_tuning.csv"))) if __import__("os").path.exists("logs/patchcore_tuning.csv") else []

cats = ["pill","zipper","capsule","screw"]
print(f"{'cat':<10} {'combo':>6} {'feature_level':<16} {'agg':<18} {'k':>3}  {'auroc':>7}  {'vs_base':>8}")
print("-"*75)
for row in sorted(tuning, key=lambda r: (r["category"], r.get("aggregation",""))):
    if row["category"] not in cats: continue
    cat = row["category"]
    base = baseline_s43.get(cat, 0)
    auroc = float(row["auroc"])
    delta = auroc - base
    sign = "+" if delta >= 0 else ""
    run_name = row.get("category","") # won't have combo id, use aggregation+level
    print(f"{cat:<10} {'?':>6} {row.get('feature_level','?'):<16} {row['aggregation']:<18} {row['topk']:>3}  {auroc:>7.4f}  {sign}{delta:.4f}")
PY
