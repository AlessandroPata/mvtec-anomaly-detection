#!/usr/bin/env bash
# Sprint 4b — PatchCore v2: layer2+layer3, topk_reweighted, k=9, coreset=10000.
# 15 cats × 3 seeds, ~40s/run -> ~30 min total.
set -euo pipefail

cd /notebooks/storage_project_outputs_datasets/project/ocgan-modernized

python -c "import hydra, pytorch_msssim, lpips" 2>/dev/null || \
  pip install --quiet --no-input hydra-core pytorch_msssim "numpy<2" \
    "opencv-python-headless<4.12" albumentations einops lpips

LOG_DIR="logs/patchcore_v2"
STATE_DIR="logs/patchcore_v2_state"
mkdir -p "${LOG_DIR}" "${STATE_DIR}"

CATEGORIES=(bottle cable capsule carpet grid hazelnut leather metal_nut pill screw tile toothbrush transistor wood zipper)
SEEDS=(43 44 45)

export PATCHCORE_CSV="logs/patchcore_v2.csv"
[ -f "${PATCHCORE_CSV}" ] && mv "${PATCHCORE_CSV}" "${PATCHCORE_CSV}.bak.$(date +%s)"

run_one() {
  local cat="$1" seed="$2"
  local run="${cat}_pcv2_s${seed}"
  local log="${LOG_DIR}/${run}.log"
  local done="${STATE_DIR}/${run}.done"
  [[ -f "${done}" ]] && { echo "SKIP ${run}"; return 0; }
  echo "RUN ${run}"
  python scripts/patchcore_pure.py \
    --config-path ../configs \
    --config-name "experiments/final_per_category/${cat}" \
    project.seed="${seed}" \
    project.experiment_name="${run}" \
    model.backbone.name=wide_resnet50_2 \
    memory_bank.enabled=true \
    "memory_bank.feature_level=layer2+layer3" \
    memory_bank.aggregation=topk_reweighted \
    memory_bank.topk=9 \
    memory_bank.max_patches=10000 \
    ++memory_bank.candidate_pool_size=20000 \
    hydra.job.chdir=false \
    > "${log}" 2>&1
  if grep -q "Training finished." "${log}"; then
    touch "${done}"
    grep "\[Test\]" "${log}"
  else
    echo "  FAIL ${run}"; tail -10 "${log}"
  fi
}

total=$(( ${#CATEGORIES[@]} * ${#SEEDS[@]} ))
i=0
for cat in "${CATEGORIES[@]}"; do
  for seed in "${SEEDS[@]}"; do
    i=$((i+1))
    echo "[${i}/${total}]"
    run_one "${cat}" "${seed}"
  done
done

echo
echo "=== PatchCore v2 DONE ==="
python3 - <<'PY'
import csv, statistics
from collections import defaultdict

v1 = {}
for r in csv.DictReader(open("logs/patchcore_pure.csv")):
    v1.setdefault(r["category"], []).append(float(r["auroc"]))

v2 = {}
for r in csv.DictReader(open("logs/patchcore_v2.csv")):
    v2.setdefault(r["category"], []).append(float(r["auroc"]))

v3_best = {"bottle":0.9938,"cable":0.7804,"capsule":0.9318,"carpet":0.9905,"grid":0.9906,
           "hazelnut":1.0000,"leather":0.9497,"metal_nut":0.8317,"pill":0.8548,
           "screw":1.0000,"tile":1.0000,"toothbrush":0.9111,"transistor":0.8783,
           "wood":1.0000,"zipper":0.9510}

cats = sorted(v2)
print(f"{'cat':<14} {'v1_mean':>8} {'v2_mean':>8} {'delta':>7} {'v3_best':>8} {'best_all':>9}")
print("-"*60)
all_best = []
for cat in cats:
    m1 = statistics.mean(v1.get(cat, [0]))
    m2 = statistics.mean(v2.get(cat, [0]))
    v3 = v3_best.get(cat, 0)
    best = max(m1, m2, v3)
    all_best.append(best)
    d = m2 - m1
    sign = "+" if d >= 0 else ""
    print(f"{cat:<14} {m1:>8.4f} {m2:>8.4f} {sign}{d:.4f}  {v3:>8.4f} {best:>9.4f}")
print("-"*60)
m1_macro = statistics.mean([statistics.mean(v) for v in v1.values()])
m2_macro = statistics.mean([statistics.mean(v) for v in v2.values()])
print(f"{'MACRO v1':<14} {m1_macro:>8.4f}")
print(f"{'MACRO v2':<14} {m2_macro:>8.4f}  ({m2_macro-m1_macro:+.4f} vs v1)")
print(f"{'MACRO best':<14} {' ':>8} {' ':>8} {' ':>7} {' ':>8} {statistics.mean(all_best):>9.4f}")
PY
