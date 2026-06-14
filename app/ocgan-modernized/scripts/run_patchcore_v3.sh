#!/usr/bin/env bash
# PatchCore v3 — full memory bank (max_patches=70000 bypasses k-center for all cats).
# 15 cats × 3 seeds, ~10s/run -> ~8 min total.
set -euo pipefail

cd /notebooks/storage_project_outputs_datasets/project/ocgan-modernized

python -c "import hydra, pytorch_msssim, lpips" 2>/dev/null || \
  pip install --quiet --no-input hydra-core pytorch_msssim "numpy<2" \
    "opencv-python-headless<4.12" albumentations einops lpips

LOG_DIR="logs/patchcore_v3"
STATE_DIR="logs/patchcore_v3_state"
mkdir -p "${LOG_DIR}" "${STATE_DIR}"

CATEGORIES=(bottle cable capsule carpet grid hazelnut leather metal_nut pill screw tile toothbrush transistor wood zipper)
SEEDS=(43 44 45)

export PATCHCORE_CSV="logs/patchcore_v3.csv"
[ -f "${PATCHCORE_CSV}" ] && mv "${PATCHCORE_CSV}" "${PATCHCORE_CSV}.bak.$(date +%s)"

run_one() {
  local cat="$1" seed="$2"
  local run="${cat}_pcv3_s${seed}"
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
    memory_bank.max_patches=70000 \
    ++memory_bank.candidate_pool_size=80000 \
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
echo "=== PatchCore v3 DONE ==="
python3 - <<'PY'
import csv, statistics
from collections import defaultdict

def load(f):
    d = {}
    try:
        for r in csv.DictReader(open(f)):
            d.setdefault(r["category"], []).append(float(r["auroc"]))
    except: pass
    return d

v1 = load("logs/patchcore_pure.csv")
v2 = load("logs/patchcore_v2.csv")
v3 = load("logs/patchcore_v3.csv")

v3_grid = {"bottle":0.9938,"cable":0.7804,"capsule":0.9318,"carpet":0.9905,"grid":0.9906,
           "hazelnut":1.0000,"leather":0.9497,"metal_nut":0.8317,"pill":0.8548,
           "screw":1.0000,"tile":1.0000,"toothbrush":0.9111,"transistor":0.8783,
           "wood":1.0000,"zipper":0.9510}

cats = sorted(v3)
print(f"{'cat':<14} {'v1':>7} {'v2':>7} {'v3':>7} {'Δv2→v3':>8} {'v3grid':>8} {'best':>8}")
print("-"*65)
all_v3, all_best = [], []
for cat in cats:
    m1 = statistics.mean(v1.get(cat,[0]))
    m2 = statistics.mean(v2.get(cat,[0]))
    m3 = statistics.mean(v3.get(cat,[0]))
    vg = v3_grid.get(cat,0)
    best = max(m3, vg)
    all_v3.append(m3)
    all_best.append(best)
    d = m3 - m2
    sign = "+" if d >= 0 else ""
    beat = " ★" if m3 > vg else ""
    print(f"{cat:<14} {m1:>7.4f} {m2:>7.4f} {m3:>7.4f} {sign}{d:.4f}   {vg:>8.4f} {best:>8.4f}{beat}")
print("-"*65)
print(f"{'MACRO v1':<14} {statistics.mean([statistics.mean(v) for v in v1.values()]):>7.4f}")
print(f"{'MACRO v2':<14} {' ':>7} {statistics.mean([statistics.mean(v) for v in v2.values()]):>7.4f}")
print(f"{'MACRO v3':<14} {' ':>7} {' ':>7} {statistics.mean(all_v3):>7.4f}")
print(f"{'MACRO best':<14} {' ':>57} {statistics.mean(all_best):>8.4f}")
PY
