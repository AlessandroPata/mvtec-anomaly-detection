#!/usr/bin/env bash
# Sprint 4 — PatchCore-pure on wide_resnet50_2.
# 15 cats × 3 seeds, ~40s/run -> ~30 min total (sequential, single GPU).
# Output CSV in logs/patchcore_pure.csv (appended).
set -euo pipefail

cd /notebooks/storage_project_outputs_datasets/project/ocgan-modernized

# Reinstall evicted deps (Paperspace)
python -c "import hydra, pytorch_msssim, lpips" 2>/dev/null || \
  pip install --quiet --no-input hydra-core pytorch_msssim "numpy<2" \
    "opencv-python-headless<4.12" albumentations einops lpips

LOG_DIR="logs/patchcore_pure"
STATE_DIR="logs/patchcore_pure_state"
mkdir -p "${LOG_DIR}" "${STATE_DIR}"

CATEGORIES=(bottle cable capsule carpet grid hazelnut leather metal_nut pill screw tile toothbrush transistor wood zipper)
SEEDS=(43 44 45)

# Reset CSV at start of full run
export PATCHCORE_CSV="logs/patchcore_pure.csv"
[ -f "${PATCHCORE_CSV}" ] && mv "${PATCHCORE_CSV}" "${PATCHCORE_CSV}.bak.$(date +%s)"

run_one() {
  local cat="$1"
  local seed="$2"
  local run="${cat}_pcpure_s${seed}"
  local log="${LOG_DIR}/${run}.log"
  local done="${STATE_DIR}/${run}.done"
  if [[ -f "${done}" ]]; then
    echo "SKIP DONE ${run}"
    return 0
  fi
  echo "RUN ${run}"
  python scripts/patchcore_pure.py \
    --config-path ../configs \
    --config-name "experiments/final_per_category/${cat}" \
    project.seed="${seed}" \
    project.experiment_name="${run}" \
    model.backbone.name=wide_resnet50_2 \
    memory_bank.enabled=true \
    "memory_bank.feature_level=layer2+layer3" \
    memory_bank.aggregation=topk_mean \
    memory_bank.topk=3 \
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

total=$(( ${#CATEGORIES[@]} * ${#SEEDS[@]} ))
i=0
for cat in "${CATEGORIES[@]}"; do
  for seed in "${SEEDS[@]}"; do
    i=$((i + 1))
    echo "[${i}/${total}]"
    run_one "${cat}" "${seed}"
  done
done

echo
echo "=== PatchCore-pure full run DONE ==="
python - <<'PY'
import csv
from collections import defaultdict
import statistics
rows = list(csv.DictReader(open("logs/patchcore_pure.csv")))
by_cat = defaultdict(list)
for r in rows:
    by_cat[r["category"]].append(float(r["auroc"]))
print(f"{'cat':<12} {'mean':>7} {'std':>6} {'n':>3}")
print("-"*32)
all_means = []
for cat in sorted(by_cat):
    aurocs = by_cat[cat]
    m = statistics.mean(aurocs)
    s = statistics.stdev(aurocs) if len(aurocs) > 1 else 0.0
    all_means.append(m)
    print(f"{cat:<12} {m:>7.4f} {s:>6.4f} {len(aurocs):>3}")
print("-"*32)
print(f"{'MACRO':<12} {statistics.mean(all_means):>7.4f}")
PY
