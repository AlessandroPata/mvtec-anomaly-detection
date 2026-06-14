#!/usr/bin/env bash
# Ablation v3a: solo scoring_topk + determinism + Perlin seeded (baseline onesto).
# No UNet skip, no backbone unfreeze. Serve a isolare se le skip UNet causano
# instabilità numerica (NaN/Inf osservati in ablation noUnfreeze).
set -euo pipefail

cd /notebooks/storage_project_outputs_datasets/project/ocgan-modernized

LOG_DIR="logs/v3a_abl_noUnet"
STATE_DIR="logs/v3a_abl_noUnet_state"
CATEGORIES=(bottle capsule metal_nut)
SEEDS=(43 44)
PARALLEL_JOBS=2

mkdir -p "${LOG_DIR}" "${STATE_DIR}"

run_one() {
  local category="$1"
  local seed="$2"
  local run_name="${category}_v3a_s${seed}"
  local log_file="${LOG_DIR}/${run_name}.log"
  local done_file="${STATE_DIR}/${run_name}.done"

  if [[ -f "${done_file}" ]]; then
    echo "===== SKIP DONE ${run_name} ====="
    return 0
  fi

  echo "===== RUNNING ${run_name} =====" | tee "${log_file}"

  python scripts/train.py \
    --config-path ../configs \
    --config-name "experiments/final_per_category/${category}" \
    project.seed="${seed}" \
    project.experiment_name="${run_name}" \
    project.deterministic=true \
    model.reconstruction.use_skip_connections=false \
    model.backbone.unfreeze_from=none \
    scoring_topk=100 \
    dataset.val_mixed_ratio=0.25 \
    logging.save_debug_images=false \
    logging.save_score_histograms=false \
    analysis.save_failure_analysis=false \
    checkpoint.save_last=false \
    checkpoint.save_best=false \
    runtime.detect_nan=false \
    training.resume=false \
    2>&1 | tee -a "${log_file}"

  if grep -q "Training finished." "${log_file}"; then
    touch "${done_file}"
  fi
}

export -f run_one
export LOG_DIR STATE_DIR

TMP_TASKS="$(mktemp)"
for category in "${CATEGORIES[@]}"; do
  for seed in "${SEEDS[@]}"; do
    echo "${category}|${seed}" >> "${TMP_TASKS}"
  done
done

echo "=== v3a ablation: $(wc -l < "${TMP_TASKS}") runs, parallel=${PARALLEL_JOBS} ==="

if command -v parallel >/dev/null 2>&1; then
  parallel --jobs "${PARALLEL_JOBS}" --colsep '\|' run_one {1} {2} :::: "${TMP_TASKS}"
else
  while IFS='|' read -r cat seed; do
    run_one "${cat}" "${seed}"
  done < "${TMP_TASKS}"
fi

rm -f "${TMP_TASKS}"
echo "=== v3a ablation DONE ==="
