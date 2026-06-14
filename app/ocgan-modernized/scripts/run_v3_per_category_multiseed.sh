#!/usr/bin/env bash
# v3 retrain: UNet skip + selective unfreeze layer3 + scoring_topk=100
# + Perlin vectorised/seeded + val_mixed_ratio=0.25 + deterministic.
# Flag attivati via CLI override (no config fork) per diff pulito v2->v3.
#
# Usage:
#   ./scripts/run_v3_per_category_multiseed.sh smoke   # 3 cat x 2 seed
#   ./scripts/run_v3_per_category_multiseed.sh full    # 15 cat x 5 seed
set -euo pipefail

cd /notebooks/storage_project_outputs_datasets/project/ocgan-modernized

MODE="${1:-smoke}"

if [[ "${MODE}" == "smoke" ]]; then
  LOG_DIR="logs/v3_smoke"
  STATE_DIR="logs/v3_smoke_state"
  CATEGORIES=(bottle capsule metal_nut)
  SEEDS=(43 44)
  PARALLEL_JOBS=2
elif [[ "${MODE}" == "full" ]]; then
  LOG_DIR="logs/v3_per_category_multiseed"
  STATE_DIR="logs/v3_per_category_multiseed_state"
  CATEGORIES=(bottle cable capsule carpet grid hazelnut leather metal_nut pill screw tile toothbrush transistor wood zipper)
  SEEDS=(43 44 45 46 47)
  PARALLEL_JOBS=2
else
  echo "Usage: $0 {smoke|full}" >&2
  exit 2
fi

mkdir -p "${LOG_DIR}" "${STATE_DIR}"

run_one() {
  local category="$1"
  local seed="$2"

  local run_name="${category}_v3_s${seed}"
  local log_file="${LOG_DIR}/${run_name}.log"
  local done_file="${STATE_DIR}/${run_name}.done"

  if [[ -f "${done_file}" ]]; then
    echo "===== SKIP DONE ${run_name} =====" | tee -a "${log_file}"
    return 0
  fi

  echo "===== RUNNING ${run_name} =====" | tee "${log_file}"

  python scripts/train.py \
    --config-path ../configs \
    --config-name "experiments/final_per_category/${category}" \
    project.seed="${seed}" \
    project.experiment_name="${run_name}" \
    project.deterministic=true \
    model.reconstruction.use_skip_connections=true \
    model.backbone.unfreeze_from=layer3 \
    model.backbone.unfreeze_lr_factor=0.1 \
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

echo "=== v3 ${MODE}: $(wc -l < "${TMP_TASKS}") runs, parallel=${PARALLEL_JOBS} ==="

if command -v parallel >/dev/null 2>&1; then
  parallel --jobs "${PARALLEL_JOBS}" --colsep '\|' run_one {1} {2} :::: "${TMP_TASKS}"
else
  # Fallback sequenziale se GNU parallel non c'e'
  while IFS='|' read -r cat seed; do
    run_one "${cat}" "${seed}"
  done < "${TMP_TASKS}"
fi

rm -f "${TMP_TASKS}"
echo "=== v3 ${MODE} DONE ==="
