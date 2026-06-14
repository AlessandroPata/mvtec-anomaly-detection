#!/usr/bin/env bash
set -euo pipefail

cd /notebooks/storage/project/ocgan-modernized
source .venv/bin/activate

LOG_DIR="logs/final_per_category_multiseed"
STATE_DIR="logs/final_per_category_multiseed_state"

mkdir -p "${LOG_DIR}"
mkdir -p "${STATE_DIR}"

PARALLEL_JOBS=4

run_one() {
  local category="$1"
  local seed="$2"

  local run_name="${category}_final_s${seed}"
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
export LOG_DIR
export STATE_DIR

TMP_TASKS="$(mktemp)"

# 3 seed
for category in \
  bottle capsule carpet grid hazelnut leather screw tile toothbrush wood zipper
do
  for seed in 43 44 45; do
    echo "${category}|${seed}" >> "${TMP_TASKS}"
  done
done

# 5 seed
for category in cable metal_nut pill transistor; do
  for seed in 43 44 45 46 47; do
    echo "${category}|${seed}" >> "${TMP_TASKS}"
  done
done

TOTAL_TASKS="$(wc -l < "${TMP_TASKS}")"
export TOTAL_TASKS
echo "Total tasks: ${TOTAL_TASKS}"

nl -w1 -s'|' "${TMP_TASKS}" | xargs -P "${PARALLEL_JOBS}" -I {} bash -c '
  IFS="|" read -r idx category seed <<< "$1"
  echo "[${idx}/${TOTAL_TASKS}] category=${category} seed=${seed}"
  run_one "$category" "$seed"
' _ {}

rm -f "${TMP_TASKS}"