#!/usr/bin/env bash
set -euo pipefail

cd /notebooks/storage/project/ocgan-modernized
source .venv/bin/activate

#CATEGORIES=(
#  bottle cable capsule carpet grid hazelnut leather metal_nut
# pill screw tile toothbrush transistor wood zipper
#)
CATEGORIES=(
  bottle 
)
BASE_CONFIG="experiments/final_mvtec_t1d_m1d_lf1c_oc0"
LOG_DIR="logs/grid_resume_all_mvtec_light"
STATE_DIR="logs/grid_resume_all_mvtec_light_state"

mkdir -p "${LOG_DIR}"
mkdir -p "${STATE_DIR}"

SEED=43
PARALLEL_JOBS=4

# Grid aggiornata
TEACHER_VALUES=( 0.02 0.05 0.08 0.1 )
MEMORY_VALUES=( 0.2 )
LEARNED_VALUES=( 5.0 6.0 7.0 8.0 )

sanitize_value() {
  echo "$1" | sed 's/\./p/g'
}

run_one() {
  local category="$1"
  local t_value="$2"
  local m_value="$3"
  local lf_c="$4"

  local t_label="t$(sanitize_value "${t_value}")"
  local m_label="m$(sanitize_value "${m_value}")"
  local lf_label="lf$(sanitize_value "${lf_c}")"

  local oc_label="oc0"
  local oc_value="0.0"

  local run_name="${category}_${t_label}_${m_label}_${lf_label}_${oc_label}_s${SEED}"
  local log_file="${LOG_DIR}/${run_name}.log"
  local done_file="${STATE_DIR}/${run_name}.done"

  if [[ -f "${done_file}" ]]; then
    echo "===== SKIP DONE ${run_name} =====" | tee -a "${log_file}"
    return 0
  fi

  echo "===== RUNNING ${run_name} =====" | tee "${log_file}"

  python scripts/train.py \
    --config-path ../configs \
    --config-name "${BASE_CONFIG}" \
    dataset.category="${category}" \
    project.seed="${SEED}" \
    project.experiment_name="${run_name}" \
    teacher_student.score_weight="${t_value}" \
    memory_bank.score_weight="${m_value}" \
    score_fusion_learned.enabled=true \
    score_fusion_learned.C="${lf_c}" \
    one_class.score_weight="${oc_value}" \
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
export -f sanitize_value
export BASE_CONFIG
export LOG_DIR
export STATE_DIR
export SEED

TMP_TASKS="$(mktemp)"

for category in "${CATEGORIES[@]}"; do
  for t_value in "${TEACHER_VALUES[@]}"; do
    for m_value in "${MEMORY_VALUES[@]}"; do
      for lf_c in "${LEARNED_VALUES[@]}"; do
        echo "${category}|${t_value}|${m_value}|${lf_c}" >> "${TMP_TASKS}"
      done
    done
  done
done

TOTAL_TASKS="$(wc -l < "${TMP_TASKS}")"
export TOTAL_TASKS
echo "Total tasks: ${TOTAL_TASKS}"

nl -w1 -s'|' "${TMP_TASKS}" | xargs -P "${PARALLEL_JOBS}" -I {} bash -c '
  IFS="|" read -r idx category t_value m_value lf_c <<< "$1"
  echo "[${idx}/${TOTAL_TASKS}] category=${category} t=${t_value} m=${m_value} lfC=${lf_c}"
  run_one "$category" "$t_value" "$m_value" "$lf_c"
' _ {}

rm -f "${TMP_TASKS}"