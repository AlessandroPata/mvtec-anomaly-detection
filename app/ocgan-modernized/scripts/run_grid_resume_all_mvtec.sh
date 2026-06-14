#!/usr/bin/env bash
set -euo pipefail

cd /notebooks/storage/project/ocgan-modernized
source .venv/bin/activate

CATEGORIES=(
  bottle cable capsule carpet grid hazelnut leather metal_nut
  pill screw tile toothbrush transistor wood zipper
)

BASE_CONFIG="experiments/final_mvtec_t1d_m1d_lf1c_oc0"
OUTPUT_ROOT="/notebooks/storage/outputs/ocgan-modernized"
LOG_DIR="logs/grid_resume_all_mvtec"
STATE_DIR="logs/grid_resume_all_mvtec_state"

mkdir -p "${LOG_DIR}"
mkdir -p "${STATE_DIR}"

# puoi cambiare qui
SEED=43
PARALLEL_JOBS=4

run_one() {
  local category="$1"
  local t_value="$2"
  local m_value="$3"
  local lf_c="$4"

  local t_label
  local m_label
  local lf_label

  case "${t_value}" in
    0.1) t_label="t1a" ;;
    0.2) t_label="t1b" ;;
    0.3) t_label="t1c" ;;
    0.4) t_label="t1d" ;;
    0.5) t_label="t1e" ;;
    *) echo "Unknown teacher value ${t_value}" ; return 1 ;;
  esac

  case "${m_value}" in
    0.05) m_label="m1a" ;;
    0.1)  m_label="m1b" ;;
    0.2)  m_label="m1c" ;;
    0.4)  m_label="m1d" ;;
    0.5)  m_label="m1e" ;;
    *) echo "Unknown memory value ${m_value}" ; return 1 ;;
  esac

  case "${lf_c}" in
    1.0) lf_label="lf1b" ;;
    2.0) lf_label="lf1c" ;;
    4.0) lf_label="lf1d" ;;
    5.0) lf_label="lf1e" ;;
    *) echo "Unknown learned fusion C ${lf_c}" ; return 1 ;;
  esac

  local oc_label="oc1"
  local oc_value="1.0"

  local run_name="${category}_${t_label}_${m_label}_${lf_label}_${oc_label}_s${SEED}"
  local log_file="${LOG_DIR}/${run_name}.log"
  local run_dir="${OUTPUT_ROOT}/${run_name}"
  local done_file="${STATE_DIR}/${run_name}.done"

  # skip se già completata
  if [[ -f "${done_file}" ]]; then
    echo "===== SKIP DONE ${run_name} =====" | tee -a "${log_file}"
    return 0
  fi

  # skip anche se il run_dir contiene già metriche finali
  if [[ -f "${run_dir}/test_blind_component_scores.csv" ]] || [[ -f "${run_dir}/selected_threshold.txt" ]]; then
    echo "===== MARK DONE FROM OUTPUT ${run_name} =====" | tee -a "${log_file}"
    touch "${done_file}"
    return 0
  fi

  echo "===== RUNNING ${run_name} =====" | tee "${log_file}"

  local resume_flag="false"
  local resume_path="null"

  if [[ -f "${run_dir}/last_checkpoint.pt" ]]; then
    resume_flag="true"
    resume_path="${run_dir}/last_checkpoint.pt"
    echo "[RESUME] found checkpoint: ${resume_path}" | tee -a "${log_file}"
  fi

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
    training.resume="${resume_flag}" \
    training.resume_path="${resume_path}" \
    2>&1 | tee -a "${log_file}"

  # marca done solo se ci sono evidenze di completamento
  if grep -q "Training finished." "${log_file}"; then
    touch "${done_file}"
  fi
}

export -f run_one
export BASE_CONFIG
export OUTPUT_ROOT
export LOG_DIR
export STATE_DIR
export SEED

TMP_TASKS="$(mktemp)"

for category in "${CATEGORIES[@]}"; do
  for t_value in 0.1 0.2 0.3 0.4 0.5; do
    for m_value in 0.05 0.1 0.2 0.4 0.5; do
      for lf_c in 1.0 2.0 4.0 5.0; do
        echo "${category}|${t_value}|${m_value}|${lf_c}" >> "${TMP_TASKS}"
      done
    done
  done
done

cat "${TMP_TASKS}" | xargs -P "${PARALLEL_JOBS}" -I {} bash -c '
  IFS="|" read -r category t_value m_value lf_c <<< "$1"
  run_one "$category" "$t_value" "$m_value" "$lf_c"
' _ {}

rm -f "${TMP_TASKS}"