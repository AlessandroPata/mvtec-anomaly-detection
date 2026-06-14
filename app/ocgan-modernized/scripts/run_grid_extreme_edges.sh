#!/usr/bin/env bash
set -euo pipefail

cd /notebooks/storage/project/ocgan-modernized
source .venv/bin/activate

BASE_CONFIG="experiments/final_mvtec_t1d_m1d_lf1c_oc0"

# stesse cartelle di prima, così non rifà le run già finite
LOG_DIR="logs/grid_resume_all_mvtec_light"
STATE_DIR="logs/grid_resume_all_mvtec_light_state"

mkdir -p "${LOG_DIR}"
mkdir -p "${STATE_DIR}"

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
    0.02) t_label="t00a" ;;
    0.03) t_label="t00b" ;;
    0.05) t_label="t0x" ;;
    0.08) t_label="t0y" ;;
    0.1)  t_label="t1a" ;;
    0.2)  t_label="t1b" ;;
    0.3)  t_label="t1c" ;;
    0.4)  t_label="t1d" ;;
    0.5)  t_label="t1e" ;;
    0.6)  t_label="t1f" ;;
    0.7)  t_label="t1g" ;;
    *) echo "Unknown teacher value ${t_value}" ; return 1 ;;
  esac

  case "${m_value}" in
    0.005) m_label="m00a" ;;
    0.01)  m_label="m00b" ;;
    0.02)  m_label="m0x" ;;
    0.03)  m_label="m0y" ;;
    0.05)  m_label="m1a" ;;
    0.1)   m_label="m1b" ;;
    0.2)   m_label="m1c" ;;
    0.4)   m_label="m1d" ;;
    0.5)   m_label="m1e" ;;
    0.6)   m_label="m1f" ;;
    0.7)   m_label="m1g" ;;
    0.8)   m_label="m1h" ;;
    1.0)   m_label="m1i" ;;
    *) echo "Unknown memory value ${m_value}" ; return 1 ;;
  esac

  case "${lf_c}" in
    0.5)  lf_label="lf0x" ;;
    0.8)  lf_label="lf0y" ;;
    1.0)  lf_label="lf1b" ;;
    2.0)  lf_label="lf1c" ;;
    4.0)  lf_label="lf1d" ;;
    5.0)  lf_label="lf1e" ;;
    6.0)  lf_label="lf1f" ;;
    8.0)  lf_label="lf1g" ;;
    10.0) lf_label="lf1h" ;;
    12.0) lf_label="lf1i" ;;
    *) echo "Unknown learned fusion C ${lf_c}" ; return 1 ;;
  esac

  local run_name="${category}_${t_label}_${m_label}_${lf_label}_oc0_s${SEED}"
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
    one_class.score_weight="0.0" \
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
export BASE_CONFIG
export LOG_DIR
export STATE_DIR
export SEED

TMP_TASKS="$(mktemp)"

# hazelnut: best su teacher min e memory min
for t_value in 0.02 0.03 0.05; do
  for m_value in 0.005 0.01 0.02; do
    for lf_c in 1.0; do
      echo "hazelnut|${t_value}|${m_value}|${lf_c}" >> "${TMP_TASKS}"
    done
  done
done

# screw: best su teacher min e memory min
for t_value in 0.02 0.03 0.05; do
  for m_value in 0.005 0.01 0.02; do
    for lf_c in 4.0; do
      echo "screw|${t_value}|${m_value}|${lf_c}" >> "${TMP_TASKS}"
    done
  done
done

# wood: best su teacher min
for t_value in 0.02 0.03 0.05; do
  for m_value in 0.1; do
    for lf_c in 4.0; do
      echo "wood|${t_value}|${m_value}|${lf_c}" >> "${TMP_TASKS}"
    done
  done
done

# transistor: best su memory alto e learned alto
for t_value in 0.4; do
  for m_value in 0.6 0.8 1.0; do
    for lf_c in 8.0 10.0 12.0; do
      echo "transistor|${t_value}|${m_value}|${lf_c}" >> "${TMP_TASKS}"
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