#!/usr/bin/env bash
set -euo pipefail

cd /notebooks/storage/project/ocgan-modernized
source .venv/bin/activate

CATEGORIES=(
  bottle cable capsule carpet grid hazelnut leather metal_nut
  pill screw tile toothbrush transistor wood zipper
)

BASE_CONFIG="experiments/final_mvtec_t1d_m1d_lf1c_oc0"

LOG_DIR="logs/big_grid_seed_memory_teacher_fusion_maxpatch"
STATE_DIR="logs/big_grid_seed_memory_teacher_fusion_maxpatch_state"

mkdir -p "${LOG_DIR}"
mkdir -p "${STATE_DIR}"

PARALLEL_JOBS=4

run_one() {
  local category="$1"
  local seed="$2"
  local t_value="$3"
  local m_value="$4"
  local lf_c="$5"
  local max_patches="$6"

  local t_label
  local m_label
  local lf_label
  local mp_label

  case "${t_value}" in
    0.05) t_label="t0x" ;;
    0.1)  t_label="t1a" ;;
    0.2)  t_label="t1b" ;;
    0.4)  t_label="t1d" ;;
    *) echo "Unknown teacher value ${t_value}" ; return 1 ;;
  esac

  case "${m_value}" in
    0.02) m_label="m0x" ;;
    0.1)  m_label="m1b" ;;
    0.4)  m_label="m1d" ;;
    0.6)  m_label="m1f" ;;
    *) echo "Unknown memory value ${m_value}" ; return 1 ;;
  esac

  case "${lf_c}" in
    1.0) lf_label="lf1b" ;;
    2.0) lf_label="lf1c" ;;
    4.0) lf_label="lf1d" ;;
    8.0) lf_label="lf1g" ;;
    *) echo "Unknown learned fusion C ${lf_c}" ; return 1 ;;
  esac

  case "${max_patches}" in
    1024) mp_label="mp1a" ;;
    2048) mp_label="mp1b" ;;
    4096) mp_label="mp1c" ;;
    *) echo "Unknown max_patches value ${max_patches}" ; return 1 ;;
  esac

  local run_name="${category}_${t_label}_${m_label}_${lf_label}_${mp_label}_oc0_s${seed}"
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
    project.seed="${seed}" \
    project.experiment_name="${run_name}" \
    teacher_student.score_weight="${t_value}" \
    memory_bank.score_weight="${m_value}" \
    memory_bank.max_patches="${max_patches}" \
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

show_progress() {
  local total="$1"
  while true; do
    local done_count
    done_count=$(find "${STATE_DIR}" -name "*.done" | wc -l | tr -d ' ')

    python - "$done_count" "$total" <<'PY'
import sys
done_count = int(sys.argv[1])
total = int(sys.argv[2])
width = 40
ratio = 0 if total == 0 else done_count / total
filled = int(width * ratio)
bar = "#" * filled + "-" * (width - filled)
print(f"\r[{bar}] {done_count}/{total} ({ratio*100:5.1f}%)", end="", flush=True)
PY

    if [[ "${done_count}" -ge "${total}" ]]; then
      echo
      break
    fi
    sleep 5
  done
}

export -f run_one
export BASE_CONFIG
export LOG_DIR
export STATE_DIR

TMP_TASKS="$(mktemp)"

for category in "${CATEGORIES[@]}"; do
  for seed in 43 44 45 46; do
    for t_value in 0.05 0.1 0.2 0.4; do
      for m_value in 0.02 0.1 0.4 0.6; do
        for lf_c in 1.0 2.0 4.0 8.0; do
          for max_patches in 1024 2048 4096; do
            echo "${category}|${seed}|${t_value}|${m_value}|${lf_c}|${max_patches}" >> "${TMP_TASKS}"
          done
        done
      done
    done
  done
done

TOTAL_TASKS="$(wc -l < "${TMP_TASKS}")"
export TOTAL_TASKS
echo "Total tasks: ${TOTAL_TASKS}"

show_progress "${TOTAL_TASKS}" &
PROGRESS_PID=$!

cleanup() {
  kill "${PROGRESS_PID}" 2>/dev/null || true
}
trap cleanup EXIT

nl -w1 -s'|' "${TMP_TASKS}" | xargs -P "${PARALLEL_JOBS}" -I {} bash -c '
  IFS="|" read -r idx category seed t_value m_value lf_c max_patches <<< "$1"
  echo "[${idx}/${TOTAL_TASKS}] category=${category} seed=${seed} t=${t_value} m=${m_value} lfC=${lf_c} max_patches=${max_patches}"
  run_one "$category" "$seed" "$t_value" "$m_value" "$lf_c" "$max_patches"
' _ {}

wait "${PROGRESS_PID}" || true
rm -f "${TMP_TASKS}"