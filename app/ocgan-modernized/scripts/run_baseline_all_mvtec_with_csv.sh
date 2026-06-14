#!/usr/bin/env bash
set -euo pipefail

cd /notebooks/storage/project/ocgan-modernized
source .venv/bin/activate

CATEGORIES=(
  bottle cable capsule carpet grid hazelnut leather metal_nut
  pill screw tile toothbrush transistor wood zipper
)

CONFIG_NAME="experiments/final_mvtec_t1d_m1d_lf1c_oc0"
LOG_DIR="logs/baseline_all_mvtec_with_csv"
mkdir -p "${LOG_DIR}"

run_one() {
  local category="$1"
  local log_file="${LOG_DIR}/${category}.log"

  echo "===== RUNNING ${category} =====" | tee "${log_file}"
  python scripts/train.py \
    --config-path ../configs \
    --config-name "${CONFIG_NAME}" \
    dataset.category="${category}" \
    2>&1 | tee -a "${log_file}"
}

export -f run_one
export CONFIG_NAME
export LOG_DIR

printf "%s\n" "${CATEGORIES[@]}" | xargs -n 1 -P 4 -I {} bash -c 'run_one "$@"' _ {}