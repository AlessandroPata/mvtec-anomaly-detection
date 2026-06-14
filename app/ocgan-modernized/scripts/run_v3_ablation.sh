#!/usr/bin/env bash
# v3 ablation driver. Same 3 cat x 2 seed of smoke, toggles one fix at a time.
# Usage: ./scripts/run_v3_ablation.sh {noUnfreeze|noUnet|noTopk}
set -euo pipefail
cd /notebooks/storage_project_outputs_datasets/project/ocgan-modernized

VARIANT="${1:-noUnfreeze}"
LOG_DIR="logs/v3_abl_${VARIANT}"
STATE_DIR="logs/v3_abl_${VARIANT}_state"
mkdir -p "${LOG_DIR}" "${STATE_DIR}"

CATEGORIES=(bottle capsule metal_nut)
SEEDS=(43 44)
PARALLEL_JOBS=2

# Defaults = v3 full config
USE_SKIP=true
UNFREEZE=layer3
TOPK=100

case "${VARIANT}" in
  noUnfreeze) UNFREEZE=none ;;
  noUnet)     USE_SKIP=false ;;
  noTopk)     TOPK=0 ;;
  *) echo "unknown variant: ${VARIANT}" >&2; exit 2 ;;
esac

run_one() {
  local category="$1" seed="$2"
  local run_name="${category}_${VARIANT}_s${seed}"
  local log_file="${LOG_DIR}/${run_name}.log"
  local done_file="${STATE_DIR}/${run_name}.done"
  [[ -f "${done_file}" ]] && { echo "SKIP ${run_name}"; return 0; }
  echo "===== RUN ${run_name} =====" | tee "${log_file}"
  python scripts/train.py \
    --config-path ../configs \
    --config-name "experiments/final_per_category/${category}" \
    project.seed="${seed}" \
    project.experiment_name="${run_name}" \
    project.deterministic=true \
    model.reconstruction.use_skip_connections=${USE_SKIP} \
    model.backbone.unfreeze_from=${UNFREEZE} \
    model.backbone.unfreeze_lr_factor=0.1 \
    scoring_topk=${TOPK} \
    dataset.val_mixed_ratio=0.25 \
    logging.save_debug_images=false \
    logging.save_score_histograms=false \
    analysis.save_failure_analysis=false \
    checkpoint.save_last=false \
    checkpoint.save_best=false \
    training.resume=false 2>&1 | tee -a "${log_file}"
  grep -q "Training finished." "${log_file}" && touch "${done_file}"
}
export -f run_one
export LOG_DIR STATE_DIR VARIANT USE_SKIP UNFREEZE TOPK

TMP="$(mktemp)"
for c in "${CATEGORIES[@]}"; do for s in "${SEEDS[@]}"; do echo "${c}|${s}" >> "${TMP}"; done; done
echo "=== ablation ${VARIANT}: $(wc -l < "${TMP}") runs | USE_SKIP=${USE_SKIP} UNFREEZE=${UNFREEZE} TOPK=${TOPK} ==="
if command -v parallel >/dev/null 2>&1; then
  parallel --jobs "${PARALLEL_JOBS}" --colsep '\|' run_one {1} {2} :::: "${TMP}"
else
  while IFS='|' read -r c s; do run_one "$c" "$s"; done < "${TMP}"
fi
rm -f "${TMP}"
echo "=== ablation ${VARIANT} DONE ==="
