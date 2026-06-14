#!/usr/bin/env bash
# Sprint 2: PatchCore-grade memory bank
# Changes vs v3/v3b:
#   - memory_bank.feature_level=layer2+layer3  (concat 512+1024=1536ch @ 16x16)
#   - memory_bank.max_train_batches=-1          (full train split)
#   - memory_bank.max_patches=10000             (coreset 10k)
#   - memory_bank.aggregation=topk_reweighted   (PatchCore top-k reweighted)
#   - memory_bank.topk=3
#   - model.backbone.unfreeze_from: per-category (capsule/transistor=none, others=layer3)
# Baseline: composite v3+v3b = 0.7666
set -euo pipefail

# Ensure deps are present (Paperspace evicts packages between sessions)
python -c "import hydra" 2>/dev/null || pip install hydra-core --quiet
python -c "import pytorch_msssim" 2>/dev/null || pip install pytorch-msssim --quiet
python -c "import hydra, pytorch_msssim; print('[deps] OK')"

cd /notebooks/storage_project_outputs_datasets/project/ocgan-modernized

LOG_DIR="logs/v4_sprint2_membank"
STATE_DIR="logs/v4_sprint2_membank_state"
CATEGORIES=(bottle cable capsule carpet grid hazelnut leather metal_nut pill screw tile toothbrush transistor wood zipper)
SEEDS=(43 44 45 46 47)
PARALLEL_JOBS=2

mkdir -p "${LOG_DIR}" "${STATE_DIR}"

# Categories where frozen backbone (unfreeze_from=none) was better in v3b
FROZEN_CATS="capsule transistor"

run_one() {
  local category="$1"
  local seed="$2"
  local run_name="${category}_v4_s${seed}"
  local log_file="${LOG_DIR}/${run_name}.log"
  local done_file="${STATE_DIR}/${run_name}.done"

  if [[ -f "${done_file}" ]]; then
    echo "===== SKIP DONE ${run_name} ====="
    return 0
  fi

  # Use frozen backbone for capsule/transistor (v3b showed improvement)
  local unfreeze_from="layer3"
  for frozen_cat in ${FROZEN_CATS}; do
    if [[ "${category}" == "${frozen_cat}" ]]; then
      unfreeze_from="none"
      break
    fi
  done

  echo "===== RUNNING ${run_name} (unfreeze=${unfreeze_from}) =====" | tee "${log_file}"

  python scripts/train.py \
    --config-path ../configs \
    --config-name "experiments/final_per_category/${category}" \
    project.seed="${seed}" \
    project.experiment_name="${run_name}" \
    project.deterministic=true \
    model.reconstruction.use_skip_connections=true \
    model.backbone.unfreeze_from="${unfreeze_from}" \
    scoring_topk=100 \
    dataset.val_mixed_ratio=0.25 \
    memory_bank.enabled=true \
    "memory_bank.feature_level=layer2+layer3" \
    memory_bank.max_train_batches=-1 \
    memory_bank.max_patches=10000 \
    memory_bank.aggregation=topk_reweighted \
    memory_bank.topk=3 \
    ++memory_bank.candidate_pool_size=20000 \
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
export LOG_DIR STATE_DIR FROZEN_CATS

TMP_TASKS="$(mktemp)"
for category in "${CATEGORIES[@]}"; do
  for seed in "${SEEDS[@]}"; do
    echo "${category}|${seed}" >> "${TMP_TASKS}"
  done
done

echo "=== Sprint 2 membank: $(wc -l < "${TMP_TASKS}") runs, parallel=${PARALLEL_JOBS} ==="

if command -v parallel >/dev/null 2>&1; then
  parallel --jobs "${PARALLEL_JOBS}" --colsep '\|' run_one {1} {2} :::: "${TMP_TASKS}"
else
  while IFS='|' read -r cat seed; do
    run_one "${cat}" "${seed}"
  done < "${TMP_TASKS}"
fi

rm -f "${TMP_TASKS}"
echo "=== Sprint 2 membank DONE ==="
