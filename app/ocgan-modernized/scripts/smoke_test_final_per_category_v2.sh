#!/usr/bin/env bash
set -euo pipefail

cd /notebooks/storage/project/ocgan-modernized
source .venv/bin/activate

CATEGORIES=(
  bottle cable capsule carpet grid hazelnut leather metal_nut
  pill screw tile toothbrush transistor wood zipper
)

LOG_DIR="logs/smoke_test_final_per_category_v2"
mkdir -p "${LOG_DIR}"

for category in "${CATEGORIES[@]}"; do
  echo "===== SMOKE TEST ${category} ====="

  python scripts/train.py \
    --config-path ../configs \
    --config-name "experiments/final_per_category_v2/${category}" \
    training.epochs=1 \
    dataset.train_normal.length=8 \
    dataset.val_normal.length=8 \
    dataset.val_mixed.length=8 \
    dataset.test_blind.length=8 \
    logging.save_debug_images=false \
    logging.save_score_histograms=false \
    analysis.save_failure_analysis=false \
    checkpoint.save_last=false \
    checkpoint.save_best=false \
    runtime.detect_nan=false \
    training.resume=false \
    2>&1 | tee "${LOG_DIR}/${category}.log"
done